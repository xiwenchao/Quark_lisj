import os
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from arguments import get_args
from policy import Policy
from data_pool import DataPool
from reward import Reward, reward_to_toxicity
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, path):
        self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }


class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['query'] for sequence in sequences]
        responses = [sequence['response'] for sequence in sequences]
        # at_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        # query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
        # query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask


class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult


class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 data_pool: DataPool,
                 score_model: Reward,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.writer = SummaryWriter()

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer)
        
        # Create training dataloader from data pool
        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=True, collate_fn=self.seq_collator)
        log.info(f"Loaded {len(sample_dataset)} examples from data pool for training")

    def add_control_code(self, input_ids, attention_mask):
        input_ids = torch.cat([input_ids.new([self.best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids, attention_mask

    def decode(self, query_input_ids, response_input_ids=None):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for p in query_input_ids]

        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        return query, response

    def train_epoch(self, epoch_num):
        """Train for one complete epoch over the dataset."""
        epoch_started_at = time.time()
        log.info(f"[Epoch {epoch_num}] Starting training...")
        
        self.policy.model.train()
        epoch_stats = {'loss/total': 0.0, 'loss/lm': 0.0, 'loss/kl': 0.0, 'loss/entropy': 0.0,
                      'objective/kl': 0.0, 'objective/entropy': 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(self.sample_dataloader, desc=f"Epoch {epoch_num}")):
            step_started_at = time.time()
            
            self.optimizer.zero_grad()
            ppo_loss, stats = self.loss(epoch_num, batch_idx, *batch)
            ppo_loss.backward()
            if self.params.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate stats (using running sum instead of list)
            for key in epoch_stats:
                epoch_stats[key] += stats[key]
            num_batches += 1

            # Update controllers
            self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
            self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)
            
            # Clear CUDA cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Log batch-level metrics at intervals
            global_step = epoch_num * len(self.sample_dataloader) + batch_idx
            if batch_idx % self.params.log_interval == 0:
                step_time = time.time() - step_started_at
                eps_per_second = float(self.params.batch_size) / step_time
                log.info(f"[Epoch {epoch_num}, Batch {batch_idx}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
                
                for metric in ['kl', 'entropy']:
                    self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], global_step)
                for metric in ['lm', 'kl', 'entropy', 'total']:
                    self.writer.add_scalar(f'Loss/{metric}', stats[f'loss/{metric}'], global_step)
                self.writer.add_scalar(f'Params/lr', self.optimizer.param_groups[0]['lr'], global_step)
                self.writer.add_scalar(f'Params/kl_coef', self.kl_ctl.value, global_step)
                self.writer.add_scalar(f'Params/entropy_coef', self.entropy_ctl.value, global_step)

        # Log epoch-level statistics
        epoch_time = time.time() - epoch_started_at
        log.info(f"[Epoch {epoch_num}] Completed in {epoch_time:.2f}s")
        for key in epoch_stats:
            mean_value = epoch_stats[key] / num_batches if num_batches > 0 else 0.0
            log.info(f"[Epoch {epoch_num}] {key}: {mean_value:.4f}")
            self.writer.add_scalar(f'Epoch/{key}', mean_value, epoch_num)
        
        self.save(epoch=epoch_num)
        self.eval(epoch=epoch_num)

    def loss(self, epoch, batch_idx, query_input_ids, query_mask, response_input_ids, response_mask):
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        logits = outputs['response/logits'][:, :, :-len(self.tree_tokens)]
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(query_input_ids, query_mask,
                                                       response_input_ids, response_mask)
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)

        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        # queries, responses = self.decode(query_input_ids, response_input_ids)
        # self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1), logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, epoch=epoch, batch_idx=batch_idx)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, epoch, batch_idx):
        if batch_idx % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, epoch):
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, f'{self.params.model_dir}/ckp_epoch_{epoch}.pth')
        log.info(f"[Epoch {epoch}] model checkpoint saved")

    def eval(self, epoch):
        log.info(f"[Epoch {epoch}] evaluating ...")

        generations, perplexities, toxicities = [], [], []
        for i, (input_ids, attention_mask) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, 1:],
                                  'query_mask': rollouts['query/mask'][:, 1:],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt = self.decode(rollouts['query/input_ids'][:, 1:])
                response = rollouts['response/text']
                score = self.score_model.get_reward(prompt, response, f'epoch{epoch}_eval{i}')
                toxicity = [reward_to_toxicity(x) for x in score if x is not None]
                toxicities.extend(toxicity)

                generations.extend(rollouts['response/text'])

        ppl_score, toxicity_score = np.mean(perplexities), np.mean(toxicities)
        dist_1, dist_2, dist_3 = distinctness(generations)
        print(f"  perplexity = {ppl_score:+.2f}")
        print(f"  toxicity = {toxicity_score:+.2f}")
        print(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}')
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, epoch)
        self.writer.add_scalar('Evaluation/toxicity', toxicity_score, epoch)
        self.writer.add_scalar('Evaluation/Dist-1', dist_1, epoch)
        self.writer.add_scalar('Evaluation/Dist-2', dist_2, epoch)
        self.writer.add_scalar('Evaluation/Dist-3', dist_3, epoch)


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    args.output_dir = "offline_sft_" + args.output_dir
    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
                  [' _TREE_TOKEN_ZERO_COMMENTS']
    
    data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens)
    data_pool.load_data(data_path=args.dataset_pool)

    log.info(f'Initializing models ...')
    ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device)
    policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device,
                    reward_cond=True, tree_tokens=tree_tokens)
    reward = Reward(save_path=args.reward_dir, rate_limit=args.perspective_rate_limit, batch_size=args.batch_size)
    log.info(f'Initialization done!')

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    train_dataset = PromptDataset(path=args.dataset_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(path=args.dataset_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-5)
    
    # Calculate total training steps based on epochs
    num_batches_per_epoch = len(data_pool.get_data()[0]) // args.batch_size
    args.total_steps = args.num_epochs * num_batches_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler)

    log.info(f"Starting training for {args.num_epochs} epochs...")
    for epoch_num in range(args.num_epochs):
        try:
            trainer.train_epoch(epoch_num)
        except RuntimeError as e:
            log.error(f"RuntimeError during epoch {epoch_num}: {e}")
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    main()
