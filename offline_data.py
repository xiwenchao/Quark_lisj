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


def offline_data(ref_policy: Policy, 
                 score_model: Reward, 
                 dataloader: DataLoader,
                 tree_tokens: List[str],
                 n_extra_tokens: int,
                 top_p: float = 1.0,
                 epoch: str = 'offline'):
    """
    Generate offline dataset from prompts using ref_policy and score_model.
    
    Args:
        ref_policy: Reference policy (GPT2) to generate responses
        score_model: Reward model to score the responses
        dataloader: DataLoader containing prompts
        tree_tokens: List of tree tokens for categorization
        n_extra_tokens: Number of reward categories
        top_p: Top-p sampling parameter
        epoch: Identifier for the current epoch/step
        
    Returns:
        Dictionary containing prompts, responses, scores, and cat_tokens
    """
    log.info(f"Generating offline data ...")


    train_responses_path = os.path.join('data/toxicity', 'train_offline_response.json')

    prompts, responses = [], []
    
    # Generate responses for all prompts
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader),
                                desc='Generating offline data from ref_policy')):
        input_ids, attention_mask = batch
        
        # Sample from reference policy (similar to step 0 in ConditionTrainer.sample)
        rollouts = ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=top_p)
        prompt, response = rollouts['query/text'], rollouts['response/text']
        
        prompts.extend(prompt)
        responses.extend(response)
    
    print(f'Generated {len(prompts)} responses')

    original_train_data = {
        'prompts': prompts,
        'responses': responses
    }

    with open(train_responses_path, 'w') as f:
        json.dump(original_train_data, f, indent=2)
    log.info(f'Saved train original_train_data data to {train_responses_path}')

    # Get rewards/scores for the generated responses
    scores = score_model.get_reward(prompts, responses, epoch)

    # Sort data by scores (same logic as DataPool.add lines 17-20)
    data = zip(prompts, responses, scores)
    data = [x for x in data if x[-1] is not None]
    sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
    prompts, responses, scores = [list(x) for x in list(zip(*sorted_data))]

    # Assign category tokens based on score ranking (same logic as DataPool.add lines 22-24)
    cat_pos = [[i] * (len(sorted_data) // n_extra_tokens) for i in range(n_extra_tokens)]
    cat_pos = [y for x in cat_pos for y in x]
    cat_pos = cat_pos + [n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
    cat_tokens = [tree_tokens[i] for i in cat_pos]
    
    log.info(f"Generated {len(prompts)} samples with offline data")
    
    return {
        'prompts': prompts,
        'responses': responses,
        'scores': scores,
        'cat_tokens': cat_tokens
    }

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

    args.output_dir = "offline_data" + args.output_dir
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
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

    train_responses_path = os.path.join('data/toxicity', 'train_offline_response.json')

    if os.path.exists(train_responses_path):
        log.info(f'Found existing offline data at {train_responses_path}, loading ...')
        with open(train_responses_path, 'r') as f:
            original_train_data = json.load(f)
        prompts = original_train_data['prompts']
        responses = original_train_data['responses']

        reward = Reward(save_path=args.reward_dir, rate_limit=args.perspective_rate_limit, batch_size=args.batch_size)
        scores = reward.get_reward(prompts, responses, 'offline')

        # Sort data by scores (same logic as DataPool.add lines 17-20)
        data = zip(prompts, responses, scores)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
        prompts, responses, scores = [list(x) for x in list(zip(*sorted_data))]

        # Assign category tokens based on score ranking (same logic as DataPool.add lines 22-24)
        cat_pos = [[i] * (len(sorted_data) // args.n_extra_tokens) for i in range(args.n_extra_tokens)]
        cat_pos = [y for x in cat_pos for y in x]
        cat_pos = cat_pos + [args.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
        cat_tokens = [tree_tokens[i] for i in cat_pos]
        
        log.info(f"Generated {len(prompts)} samples with offline data")

        train_data = {
            'prompts': prompts,
            'responses': responses,
            'scores': scores,
            'cat_tokens': cat_tokens
        }

    else:

        log.info(f'Initializing models ...')
        ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device)
        reward = Reward(save_path=args.reward_dir, rate_limit=args.perspective_rate_limit, batch_size=args.batch_size)
        log.info(f'Initialization done!')

        prompt_collator = PromptCollator(tokenizer=ref_policy.tokenizer)
        train_dataset = PromptDataset(path=args.dataset_train)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=prompt_collator)
        log.info(f'Load train set with {len(train_dataset)} examples')

        # Generate offline data for training set
        train_data = offline_data(
            ref_policy=ref_policy, 
            score_model=reward,
            dataloader=train_dataloader,
            tree_tokens=tree_tokens,
            n_extra_tokens=args.n_extra_tokens,
            top_p=args.top_p,
            epoch='train_offline'
        )

    
    # Save the generated offline data
    train_output_path = os.path.join('data/toxicity', 'train_offline.json')

    with open(train_output_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    log.info(f'Saved train offline data to {train_output_path}')


if __name__ == "__main__":
    main()
