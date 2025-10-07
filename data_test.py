import json
import os


train_output_path = os.path.join('data/toxicity', 'train_offline.json')

# read offline data
with open(train_output_path, 'r') as f:
    original_train_data = json.load(f)
    prompts = original_train_data['prompts']
    responses = original_train_data['responses']
    scores = original_train_data['scores']
    cat_tokens = original_train_data['cat_tokens']
    print(len(responses))

# print the first 10 scores
print(scores[:10])
