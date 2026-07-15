#%%
import re
import json
import random
from transformers import AutoTokenizer
from pathlib import Path
from datasets import load_dataset
random.seed(0)

#%%
tokenizer = AutoTokenizer.from_pretrained('../../../models/llms/gpt2')
p = Path('../../../datasets/text_reasoning/gpqa')
gpqa_extended = load_dataset(path=str(p), name='gpqa_extended')
option_letters = ['A', 'B', 'C', 'D']
#%%
tgt = []
for idx, item in enumerate(gpqa_extended['train']):
    q = item['Question'].strip(' \n')
    if len(tokenizer.tokenize(q)) > 1000:
        print(f'skip: {item}')
        continue

    options = [
        item['Correct Answer'],
        item['Incorrect Answer 1'],
        item['Incorrect Answer 2'],
        item['Incorrect Answer 3'],
    ]
    random.shuffle(options)
    options_text = f"\nOptions:\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}"
    q = q + options_text
    # find the index of correct answer
    a_idx = options.index(item['Correct Answer'])
    a = option_letters[a_idx]

    s = item['Explanation'].strip('\n ')
    # remove all http links in steps
    s = re.sub(r'http\S+', '', s)
    if len(tokenizer.tokenize(s)) > 1000:
        print(f'skip: {item}')
        continue

    tgt.append({
        'id': idx,
        'question': q,
        'steps': [s],
        'answer': a,
    })

random.shuffle(tgt)
length = len(tgt)
train_length = int(length * 0.8)
val_length = int(length * 0.1)
test_length = length - train_length - val_length

#%%
with open(p / 'train.json', 'w') as f:
    json.dump(tgt[:train_length], f, ensure_ascii=False, indent=4)

with open(p / 'val.json', 'w') as f:
    json.dump(tgt[train_length:train_length+val_length], f, ensure_ascii=False, indent=4)

with open(p / 'test.json', 'w') as f:
    json.dump(tgt[train_length+val_length:], f, ensure_ascii=False, indent=4)

# %%
