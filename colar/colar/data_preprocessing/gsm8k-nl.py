# %%
from pathlib import Path
import random
import json

random.seed(0)

#%%
p = Path('../../../datasets/text_reasoning/gsm8k-nl')
train_data = []
for idx, line in enumerate((p / 'train.jsonl').open('r').readlines()):
    try:
        item = json.loads(line)
        train_data.append({
            'id': idx,
            'question': item['question'].strip('\n '),
            'steps': [item['cot'].strip('\n ')],
            'answer': item['answer'].strip('\n ')
        })
    except Exception as e:
        print(f'{e}: {line}')

#%%
random.shuffle(train_data)
train_length = len(train_data)
train_ds = train_data[:int(train_length * 0.8)]
val_ds = train_data[int(train_length * 0.8):]

test_ds = []
for idx, line in enumerate((p / 'test.jsonl').open('r').readlines()):
    try:
        item = json.loads(line)
        test_ds.append({
            'id': idx,
            'question': item['question'].strip('\n '),
            'steps': [item['cot'].strip('\n ')],
            'answer': item['answer'].strip('\n ')
        })
    except Exception as e:
        print(f'{e}: {line}')

# %%
with open(p / 'train.json', 'w') as f:
    json.dump(train_ds, f, indent=2)

with open(p / 'val.json', 'w') as f:
    json.dump(val_ds, f, indent=2)

with open(p / 'test.json', 'w') as f:
    json.dump(test_ds, f, indent=2)

# %%
