# %%
from pathlib import Path
import random
import json

random.seed(0)

dataset_dir = Path("../../../datasets/math")

train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"

# %%
train_data = []
test_data = []

for child_dir in train_dir.iterdir():
    for file in child_dir.iterdir():
        with open(file, "r") as f:
            train_data.append(json.load(f))

for child_dir in test_dir.iterdir():
    for file in child_dir.iterdir():
        with open(file, "r") as f:
            test_data.append(json.load(f))


# %%
def get_data(data):
    final_data = []
    data_cnt = 0
    for item in data:
        # split last step and answer
        solution = item["solution"].strip()
        if "\\boxed{" not in solution:
            print(f"no boxed: {item}")
            continue
        try:
            [step, answer_and_remaining] = solution.split("\\boxed{")
            r_brace_idx = answer_and_remaining.rfind("}")
            answer = answer_and_remaining[:r_brace_idx]
            step = step + answer_and_remaining[r_brace_idx:]
        except ValueError or IndexError:
            print(f"last step error: {item}")
            continue

        processed_item = {
            "level": item["level"],
            "type": item["type"],
            "idx": data_cnt,
            "question": item["problem"],
            "steps": [step],  # considered as one step here as they are diffucult to split
            "answer": answer,
        }
        final_data.append(processed_item)
        data_cnt += 1

    return final_data


processed_train_data = get_data(train_data)
processed_test_data = get_data(test_data)
print(len(processed_train_data))
print(len(processed_test_data))

random.shuffle(processed_train_data)
# %%
n_train_samples = int(len(processed_train_data) * 0.9)

train_json = processed_train_data[:n_train_samples]
val_json = processed_train_data[n_train_samples:]
test_json = processed_test_data

# %%

with open(dataset_dir / "train.json", "w") as f:
    json.dump(train_json, f, indent=4)

with open(dataset_dir / "val.json", "w") as f:
    json.dump(val_json, f, indent=4)

with open(dataset_dir / "test.json", "w") as f:
    json.dump(test_json, f, indent=4)

# %%
