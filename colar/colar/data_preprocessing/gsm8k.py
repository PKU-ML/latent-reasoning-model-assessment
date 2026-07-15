# %%
import json
from datasets import load_dataset

p = "../../../datasets/gsm8k"

# %%
test_ds = load_dataset(p, "main", split="test")
train_ds = load_dataset(p, "main", split="train[:90%]")
val_ds = load_dataset(p, "main", split="train[90%:]")
# %%
for split, ds in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
    ds_json = []
    for d in ds:
        q = d["question"]
        a = d["answer"]
        [steps, answer] = a.split("\n####")
        steps = steps.split("\n")
        answer = answer.strip()
        ds_json.append({"question": q, "steps": steps, "answer": answer})
    with open(f"{p}/{split}.json", "w") as f:
        json.dump(ds_json, f)

# %%
