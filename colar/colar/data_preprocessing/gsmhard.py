# %%
import jsonlines
import json
from pathlib import Path

d = Path("../../../datasets/gsmhard")
# %%
with jsonlines.open(str(d / "gsmhardv2.jsonl")) as reader:
    data = list(reader)

data = [
    {
        "idx": i,
        "question": d["input"],
        "steps": [d["code"]],
        "answer": str(d["target"]),
    }
    for i, d in enumerate(data)
]

# %%
with (d / "test.json").open("w") as f:
    json.dump(data, f)

# %%
