# %%
import json
from pathlib import Path

dataset_dir = Path("../../../datasets/svamp/")

data = json.load((dataset_dir / "SVAMP.json").open("r"))
# %%
test_json = [
    {
        "idx": idx,
        "question": d["Body"].strip(". ") + ". " + d["Question"].strip(),
        "steps": [d["Equation"]],
        "answer": str(d["Answer"]),
    }
    for idx, d in enumerate(data)
]
# %%
with (dataset_dir / "test.json").open("w") as f:
    json.dump(test_json, f, indent=4)
# %%
