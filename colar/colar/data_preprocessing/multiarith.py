# %%
import json
from pathlib import Path

dataset_dir = Path("../../../datasets/multiarith")

data = json.load((dataset_dir / "MultiArith.json").open("r"))
# %%
test_json = [
    {
        "idx": idx,
        "question": d["sQuestion"].strip(),
        "steps": d["lEquations"],
        "answer": str(d["lSolutions"][0]),
    }
    for idx, d in enumerate(data)
]

# %%
with (dataset_dir / "test.json").open("w") as f:
    json.dump(test_json, f)

# %%
