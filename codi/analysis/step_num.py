import json
import os
dirname = "adv_results/results_cos_8"
files = os.listdir(dirname)
step = 0
for file in files:
    step += len(json.load(open(f"{dirname}/{file}"))["scores"])
print(step/len(files))

#python analysis/step_num.py