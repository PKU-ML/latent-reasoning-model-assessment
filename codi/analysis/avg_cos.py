import json
import os
dirname = "adv_results/results_cos_3"
files = os.listdir(dirname)
cosine = 0
num = 0
for file in files:
    if json.load(open(f"{dirname}/{file}"))["attack_success"] == True:
        num += 1    
        try:
            cosine += json.load(open(f"{dirname}/{file}"))["adv_cosine"]
        except:
            cosine += json.load(open(f"{dirname}/{file}"))["best_cosine"]
print(cosine/num)
# python analysis/avg_cos.py