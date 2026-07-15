import json
import os
dirname = "adv_results/results_black"
l =os.listdir(dirname)
facts = {}
for i in l:
    j = dirname + "/" + i
    fact = json.load(open(j))["best_fact"]
    if fact in facts:
        facts[fact] += 1
    else:
        facts[fact] = 0

for fact in facts:
    print(fact, facts[fact])
