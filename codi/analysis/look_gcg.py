import os
import json
di = "gcg_results_explicit"
results = os.listdir(di)
results = sorted(results, key=lambda x: int(x.split("_")[1].split(".")[0]))
ans = 0
tot = 0
for result in results:
    tot += 1
    file = f"{di}/{result}"
    if json.load(open(file))["attack_success"] == True:
        ans += 1
print(f"{ans/tot *100}%")