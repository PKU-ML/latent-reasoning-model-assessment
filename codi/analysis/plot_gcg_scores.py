import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 读取所有json文件
results_dir = "gcg_results_explicit"
results_dir = "gcg_results"
json_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

# 定义颜色
warm_colors = ['#FF6B6B', '#FF8E53', '#FFA07A', '#FF4500', '#FF6347', '#FF7F50', '#FF1493', '#FF69B4', '#FFD700', '#FFA500']
cold_colors = ['#4169E1', '#6495ED', '#87CEEB', '#4682B4', '#5F9EA0', '#00CED1', '#20B2AA', '#778899', '#6A5ACD', '#483D8B']

plt.figure(figsize=(14, 8))

warm_idx = 0
cold_idx = 0

for json_file in json_files:
    file_path = os.path.join(results_dir, json_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    scores = data.get("scores", [])
    attack_success = data.get("attack_success", False)

    # 只取前50个值
    scores = scores[:50]

    if len(scores) == 0:
        continue

    # 横轴：1到50
    x = list(range(1, len(scores) + 1))

    # 选择颜色
    if attack_success:
        color = warm_colors[warm_idx % len(warm_colors)]
        warm_idx += 1
        label = f"{json_file.replace('.json', '')} (success)"
    else:
        color = cold_colors[cold_idx % len(cold_colors)]
        cold_idx += 1
        label = f"{json_file.replace('.json', '')} (fail)"

    plt.plot(x, scores, color=color, label=label, linewidth=1.5, alpha=0.8)

# 设置图表属性
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("GCG Attack Scores Over Iterations", fontsize=14)
plt.xlim(1, 50)
plt.ylim(-12, 5)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig("gcg_scores_plot.png", dpi=150, bbox_inches='tight')
#plt.savefig("gcg_scores_plot.pdf", bbox_inches='tight')
print("图片已保存到 gcg_scores_plot.png")
