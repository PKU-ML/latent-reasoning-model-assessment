"""
批量运行RoT攻击脚本

功能：
- 对多个数据集运行三种攻击方法（白盒、黑盒、随机）
- 依次对每个问题ID运行攻击脚本
- 统计攻击成功率
"""

import json
import os
import subprocess
import argparse
from pathlib import Path

# 子项目根目录（默认本脚本所在目录），可通过 ROT_PROJECT_DIR 覆盖
_SUBPROJECT_DIR = Path(os.environ.get("ROT_PROJECT_DIR", Path(__file__).resolve().parent))

# 配置
RESULTS_FILE = os.environ.get(
    "ROT_RESULTS_FILE",
    str(_SUBPROJECT_DIR / "results" / "org" / "test_org_results.json"),
)
GPU_DEVICE = "0"
DATA = "gsm8k"
DATA = "MultiArith"
#DATA = "SVAMP"

method = "W"  # W=white, B=black, R=random

if method == "W":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_white.py")
    OUTPUT_DIR = os.environ.get(
        "ROT_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_white_3"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)
    ATTACK_PARAMS = {
        "--prefix-length": "3",
        "--n-iters": "30",
        "--dataset": DATA,
        "--output-dir": OUTPUT_DIR,
    }
elif method == "B":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_black.py")
    OUTPUT_DIR = os.environ.get(
        "ROT_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_black"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)
    ATTACK_PARAMS = {
        "--dataset": DATA,
        "--output-dir": OUTPUT_DIR,
    }
elif method == "R":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_random.py")
    OUTPUT_DIR = os.environ.get(
        "ROT_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_random"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)
    ATTACK_PARAMS = {
        "--dataset": DATA,
        "--output-dir": OUTPUT_DIR,
    }


def load_correct_ids():
    """从test_org_results.json加载正确问题ID"""
    global DATA
    if not os.path.exists(RESULTS_FILE):
        print(f"结果文件不存在: {RESULTS_FILE}")
        return []
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    correct_ids = data.get(DATA, {}).get("correct_ids", [])
    print(f"加载到 {len(correct_ids)} 个正确问题ID")
    return correct_ids


def load_existing_results():
    """加载已存在的结果，避免重复运行"""
    existing_ids = set()
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith('.json'):
                try:
                    problem_id = int(fname.replace('.json', '').replace('problem_', ''))
                    existing_ids.add(problem_id)
                except:
                    pass
    return existing_ids


def run_attack_for_id(problem_id):
    """对指定问题ID运行攻击"""
    print(f"\n{'='*60}")
    print(f"开始攻击问题 ID: {problem_id}")
    print(f"{'='*60}")

    cmd = [
        "python", ATTACK_SCRIPT,
        "--problem-id", str(problem_id),
    ]
    for k, v in ATTACK_PARAMS.items():
        cmd.append(k)
        cmd.append(v)

    env = os.environ.copy()

    result = subprocess.run(
        cmd,
        env=env,
        cwd=os.path.dirname(ATTACK_SCRIPT)
    )

    return result.returncode, "", ""


def parse_attack_result(problem_id):
    """从结果文件解析攻击结果"""
    result_file = os.path.join(OUTPUT_DIR, f"problem_{problem_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        return result.get('attack_success', False)
    else:
        print(f"  [!] 结果文件不存在: {result_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量运行RoT攻击")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--method", type=str, choices=["W", "B", "R"], default="R",
                        help="W=white, B=black, R=random")
    args = parser.parse_args()

    global GPU_DEVICE, method, ATTACK_SCRIPT, OUTPUT_DIR, ATTACK_PARAMS
    GPU_DEVICE = args.gpu

    correct_ids = load_correct_ids()

    start_idx = args.start
    end_idx = args.end if args.end is not None else len(correct_ids)
    ids_to_process = correct_ids[start_idx:end_idx]

    print(ids_to_process)
    print(f"将处理第 {start_idx} 到 {end_idx} 个问题，共 {len(ids_to_process)} 个")
    print(f"输出目录: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    success = 0
    failed = 0
    skipped = 0

    existing_ids = set()
    if args.resume:
        existing_ids = load_existing_results()
        print(f"发现 {len(existing_ids)} 个已存在的结果")

    for i, problem_id in enumerate(ids_to_process):
        if args.resume and problem_id in existing_ids:
            print(f"\n[{i+1}/{len(ids_to_process)}] 跳过已存在的问题 ID: {problem_id}")
            skipped += 1
            continue

        total += 1
        returncode, stdout, stderr = run_attack_for_id(problem_id)
        attack_success = parse_attack_result(problem_id)

        if attack_success:
            success += 1
            print(f"\n[{i+1}/{len(ids_to_process)}] 问题 ID: {problem_id} - 攻击成功!")
        else:
            failed += 1
            print(f"\n[{i+1}/{len(ids_to_process)}] 问题 ID: {problem_id} - 攻击失败")

    print(f"\n{'='*60}")
    print("批量攻击完成!")
    print(f"{'='*60}")
    print(f"总处理: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    if total > 0:
        print(f"成功率: {success}/{total} = {success/total*100:.2f}%")


if __name__ == "__main__":
    main()