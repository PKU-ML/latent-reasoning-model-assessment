"""
批量运行llama GCG对抗攻击脚本

功能：
- 从 test_org_results.json 读取 gsm8k 的 correct_ids（模型能正确回答的问题ID）
- 依次对每个问题ID运行 attack_prefix.py 进行攻击
- 统计攻击成功率
"""

import json
import os
import subprocess
import argparse
from pathlib import Path

# 子项目根目录（默认本脚本所在目录），可通过 LLAMA_PROJECT_DIR 覆盖
_SUBPROJECT_DIR = Path(os.environ.get("LLAMA_PROJECT_DIR", Path(__file__).resolve().parent))

method = "R"
# 配置
RESULTS_FILE = os.environ.get(
    "LLAMA_RESULTS_FILE",
    str(_SUBPROJECT_DIR / "results" / "org" / "test_org_results.json"),
)
GPU_DEVICE = "0"  # 使用第1个卡

DATA = "gsm8k"
# DATA = "MultiArith"
# DATA = "SVAMP"

if method == "W":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_white_set.py")
    OUTPUT_DIR = os.environ.get(
        "LLAMA_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_set"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)

    # 攻击参数
    ATTACK_PARAMS = {
        "--prefix-length": "5",
        "--n-iters": "10",
        "--candidate-selection": "gradient",
        "--candidate-set-size" : "10",
        "--output-dir" : OUTPUT_DIR,
    }
elif method == "B":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_black.py")
    OUTPUT_DIR = os.environ.get(
        "LLAMA_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_black"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)

    ATTACK_PARAMS = {
        "--output-dir" : OUTPUT_DIR,
    }
elif method == "R":
    ATTACK_SCRIPT = str(_SUBPROJECT_DIR / "attack_random.py")
    OUTPUT_DIR = os.environ.get(
        "LLAMA_OUTPUT_DIR",
        str(_SUBPROJECT_DIR / "adv_results" / "results_random"),
    )
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATA)

    ATTACK_PARAMS = {
        "--output-dir" : OUTPUT_DIR,
    }

def load_correct_ids():
    """从test_org_results.json加载correct的问题ID"""
    global DATA
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    correct_ids = data[DATA]["correct_ids"]
    print(f"加载到 {len(correct_ids)} 个正确问题ID")
    return correct_ids


def load_existing_results():
    """加载已存在的结果，避免重复运行"""
    existing_ids = set()
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith('.json'):
                try:
                    result_path = os.path.join(OUTPUT_DIR, fname)
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    # 尝试从结果文件中获取problem_id
                    #if 'prefix_token_ids' in result:
                        # 从文件名获取
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

    # 构建命令
    cmd = [
        "python", ATTACK_SCRIPT,
        "--problem-id", str(problem_id),
    ]
    for k, v in ATTACK_PARAMS.items():
        cmd.append(k)
        cmd.append(v)

    # 设置GPU并运行
    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

    result = subprocess.run(
        cmd,
        env=env,
        cwd=os.path.dirname(ATTACK_SCRIPT)
    )

    return result.returncode, "", ""


def parse_attack_result(problem_id):
    """从结果文件解析攻击结果"""
    success = False
    logit_diff = None

    # 从结果文件读取
    result_file = os.path.join(OUTPUT_DIR, f"problem_{problem_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        success = result.get('attack_success', False)
        logit_diff = result.get('adv_logit_diff')
    else:
        print(f"  [!] 结果文件不存在: {result_file}")

    return success, logit_diff


def main():
    parser = argparse.ArgumentParser(description="批量运行llama GCG攻击")
    parser.add_argument("--start", type=int, default=0,
                       help="从第几个correct_ids开始（索引）")
    parser.add_argument("--end", type=int, default=None,
                       help="到第几个correct_ids结束（索引）")
    parser.add_argument("--resume", action="store_true",
                       help="跳过已存在的问题ID")
    parser.add_argument("--gpu", type=str, default="0",
                       help="GPU设备号")
    args = parser.parse_args()
    # python run_attack.py --end=168 --resume

    global GPU_DEVICE
    GPU_DEVICE = args.gpu

    # 加载问题ID
    correct_ids = load_correct_ids()

    # 截取需要处理的范围
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(correct_ids)
    ids_to_process = correct_ids[start_idx:end_idx]

    print(ids_to_process)
    print(f"将处理第 {start_idx} 到 {end_idx} 个问题，共 {len(ids_to_process)} 个")
    print(f"输出目录: {OUTPUT_DIR}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 统计
    total = 0
    success = 0
    failed = 0
    skipped = 0

    # 加载已存在的结果
    existing_ids = set()
    if args.resume:
        existing_ids = load_existing_results()
        print(f"发现 {len(existing_ids)} 个已存在的结果")

    for i, problem_id in enumerate(ids_to_process):
        # 如果resume模式且结果已存在，跳过
        if args.resume and problem_id in existing_ids:
            print(f"\n[{i+1}/{len(ids_to_process)}] 跳过已存在的问题 ID: {problem_id}")
            skipped += 1
            continue

        total += 1

        # 运行攻击
        returncode, stdout, stderr = run_attack_for_id(problem_id)

        # 解析结果
        attack_success, logit_diff = parse_attack_result(problem_id)

        if attack_success:
            success += 1
            print(f"\n[{i+1}/{len(ids_to_process)}] 问题 ID: {problem_id} - 攻击成功!")
        else:
            failed += 1
            print(f"\n[{i+1}/{len(ids_to_process)}] 问题 ID: {problem_id} - 攻击失败")

        if logit_diff is not None:
            print(f"    Logit差值: {logit_diff:.4f}")

    # 打印最终统计
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
