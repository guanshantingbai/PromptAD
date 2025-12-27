#!/usr/bin/env python3
"""
快速单类别测试：验证纯语义性能
用法：python quick_test_semantic.py screw 2
"""

import subprocess
import sys

if len(sys.argv) < 2:
    print("用法: python quick_test_semantic.py <class_name> [k_shot]")
    print("示例: python quick_test_semantic.py screw 2")
    sys.exit(1)

class_name = sys.argv[1]
k_shot = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# 已知的baseline和期望性能
BASELINE_SEMANTIC = {
    "screw": 66.42, "toothbrush": 69.58, "hazelnut": 80.11,
    "capsule": 73.69, "pill": 85.50, "metal_nut": 85.56,
    "cable": 83.60, "bottle": 95.52, "transistor": 89.60,
}

EXPECTED_SEMANTIC = {
    "screw": 79.57, "toothbrush": 89.44, "hazelnut": 91.14,
    "capsule": 80.65, "pill": 86.12, "metal_nut": 88.71,
    "cable": 86.00, "bottle": 98.25, "transistor": 78.08,
}

print(f"\n测试 {class_name} (k={k_shot})...")
if class_name in BASELINE_SEMANTIC:
    print(f"  Baseline语义: {BASELINE_SEMANTIC[class_name]:.2f}%")
    print(f"  期望语义: {EXPECTED_SEMANTIC[class_name]:.2f}%")
else:
    print(f"  (无参考数据)")

print(f"\n🔍 测试纯语义性能...")

cmd = [
    "python", "test_cls.py",
    "--dataset", "mvtec",
    "--class_name", class_name,
    "--k-shot", str(k_shot),
    "--semantic-only", "True",
    "--vis", "False",
    "--n_pro", "3",
    "--n_pro_ab", "4",
    "--root-dir", "result/prompt1_fixed",  # 修复后重训的checkpoint
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)

if result.returncode != 0:
    print(f"❌ 测试失败")
    print(result.stderr)
    sys.exit(1)

# 解析结果
for line in result.stdout.split('\n'):
    if "Pixel-AUROC" in line:
        auroc_str = line.split("Pixel-AUROC:")[-1].strip()
        actual = float(auroc_str)
        
        if class_name in BASELINE_SEMANTIC:
            baseline = BASELINE_SEMANTIC[class_name]
            expected = EXPECTED_SEMANTIC[class_name]
            
            print(f"\n📊 结果对比:")
            print(f"  Baseline语义: {baseline:.2f}%")
            print(f"  期望语义: {expected:.2f}%")
            print(f"  实际语义: {actual:.2f}%")
            print(f"  vs Baseline: {actual - baseline:+.2f}%")
            print(f"  vs 期望: {actual - expected:+.2f}%")
            
            if actual >= expected * 0.98:
                print(f"  ✅ 很好！达到期望性能")
            elif actual > baseline:
                print(f"  ⚠️  一般，高于baseline但未达期望")
            else:
                print(f"  ❌ 需重训，未超过baseline")
        break
