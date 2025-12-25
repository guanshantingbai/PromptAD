#!/usr/bin/env python3
"""
快速验证脚本：测试现有checkpoint的纯语义性能
对比baseline语义 vs 当前checkpoint的语义性能

如果当前checkpoint的语义性能已经接近期望值，说明不需要重新训练
"""

import subprocess
import os
import pandas as pd
import sys

# 测试配置
DATASET = "mvtec"
K_SHOT = 2
RESULT_DIR = "result/prompt1_fixed"  # 修复后的checkpoint

# 关键类别（语义提升大但融合下降）
TEST_CLASSES = ["screw", "toothbrush", "hazelnut", "capsule", "pill"]

# Baseline语义分支的性能（来自fair_comparison_semantic_only_k2.csv）
BASELINE_SEMANTIC = {
    "screw": 66.42,
    "toothbrush": 69.58,
    "hazelnut": 80.11,
    "capsule": 73.69,
    "pill": 85.50,
}

# Prompt1期望语义性能（训练时应该达到的）
EXPECTED_SEMANTIC = {
    "screw": 79.57,
    "toothbrush": 89.44,
    "hazelnut": 91.14,
    "capsule": 80.65,
    "pill": 86.12,
}

print("=" * 80)
print("策略1验证：测试现有checkpoint的纯语义性能")
print("=" * 80)
print()
print(f"数据集: {DATASET}")
print(f"K-shot: {K_SHOT}")
print(f"测试类别: {', '.join(TEST_CLASSES)}")
print()
print("判断标准:")
print("  ✅ 实际 ≥ 期望 → checkpoint很好，不需要重训")
print("  ⚠️  Baseline < 实际 < 期望 → checkpoint一般，考虑重训关键类别")
print("  ❌ 实际 ≤ Baseline → checkpoint不好，需要重训")
print()
print("=" * 80)
print()

results = []

for class_name in TEST_CLASSES:
    print(f"测试类别: {class_name}")
    print(f"  Baseline语义: {BASELINE_SEMANTIC[class_name]:.2f}%")
    print(f"  期望语义: {EXPECTED_SEMANTIC[class_name]:.2f}%")
    print(f"  测试中...", end=" ", flush=True)
    
    # 运行测试（纯语义模式）
    cmd = [
        "python", "test_cls.py",
        "--dataset", DATASET,
        "--class_name", class_name,
        "--k-shot", str(K_SHOT),
        "--semantic-only", "True",
        "--vis", "False",
        "--n_pro", "3",
        "--n_pro_ab", "4",
        "--root-dir", RESULT_DIR,
    ]
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        # 解析输出中的AUROC
        for line in output.split('\n'):
            if "Pixel-AUROC" in line:
                # 提取数字，格式类似：Object:screw =========================== Pixel-AUROC:75.23
                auroc_str = line.split("Pixel-AUROC:")[-1].strip()
                actual_semantic = float(auroc_str)
                
                baseline = BASELINE_SEMANTIC[class_name]
                expected = EXPECTED_SEMANTIC[class_name]
                
                diff_vs_baseline = actual_semantic - baseline
                diff_vs_expected = actual_semantic - expected
                
                # 判断
                if actual_semantic >= expected * 0.98:  # 允许2%误差
                    status = "✅ 很好"
                elif actual_semantic > baseline:
                    status = "⚠️  一般"
                else:
                    status = "❌ 需重训"
                
                print(f"实际: {actual_semantic:.2f}% ({status})")
                
                results.append({
                    "Class": class_name,
                    "Baseline": baseline,
                    "Expected": expected,
                    "Actual": actual_semantic,
                    "vs_Baseline": diff_vs_baseline,
                    "vs_Expected": diff_vs_expected,
                    "Status": status,
                })
                break
        else:
            print("❌ 解析失败")
            results.append({
                "Class": class_name,
                "Baseline": baseline,
                "Expected": expected,
                "Actual": -1,
                "vs_Baseline": -1,
                "vs_Expected": -1,
                "Status": "❌ 测试失败",
            })
    
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {e}")
        results.append({
            "Class": class_name,
            "Baseline": BASELINE_SEMANTIC[class_name],
            "Expected": EXPECTED_SEMANTIC[class_name],
            "Actual": -1,
            "vs_Baseline": -1,
            "vs_Expected": -1,
            "Status": "❌ 测试失败",
        })
    
    print()

# 显示汇总结果
print()
print("=" * 80)
print("验证结果汇总")
print("=" * 80)
print()

df = pd.DataFrame(results)
print(df.to_string(index=False))
print()

# 统计
good_count = sum(1 for r in results if "✅" in r["Status"])
ok_count = sum(1 for r in results if "⚠️" in r["Status"])
bad_count = sum(1 for r in results if "❌" in r["Status"] and "测试失败" not in r["Status"])
failed_count = sum(1 for r in results if "测试失败" in r["Status"])

print(f"统计:")
print(f"  ✅ 很好（≥期望）: {good_count}/{len(TEST_CLASSES)}")
print(f"  ⚠️  一般（Baseline < 实际 < 期望）: {ok_count}/{len(TEST_CLASSES)}")
print(f"  ❌ 需重训（≤Baseline）: {bad_count}/{len(TEST_CLASSES)}")
if failed_count > 0:
    print(f"  ❌ 测试失败: {failed_count}/{len(TEST_CLASSES)}")
print()

# 给出建议
print("=" * 80)
print("建议")
print("=" * 80)
print()

if good_count >= len(TEST_CLASSES) * 0.8:
    print("✅ 现有checkpoint的语义性能很好！")
    print("   → 不需要重新训练")
    print("   → 可以直接用--semantic-only模式测试所有类别")
    print("   → 或者考虑改进融合策略（自适应权重）")
elif good_count + ok_count >= len(TEST_CLASSES) * 0.6:
    print("⚠️  现有checkpoint的语义性能一般")
    print("   → 建议重训表现不佳的类别（标记为⚠️或❌的）")
    print("   → 预计时间：约1-2小时")
else:
    print("❌ 现有checkpoint的语义性能不理想")
    print("   → 建议修复train_cls.py后重新训练")
    print("   → 可以先重训这5个关键类别验证修复效果")
    print("   → 预计时间：约1-2小时")

print()

# 保存结果
output_file = f"{RESULT_DIR}/semantic_validation_k{K_SHOT}.csv"
df.to_csv(output_file, index=False)
print(f"结果已保存到: {output_file}")
print()
