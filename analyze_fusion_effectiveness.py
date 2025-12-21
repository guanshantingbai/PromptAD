#!/usr/bin/env python3
"""
分析多原型改进在融合后是否仍然有效

关键问题：
1. 多原型在纯语义上提升了 +3.37%
2. 但当与记忆库融合时，这个提升会保持吗？
3. 还是说融合会"抹平"语义分支的差异？
"""

import pandas as pd
import numpy as np


def analyze_fusion_preservation():
    """
    分析融合后多原型优势是否保持
    """
    
    print("="*100)
    print("融合有效性分析：多原型改进在融合后是否保持？")
    print("="*100)
    
    # 已知数据
    baseline_semantic = 88.10  # 单原型纯语义
    prompt1_semantic = 91.47   # 多原型纯语义
    semantic_improvement = prompt1_semantic - baseline_semantic
    
    baseline_full = 94.30      # 单原型 + 记忆库 (harmonic融合)
    
    print("\n## 1. 已知数据")
    print("-"*100)
    print(f"Baseline 纯语义 (单原型):           {baseline_semantic:.2f}%")
    print(f"Prompt1 纯语义 (多原型):            {prompt1_semantic:.2f}%")
    print(f"纯语义改进:                         +{semantic_improvement:.2f}%")
    print()
    print(f"Baseline 完整系统 (单原型+记忆库):  {baseline_full:.2f}%")
    print(f"记忆库贡献:                         +{baseline_full - baseline_semantic:.2f}%")
    
    # 理论分析
    print("\n## 2. 理论分析：融合后的期望效果")
    print("-"*100)
    
    print("\n### 假设1: 线性可加性 (乐观估计)")
    print("   如果多原型改进和记忆库改进是独立的：")
    expected_linear = baseline_semantic + semantic_improvement + (baseline_full - baseline_semantic)
    print(f"   预期 = {baseline_semantic:.2f} + {semantic_improvement:.2f} + {baseline_full - baseline_semantic:.2f}")
    print(f"        = {expected_linear:.2f}%")
    print(f"   vs Baseline完整: +{expected_linear - baseline_full:.2f}%")
    
    print("\n### 假设2: 调和平均融合 (实际情况)")
    print("   Baseline使用 harmonic mean 融合：")
    print("   score = 2 * (semantic * visual) / (semantic + visual)")
    print()
    
    # 模拟不同记忆库性能下的融合效果
    print("   模拟：如果记忆库性能保持不变...")
    
    # 推断记忆库单独性能
    # baseline_full = 2 * semantic * visual / (semantic + visual)
    # => visual = (baseline_full * semantic) / (2 * semantic - baseline_full)
    
    visual_score = (baseline_full * baseline_semantic) / (2 * baseline_semantic - baseline_full)
    print(f"   推断记忆库单独得分: ~{visual_score:.2f}%")
    
    # 多原型 + 相同记忆库
    expected_harmonic = 2 * prompt1_semantic * visual_score / (prompt1_semantic + visual_score)
    print(f"\n   多原型 + 记忆库 (调和融合):")
    print(f"   = 2 * {prompt1_semantic:.2f} * {visual_score:.2f} / ({prompt1_semantic:.2f} + {visual_score:.2f})")
    print(f"   = {expected_harmonic:.2f}%")
    print(f"   vs Baseline完整: +{expected_harmonic - baseline_full:.2f}%")
    
    # 分析融合保持率
    actual_improvement = expected_harmonic - baseline_full
    preservation_rate = (actual_improvement / semantic_improvement) * 100 if semantic_improvement > 0 else 0
    
    print(f"\n   改进保持率: {preservation_rate:.1f}%")
    print(f"   (纯语义改进 {semantic_improvement:.2f}% → 融合后改进 {actual_improvement:.2f}%)")
    
    # 关键洞察
    print("\n## 3. 关键洞察")
    print("-"*100)
    
    print("\n### 🔍 Harmonic融合的特性：")
    print("   • 调和平均偏向于较小值")
    print("   • 两个分支都强时，融合效果才最好")
    print("   • 一个分支的改进会被另一个分支\"稀释\"")
    
    if preservation_rate > 80:
        print(f"\n   ✅ 保持率 {preservation_rate:.1f}% > 80%: 改进在融合后大部分保持！")
    elif preservation_rate > 50:
        print(f"\n   ⚠️  保持率 {preservation_rate:.1f}% (50-80%): 改进部分保持，但有明显衰减")
    else:
        print(f"\n   ❌ 保持率 {preservation_rate:.1f}% < 50%: 改进在融合后大幅衰减")
    
    # 实验验证需求
    print("\n## 4. 现有数据能否证明？")
    print("-"*100)
    
    print("\n❌ **不能完全证明！需要额外推理！**")
    print("\n原因：")
    print("   1. 我们有: Baseline完整 (单原型+记忆库) = 94.30%")
    print("   2. 我们有: Prompt1纯语义 (多原型) = 91.47%")
    print("   3. 缺少: Prompt1完整 (多原型+记忆库) = ???")
    print()
    print("   当前无法知道多原型+记忆库的实际融合效果！")
    
    print("\n## 5. 验证方案")
    print("-"*100)
    
    print("\n### 方案A: 理论推断 (已完成)")
    print(f"   • 基于harmonic融合公式推断: {expected_harmonic:.2f}%")
    print(f"   • 预期改进: +{actual_improvement:.2f}%")
    print("   • 局限性: 假设记忆库性能不变，但实际可能因训练变化而不同")
    
    print("\n### 方案B: 实际测试 (推荐)")
    print("   步骤:")
    print("   1. 在当前多原型模型中加回记忆库")
    print("   2. 使用harmonic融合")
    print("   3. 在MVTec k=2上测试")
    print("   4. 对比实际结果 vs 理论预测")
    print()
    print("   需要的工作:")
    print("   • 修改 PromptAD/model.py 恢复 calculate_visual_anomaly_score")
    print("   • 修改 forward() 实现双分支融合")
    print("   • 运行 test_cls.py --k_shot 2")
    
    print("\n### 方案C: 使用已有checkpoint (快速验证)")
    print("   如果prompt1训练时保存了特征库...")
    print("   1. 检查 result/prompt1/mvtec/bottle_k2/ 是否有 feature_gallery*")
    print("   2. 如果有，可以直接加载并计算视觉分数")
    print("   3. 快速验证融合效果")
    
    # 逐类分析融合效果预测
    print("\n## 6. 逐类融合效果预测")
    print("-"*100)
    
    # 读取详细数据
    prompt1_data = {
        'bottle': 98.25, 'cable': 86.00, 'capsule': 80.65, 'carpet': 100.00,
        'grid': 99.00, 'hazelnut': 91.14, 'leather': 100.00, 'metal_nut': 88.71,
        'pill': 86.12, 'screw': 79.57, 'tile': 99.93, 'toothbrush': 89.44,
        'transistor': 78.08, 'wood': 99.65, 'zipper': 95.46
    }
    
    baseline_semantic_data = {
        'bottle': 95.52, 'cable': 83.60, 'capsule': 73.69, 'carpet': 100.00,
        'grid': 98.87, 'hazelnut': 80.11, 'leather': 100.00, 'metal_nut': 85.56,
        'pill': 85.50, 'screw': 66.42, 'tile': 99.96, 'toothbrush': 69.58,
        'transistor': 89.60, 'wood': 98.82, 'zipper': 94.22
    }
    
    # 从baseline完整结果推断各类记忆库得分
    # 这需要baseline完整的逐类结果...
    print("\n   ⚠️ 需要baseline完整系统的逐类结果来做精确预测")
    print("   当前只能给出平均水平的估计")
    
    print("\n## 7. 推荐行动")
    print("-"*100)
    
    print("\n优先级排序:")
    print()
    print("1️⃣  **快速验证** (1-2小时)")
    print("   • 检查prompt1训练checkpoint是否保存了feature_gallery")
    print("   • 如果有，编写脚本加载并计算融合得分")
    print("   • 这能快速验证理论预测")
    print()
    print("2️⃣  **完整实验** (4-8小时)")
    print("   • 在prompt1代码中恢复记忆库功能")
    print("   • 重新测试所有类别 (不需要重新训练！)")
    print("   • 得到确切的融合效果数据")
    print()
    print("3️⃣  **深入分析** (如果融合有效)")
    print("   • 分析哪些类受益于多原型")
    print("   • 研究语义-视觉协同机制")
    print("   • 探索最优融合策略")
    
    print("\n" + "="*100)
    
    return {
        'semantic_improvement': semantic_improvement,
        'expected_fusion_improvement': actual_improvement,
        'preservation_rate': preservation_rate,
        'needs_verification': True
    }


if __name__ == "__main__":
    result = analyze_fusion_preservation()
    
    print("\n📌 结论：")
    print("-"*100)
    print("理论预测多原型改进在融合后能保持，但**需要实际测试验证**！")
    print()
    print("建议先检查已有checkpoint能否快速验证，否则需要恢复记忆库功能。")
    print()
