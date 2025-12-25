#!/bin/bash
# 批量测试脚本 - MVTec CLS k=2
# 按优先级顺序测试剩余的13个类别

set -e  # 遇到错误时停止

# 创建日志目录
mkdir -p test_logs/batch_testing

# 时间戳
START_TIME=$(date +%s)
echo "=============================================="
echo "开始批量测试 - $(date)"
echo "=============================================="

# 定义类别列表（按优先级）
HIGH_PRIORITY=(toothbrush hazelnut)
MED_PRIORITY=(capsule)
LOW_PRIORITY=(metal_nut cable zipper wood pill grid carpet leather tile transistor)

# 测试函数
test_class() {
    local class=$1
    local priority=$2
    
    echo ""
    echo "=============================================="
    echo "[$priority] Testing: $class"
    echo "Time: $(date)"
    echo "=============================================="
    
    # 复制checkpoint
    cp result/prompt1/mvtec/k_2/checkpoint/CLS-Seed_111-${class}-check_point.pt \
       result/mvtec/k_2/checkpoint/ 2>/dev/null || echo "Checkpoint already exists"
    
    # 运行测试
    python test_cls.py \
        --class_name $class \
        --k-shot 2 \
        --batch-size 1 \
        2>&1 | tee "test_logs/batch_testing/${class}_result.log"
    
    # 提取结果
    local result=$(grep "Object:${class}" "test_logs/batch_testing/${class}_result.log" | tail -1)
    echo "$result"
    
    # 保存到汇总文件
    echo "$class,$result" >> test_logs/batch_testing/summary.csv
}

# 创建汇总文件头
echo "class,result_line" > test_logs/batch_testing/summary.csv

# 阶段1: 高优先级
echo ""
echo "================================================"
echo "阶段1: 高优先级类别（语义提升>10%）"
echo "================================================"
for class in "${HIGH_PRIORITY[@]}"; do
    test_class $class "HIGH"
done

# 阶段2: 中优先级
echo ""
echo "================================================"
echo "阶段2: 中优先级类别（语义提升5-10%）"
echo "================================================"
for class in "${MED_PRIORITY[@]}"; do
    test_class $class "MEDIUM"
done

# 阶段3: 低优先级
echo ""
echo "================================================"
echo "阶段3: 低优先级类别（语义提升<5%）"
echo "================================================"
for class in "${LOW_PRIORITY[@]}"; do
    test_class $class "LOW"
done

# 计算总时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "批量测试完成！"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "时间: $(date)"
echo "=============================================="

# 生成结果报告
python3 << 'PYTHON_SCRIPT'
import pandas as pd
import re

print("\n" + "="*80)
print("测试结果汇总")
print("="*80)

# 读取所有日志文件
import glob
import os

results = []
for log_file in sorted(glob.glob("test_logs/batch_testing/*_result.log")):
    class_name = os.path.basename(log_file).replace("_result.log", "")
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # 提取AUROC
    match = re.search(r'Pixel-AUROC:([\d.]+)', content)
    if match:
        auroc = float(match.group(1))
        results.append({
            'class': class_name,
            'auroc': auroc
        })

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('auroc', ascending=False)
    
    print(f"\n{'类别':<15} {'Image AUROC':>12}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['class']:<15} {row['auroc']:>12.2f}")
    
    print("-"*80)
    print(f"{'平均':<15} {df['auroc'].mean():>12.2f}")
    print(f"{'最高':<15} {df['auroc'].max():>12.2f}")
    print(f"{'最低':<15} {df['auroc'].min():>12.2f}")
    
    # 保存结果
    df.to_csv('test_logs/batch_testing/final_results.csv', index=False)
    print(f"\n结果已保存到: test_logs/batch_testing/final_results.csv")
else:
    print("未找到有效结果")

print("="*80)
PYTHON_SCRIPT

echo ""
echo "详细日志位于: test_logs/batch_testing/"
