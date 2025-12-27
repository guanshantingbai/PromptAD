#!/bin/bash
# Prompt2全类别训练脚本示例

# ============================================================
# 使用方法：
# ============================================================

# 1. 训练MVTec所有类别，k=1
# python train_prompt2_all.py --k-shot 1 --dataset mvtec

# 2. 训练VisA所有类别，k=2
# python train_prompt2_all.py --k-shot 2 --dataset visa

# 3. 训练所有数据集（MVTec+VisA），k=4
# python train_prompt2_all.py --k-shot 4 --dataset all

# 4. 自定义参数训练
# python train_prompt2_all.py --k-shot 1 --dataset mvtec --workers 4 --epoch 150

# ============================================================
# 完整实验：依次训练k=1,2,4的所有类别
# ============================================================

echo "开始Prompt2全类别训练实验"
echo "======================================"
echo "配置: n_pro=1, n_pro_ab=4"
echo "并行度: 2进程"
echo "======================================"

# k=1: 所有数据集
echo -e "\n[1/3] 开始训练 k=1..."
python train_prompt2_all.py --k-shot 1 --dataset all

# k=2: 所有数据集
echo -e "\n[2/3] 开始训练 k=2..."
python train_prompt2_all.py --k-shot 2 --dataset all

# k=4: 所有数据集
echo -e "\n[3/3] 开始训练 k=4..."
python train_prompt2_all.py --k-shot 4 --dataset all

echo -e "\n======================================"
echo "全部训练完成！"
echo "======================================"
