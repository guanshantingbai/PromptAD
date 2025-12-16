#!/bin/bash
################################################################################
# 服务器代码验证脚本
# 检查服务器上的代码是否包含所有优化
################################################################################

SERVER_PATH="/home/grj/codes/mywork/PromptAD"

echo "=========================================="
echo "检查服务器代码状态"
echo "路径: $SERVER_PATH"
echo "=========================================="
echo ""

cd "$SERVER_PATH" || exit 1

echo "=== Git 状态 ==="
git status
echo ""

echo "=== 当前分支和最新提交 ==="
git log --oneline -3
echo ""

echo "=========================================="
echo "关键优化检查"
echo "=========================================="
echo ""

echo "✓ 检查 1: resize 是否降到 256"
if grep -q "cv2.resize(img, (256, 256))" datasets/dataset.py; then
    echo "  ✅ PASS: resize 使用 256x256"
else
    echo "  ❌ FAIL: resize 未优化，仍可能是 1024x1024"
    grep -n "cv2.resize(img," datasets/dataset.py
fi
echo ""

echo "✓ 检查 2: transform 是否在 Dataset 中"
if grep -q "if self.transform is not None:" datasets/dataset.py; then
    echo "  ✅ PASS: transform 集成到 Dataset"
else
    echo "  ❌ FAIL: transform 未集成"
fi
echo ""

echo "✓ 检查 3: num_workers 默认值"
NUM_WORKERS=$(grep "num_workers = kwargs.get" datasets/__init__.py | grep -o "([0-9]*)" | tr -d '()')
if [ "$NUM_WORKERS" = "1" ]; then
    echo "  ✅ PASS: num_workers 默认值是 1"
elif [ "$NUM_WORKERS" = "0" ]; then
    echo "  ⚠️  WARNING: num_workers 默认值是 0（未优化但不影响）"
else
    echo "  ❌ FAIL: num_workers 配置异常"
fi
echo ""

echo "✓ 检查 4: 训练循环是否移除图像转换"
if grep -q "cv2.cvtColor.*for f in data" train_cls.py; then
    echo "  ❌ FAIL: 训练循环仍有 cv2.cvtColor（未优化！）"
    grep -n "cv2.cvtColor" train_cls.py
elif grep -q "data is already transformed by Dataset" train_cls.py; then
    echo "  ✅ PASS: 训练循环已移除重复转换"
else
    echo "  ⚠️  UNKNOWN: 无法确认状态"
fi
echo ""

echo "✓ 检查 5: 类型转换修复"
if grep -q "torch.from_numpy(gt).long()" datasets/dataset.py; then
    echo "  ✅ PASS: gt/label 类型转换已修复"
else
    echo "  ❌ FAIL: 类型转换未修复，可能导致 DataLoader 错误"
fi
echo ""

echo "=========================================="
echo "正在运行的训练进程"
echo "=========================================="
ps aux | grep "python.*train_" | grep -v grep | head -5
echo ""

echo "=========================================="
echo "进程详细参数"
echo "=========================================="
for pid in $(pgrep -f "python.*train_cls"); do
    echo "PID: $pid"
    ps -p "$pid" -o args= | tr ' ' '\n' | grep -E "num-workers|batch-size" | head -2
    echo ""
done

echo "=========================================="
echo "检查完成"
echo "=========================================="
