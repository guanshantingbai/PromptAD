#!/bin/bash

echo "=== 开始清理工作目录 ==="

# 1. 删除所有log和csv文件
echo "��️  删除log和csv文件..."
rm -f *.log *.csv nohup.out 2>/dev/null
echo "   ✅ 已删除所有log和csv文件"

# 2. 归档verify/visualize/test开头的文件
echo "📦 归档测试和验证脚本..."
mv verify*.py verify*.sh visualize*.py test*.py test*.sh archived/ 2>/dev/null
echo "   ✅ 已归档verify/visualize/test文件"

# 3. 归档所有.sh脚本（除了刚创建的清理脚本）
echo "📦 归档shell脚本..."
for script in *.sh; do
    if [[ "$script" != "cleanup_workspace.sh" ]]; then
        mv "$script" archived/ 2>/dev/null
    fi
done
echo "   ✅ 已归档所有.sh脚本"

# 4. 统计结果
echo ""
echo "=== 清理完成 ==="
echo "📊 归档文件统计："
echo "   archived/目录: $(ls -1 archived/ | wc -l) 个文件"
echo ""
echo "✨ 当前根目录核心文件："
ls -1 *.py 2>/dev/null | head -10
