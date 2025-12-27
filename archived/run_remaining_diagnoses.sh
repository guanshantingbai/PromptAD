#!/bin/bash
# 串行运行剩余的诊断

classes=(
    "visa-pcb3"
    "visa-macaroni2"
    "visa-macaroni1"
    "mvtec-metal_nut"
    "visa-chewinggum"
    "visa-candle"
    "visa-capsules"
    "visa-cashew"
    "mvtec-transistor"
    "visa-pcb1"
    "mvtec-zipper"
    "mvtec-bottle"
    "mvtec-tile"
    "mvtec-wood"
    "mvtec-leather"
    "mvtec-carpet"
    "visa-pcb4"
)

total=${#classes[@]}
completed=0

echo "===================================================================================="
echo "串行运行剩余的 $total 个类别诊断"
echo "===================================================================================="
echo ""

for cls in "${classes[@]}"; do
    completed=$((completed + 1))
    echo "[$completed/$total] 诊断 $cls ..."
    
    timeout 120 python diagnose_prototypes.py --k-shot 2 --classes "$cls" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$completed/$total] ✅ $cls 完成"
    else
        echo "[$completed/$total] ❌ $cls 失败或超时"
    fi
    echo ""
done

echo "===================================================================================="
echo "所有诊断已完成！"
echo "===================================================================================="
