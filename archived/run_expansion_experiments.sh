#!/bin/bash

# PromptAD扩展实验自动化脚本
# 按三阶段顺序执行：SEG验证 -> MVTec完整 -> VisA完整

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p logs

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PromptAD 扩展实验 - 三阶段执行${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "总任务数: 180"
echo "  Phase 1: SEG验证   - 18个任务 (~30分钟)"
echo "  Phase 2: MVTec完整 - 90个任务 (~2.5小时)"
echo "  Phase 3: VisA完整  - 72个任务 (~2小时)"
echo "预计总时间: ~5小时"
echo ""

# Phase 1: SEG任务验证
echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}Phase 1: SEG任务验证 (6类 × 3个k值 = 18任务)${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python retrain_key_classes_seg.py 2>&1 | tee logs/phase1_seg.log

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Phase 1 完成！${NC}"
    echo ""
    
    # 显示结果预览
    if [ -f "result/prompt1_fixed/key_classes_seg_results.csv" ]; then
        echo "结果预览："
        python -c "import pandas as pd; df=pd.read_csv('result/prompt1_fixed/key_classes_seg_results.csv'); print(df.to_string(index=False))" || true
        echo ""
    fi
    
    echo -e "${YELLOW}是否继续Phase 2和Phase 3？[Y/n]${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Nn]$ ]]; then
        
        # Phase 2: MVTec完整训练
        echo ""
        echo -e "${YELLOW}================================================${NC}"
        echo -e "${YELLOW}Phase 2: MVTec完整训练 (15类 × 2任务 × 3k = 90任务)${NC}"
        echo -e "${YELLOW}================================================${NC}"
        echo ""
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "后台运行: logs/phase2_mvtec.log"
        echo ""
        
        nohup python retrain_mvtec_all.py > logs/phase2_mvtec.log 2>&1 &
        MVTEC_PID=$!
        echo "MVTec PID: $MVTEC_PID"
        
        # Phase 3: VisA完整训练
        echo ""
        echo -e "${YELLOW}================================================${NC}"
        echo -e "${YELLOW}Phase 3: VisA完整训练 (12类 × 2任务 × 3k = 72任务)${NC}"
        echo -e "${YELLOW}================================================${NC}"
        echo ""
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "后台运行: logs/phase3_visa.log"
        echo ""
        
        nohup python retrain_visa_all.py > logs/phase3_visa.log 2>&1 &
        VISA_PID=$!
        echo "VisA PID: $VISA_PID"
        
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Phase 2和3已在后台启动${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "监控命令："
        echo "  tail -f logs/phase2_mvtec.log"
        echo "  tail -f logs/phase3_visa.log"
        echo ""
        echo "检查进度："
        echo "  ps -p $MVTEC_PID  # MVTec"
        echo "  ps -p $VISA_PID   # VisA"
        echo ""
        echo "停止任务："
        echo "  kill $MVTEC_PID  # 停止MVTec"
        echo "  kill $VISA_PID   # 停止VisA"
        echo ""
        echo "预计完成时间: $(date -d '+5 hours' '+%Y-%m-%d %H:%M:%S')"
        
    else
        echo -e "${YELLOW}Phase 2和3已跳过。可稍后手动运行：${NC}"
        echo "  python retrain_mvtec_all.py"
        echo "  python retrain_visa_all.py"
    fi
    
else
    echo ""
    echo -e "${RED}✗ Phase 1 失败！${NC}"
    echo "请检查日志: logs/phase1_seg.log"
    exit 1
fi

echo ""
echo -e "${GREEN}脚本执行完成！${NC}"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
