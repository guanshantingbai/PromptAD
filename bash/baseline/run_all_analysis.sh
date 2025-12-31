#!/bin/bash
# Baseline Analysis Script for PromptAD Classification Task
# Analyzes all trained classification models and generates comprehensive reports
#
# Usage: bash bash/baseline/run_all_analysis.sh <dataset> <k_shot>
# Example: bash bash/baseline/run_all_analysis.sh mvtec 2

set -e  # Exit on error

DATASET=${1:-"mvtec"}
K_SHOT=${2:-2}
CHECKPOINT_DIR="./result/baseline/${DATASET}/k_${K_SHOT}/checkpoint"
OUTPUT_DIR="./result/baseline/${DATASET}/k_${K_SHOT}/analysis"

echo "=========================================="
echo "PromptAD Baseline Analysis (Classification)"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

# Check if checkpoint directory exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "❌ Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo "Please train the models first using run_cls.py"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Get all checkpoint files
CHECKPOINTS=$(find ${CHECKPOINT_DIR} -name "*.pt" -type f 2>/dev/null | sort)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No .pt checkpoint files found in ${CHECKPOINT_DIR}"
    exit 1
fi

# Count total checkpoints
TOTAL=$(echo "$CHECKPOINTS" | wc -l)
echo "Found ${TOTAL} checkpoints to analyze"
echo ""

# Counter
COUNT=0
FAILED=0

# Analyze each checkpoint
for CKPT_PATH in $CHECKPOINTS; do
    COUNT=$((COUNT + 1))
    
    # Extract class name from checkpoint filename
    # Format: CLS-Seed_111-CLASS_NAME-check_point.pt
    CKPT_FILENAME=$(basename ${CKPT_PATH})
    CLASS_NAME=$(echo "${CKPT_FILENAME}" | sed 's/CLS-Seed_[0-9]*-//;s/-check_point\.pt$//')
    
    echo "----------------------------------------"
    echo "[${COUNT}/${TOTAL}] Analyzing: ${CLASS_NAME}"
    echo "----------------------------------------"
    echo "Checkpoint: ${CKPT_PATH}"
    
    # Run analysis
    if python test_baseline_analysis.py \
        --dataset ${DATASET} \
        --k-shot ${K_SHOT} \
        --class_name ${CLASS_NAME} \
        --checkpoint-path ${CKPT_PATH} \
        --root-dir "./result/baseline" \
        --gpu-id 0; then
        echo "✅ Completed: ${CLASS_NAME}"
    else
        echo "❌ Failed: ${CLASS_NAME}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=========================================="
echo "Batch Analysis Summary"
echo "=========================================="
echo "Total checkpoints: ${TOTAL}"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: ${FAILED}"
echo ""

if [ ${FAILED} -eq 0 ]; then
    echo "✅ All analysis completed successfully!"
else
    echo "⚠️  Some analyses failed. Check the output above."
fi

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# Generate summary report if analysis succeeded
if [ ${FAILED} -eq ${TOTAL} ]; then
    echo "❌ All analyses failed. Skipping summary generation."
    exit 1
fi

echo ""
echo "Generating summary report..."

if python analysis/baseline/generate_summary.py \
    --analysis-dir ${OUTPUT_DIR} \
    --output ${OUTPUT_DIR}/summary_report.csv; then
    echo "✅ Summary report: ${OUTPUT_DIR}/summary_report.csv"
else
    echo "⚠️  Failed to generate summary report."
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
