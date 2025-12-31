#!/bin/bash

# Run weighted harmonic mean fusion tests for all VisA classes (k=1)
# Formula: fusion = 1 / (1/memory + alpha/semantic)
# Usage: bash run_alpha_visa.sh

DATASET="visa"
K_SHOT=1
OUTPUT_DIR="./result/test_alpha_weighted"

# Alpha values to test (semantic weight in weighted harmonic mean)
# alpha=0.0: ignore semantic (fusion=memory)
# alpha=1.0: equal weights (standard harmonic mean)
# alpha>1.0: semantic has more weight
ALPHA_VALUES=(0.8 0.85 0.9 0.95 1.1)

echo "=========================================="
echo "Running Weighted Harmonic Mean Tests for VisA"
echo "Formula: fusion = 1/(1/memory + alpha/semantic)"
echo "Alpha values: ${ALPHA_VALUES[@]}"
echo "K-shot: $K_SHOT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# VisA classes (12 total)
CLASSES=(
    "candle"
    "capsules"
    "cashew"
    "chewinggum"
    "fryum"
    "macaroni1"
    "macaroni2"
    "pcb1"
    "pcb2"
    "pcb3"
    "pcb4"
    "pipe_fryum"
)

# Loop over each alpha value
for ALPHA in "${ALPHA_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing Alpha = $ALPHA"
    echo "=========================================="
    echo ""
    
    # Remove previous CSV if exists
    CSV_PATH="$OUTPUT_DIR/$DATASET/k_$K_SHOT/csv/$DATASET.csv"
    if [ -f "$CSV_PATH" ]; then
        rm "$CSV_PATH"
        echo "[INFO] Removed previous CSV: $CSV_PATH"
    fi
    
    # Run tests for each class with current alpha
    for class in "${CLASSES[@]}"; do
        echo "----------------------------------------"
        echo "Testing: $class (Alpha=$ALPHA)"
        echo "----------------------------------------"
        
        python test_cls.py \
            --dataset $DATASET \
            --class_name $class \
            --k-shot $K_SHOT \
            --semantic-alpha $ALPHA \
            --output-dir $OUTPUT_DIR \
            --gpu-id 0
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] $class completed"
        else
            echo "[FAILED] $class failed"
        fi
    done
    
    # Rename CSV file with alpha value
    if [ -f "$CSV_PATH" ]; then
        # Format alpha for filename (replace . with empty string for cleaner names)
        ALPHA_STR=$(echo "$ALPHA" | sed 's/\.//g')
        NEW_CSV_PATH="$OUTPUT_DIR/$DATASET/k_$K_SHOT/csv/${DATASET}_alpha${ALPHA_STR}.csv"
        mv "$CSV_PATH" "$NEW_CSV_PATH"
        echo ""
        echo "[INFO] Results saved to: $NEW_CSV_PATH"
        echo ""
    else
        echo "[WARNING] CSV file not found: $CSV_PATH"
    fi
done

echo ""
echo "=========================================="
echo "All alpha grid search completed!"
echo "Results location: $OUTPUT_DIR/$DATASET/k_$K_SHOT/csv/"
echo "Files:"
for ALPHA in "${ALPHA_VALUES[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | sed 's/\.//g')
    echo "  - ${DATASET}_alpha${ALPHA_STR}.csv (alpha=$ALPHA)"
done
echo "=========================================="
