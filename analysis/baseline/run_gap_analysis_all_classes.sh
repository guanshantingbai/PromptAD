#!/bin/bash

# Gap analysis for ALL classes (MVTec + VisA)

echo "======================================================"
echo "Running Gap Analysis on All Classes"
echo "======================================================"

# All MVTec-AD classes (15)
MVTEC_CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

# All VisA classes (12)
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1"
    "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"
)

# Combine all classes
CLASSES=()
for class_name in "${MVTEC_CLASSES[@]}"; do
    CLASSES+=("mvtec:${class_name}")
done
for class_name in "${VISA_CLASSES[@]}"; do
    CLASSES+=("visa:${class_name}")
done

echo "Total classes to analyze: ${#CLASSES[@]} (15 MVTec + 12 VisA)"
echo ""

for class_spec in "${CLASSES[@]}"; do
    IFS=':' read -r dataset class_name <<< "$class_spec"
    
    echo ""
    echo "======================================================"
    echo "Processing: ${dataset} - ${class_name}"
    echo "======================================================"
    
    # Step 1: Run test to collect gap statistics (from project root)
    echo "[1/2] Collecting gap statistics..."
    cd ../..
    python test_cls.py \
        --dataset ${dataset} \
        --class_name ${class_name} \
        --root-dir ./result/baseline \
        --k-shot 2 \
        --gpu-id 0 \
        --use-lap True \
        --seed 111
    
    if [ $? -ne 0 ]; then
        echo "ERROR: test_cls.py failed for ${dataset}-${class_name}"
        cd analysis/baseline
        continue
    fi
    
    # Step 2: Analyze gap informativeness (from analysis/baseline)
    cd analysis/baseline
    echo "[2/2] Analyzing gap informativeness..."
    python analyze_gap_informativeness.py \
        --dataset ${dataset} \
        --class_name ${class_name} \
        --root-dir ../../result/test_gate
    
    if [ $? -ne 0 ]; then
        echo "ERROR: gap analysis failed for ${dataset}-${class_name}"
        continue
    fi
    
    echo "✓ Completed: ${dataset}-${class_name}"
done

echo ""
echo "======================================================"
echo "All gap analyses completed!"
echo "======================================================"
echo ""
echo "Generating summary CSV..."
python summarize_gap_screening.py

echo ""
echo "Results saved in:"
echo "  - Individual: ../../result/test_gate/{dataset}/semantic_gap/"
echo "  - Summary CSV: ../../result/test_gate/summary/gap_screening_all.csv"
echo ""

