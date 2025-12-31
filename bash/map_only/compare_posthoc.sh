#!/bin/bash

# Compare MAP-only (post-hoc) vs Baseline
# Usage: bash bash/map_only/compare_posthoc.sh <dataset> <k_shot>

DATASET=$1
K_SHOT=$2

if [ -z "$DATASET" ] || [ -z "$K_SHOT" ]; then
    echo "Usage: bash $0 <dataset> <k_shot>"
    echo "Example: bash $0 mvtec 2"
    exit 1
fi

BASELINE_CSV="./result/baseline/${DATASET}/k_${K_SHOT}/csv/Seed_111-results.csv"
MAP_ONLY_CSV="./result/map_only_posthoc/${DATASET}/k_${K_SHOT}/csv/Seed_111-results.csv"
OUTPUT_DIR="./result/comparison/map_vs_baseline_posthoc/${DATASET}_k${K_SHOT}"

echo "==========================================="
echo "MAP-only (Post-hoc) vs Baseline Comparison"
echo "Dataset: $DATASET, K-shot: $K_SHOT"
echo "==========================================="

# Check if results exist
if [ ! -f "$BASELINE_CSV" ]; then
    echo "[ERROR] Baseline results not found: $BASELINE_CSV"
    exit 1
fi

if [ ! -f "$MAP_ONLY_CSV" ]; then
    echo "[ERROR] MAP-only results not found: $MAP_ONLY_CSV"
    echo "[INFO] Run: bash bash/map_only/test_map_only_with_baseline_ckpt.sh $DATASET $K_SHOT 0"
    exit 1
fi

mkdir -p $OUTPUT_DIR

# Python comparison script
python -c "
import pandas as pd
import numpy as np

# Load results
baseline = pd.read_csv('$BASELINE_CSV')
map_only = pd.read_csv('$MAP_ONLY_CSV')

# Ensure class names match
baseline = baseline.sort_values('class').reset_index(drop=True)
map_only = map_only.sort_values('class').reset_index(drop=True)

# Create comparison dataframe
comparison = pd.DataFrame()
comparison['class'] = baseline['class']
comparison['baseline_semantic'] = baseline['semantic']
comparison['baseline_fusion'] = baseline['fusion']
comparison['map_only_semantic'] = map_only['semantic']
comparison['map_only_fusion'] = map_only['fusion']

# Calculate deltas
comparison['delta_semantic'] = map_only['semantic'] - baseline['semantic']
comparison['delta_fusion'] = map_only['fusion'] - baseline['fusion']

# Save detailed comparison
comparison.to_csv('$OUTPUT_DIR/comparison.csv', index=False, float_format='%.2f')

# Print summary
print('\n' + '='*80)
print('MAP-only (Post-hoc) vs Baseline Comparison')
print('='*80)
print(f'Dataset: $DATASET, K-shot: $K_SHOT')
print(f'Mode: Post-hoc evaluation (using baseline checkpoints)')
print('='*80)
print()
print(comparison.to_string(index=False))
print()
print('='*80)
print('Summary Statistics:')
print('='*80)
print(f'Average Baseline Semantic:  {baseline[\"semantic\"].mean():.2f}%')
print(f'Average MAP-only Semantic:  {map_only[\"semantic\"].mean():.2f}%')
print(f'Average Delta Semantic:     {comparison[\"delta_semantic\"].mean():+.2f}%')
print()
print(f'Average Baseline Fusion:    {baseline[\"fusion\"].mean():.2f}%')
print(f'Average MAP-only Fusion:    {map_only[\"fusion\"].mean():.2f}%')
print(f'Average Delta Fusion:       {comparison[\"delta_fusion\"].mean():+.2f}%')
print()
print('='*80)
print('Semantic Branch Changes:')
print('='*80)
improved = comparison[comparison['delta_semantic'] > 1.0]
degraded = comparison[comparison['delta_semantic'] < -1.0]
print(f'Improved (>+1%):  {len(improved)} classes')
if len(improved) > 0:
    for _, row in improved.iterrows():
        print(f'  {row[\"class\"]:20s}: {row[\"baseline_semantic\"]:6.2f}% → {row[\"map_only_semantic\"]:6.2f}% ({row[\"delta_semantic\"]:+.2f}%)')
print()
print(f'Degraded (<-1%):  {len(degraded)} classes')
if len(degraded) > 0:
    for _, row in degraded.iterrows():
        print(f'  {row[\"class\"]:20s}: {row[\"baseline_semantic\"]:6.2f}% → {row[\"map_only_semantic\"]:6.2f}% ({row[\"delta_semantic\"]:+.2f}%)')
print()
print('='*80)
print(f'Results saved to: $OUTPUT_DIR/comparison.csv')
print('='*80)
"

echo ""
echo "Comparison complete!"
