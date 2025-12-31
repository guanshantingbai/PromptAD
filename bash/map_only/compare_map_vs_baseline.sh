#!/bin/bash
# Compare MAP-only vs Baseline (MAP+LAP) performance

set -e

DATASET=${1:-"visa"}
K_SHOT=${2:-2}

echo "=========================================="
echo "MAP-only vs Baseline Comparison"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "=========================================="
echo ""

# Check if both results exist
BASELINE_DIR="./result/baseline/${DATASET}/k_${K_SHOT}/csv"
MAP_ONLY_DIR="./result/map_only/${DATASET}/k_${K_SHOT}/csv"

if [ ! -f "${BASELINE_DIR}/Seed_111-results.csv" ]; then
    echo "❌ Baseline results not found: ${BASELINE_DIR}/Seed_111-results.csv"
    exit 1
fi

if [ ! -f "${MAP_ONLY_DIR}/Seed_111-results.csv" ]; then
    echo "❌ MAP-only results not found: ${MAP_ONLY_DIR}/Seed_111-results.csv"
    exit 1
fi

# Run comparison analysis
python << 'EOF'
import pandas as pd
import sys

dataset = sys.argv[1] if len(sys.argv) > 1 else "visa"
k_shot = sys.argv[2] if len(sys.argv) > 2 else "2"

# Load data
baseline_df = pd.read_csv(f'./result/baseline/{dataset}/k_{k_shot}/csv/Seed_111-results.csv', index_col=0)
map_only_df = pd.read_csv(f'./result/map_only/{dataset}/k_{k_shot}/csv/Seed_111-results.csv', index_col=0)

print("=" * 100)
print(f"MAP-only vs Baseline (MAP+LAP) Comparison - {dataset.upper()} k={k_shot}")
print("=" * 100)
print()

# Align indices
common_classes = baseline_df.index.intersection(map_only_df.index)
baseline_df = baseline_df.loc[common_classes]
map_only_df = map_only_df.loc[common_classes]

print(f"{'Class':<20} {'Baseline (MAP+LAP)':<25} {'MAP-only':<25} {'Δ (MAP-only - Baseline)'}")
print(f"{'':20} {'Sem':>7} {'Mem':>7} {'Fus':>7}   {'Sem':>7} {'Mem':>7} {'Fus':>7}   {'Sem':>7} {'Fus':>7}")
print("-" * 100)

improvements = []
degradations = []

for cls in common_classes:
    baseline_sem = baseline_df.loc[cls, 'semantic_i_roc']
    baseline_mem = baseline_df.loc[cls, 'memory_i_roc']
    baseline_fus = baseline_df.loc[cls, 'fusion_i_roc']
    
    map_only_sem = map_only_df.loc[cls, 'semantic_i_roc']
    map_only_mem = map_only_df.loc[cls, 'memory_i_roc']
    map_only_fus = map_only_df.loc[cls, 'fusion_i_roc']
    
    delta_sem = map_only_sem - baseline_sem
    delta_fus = map_only_fus - baseline_fus
    
    # Track improvements
    if delta_sem > 1.0:  # >1% improvement
        improvements.append((cls, delta_sem, delta_fus))
    elif delta_sem < -1.0:  # >1% degradation
        degradations.append((cls, delta_sem, delta_fus))
    
    delta_sem_str = f"{delta_sem:+6.2f}"
    delta_fus_str = f"{delta_fus:+6.2f}"
    
    print(f"{cls:<20} {baseline_sem:>7.2f} {baseline_mem:>7.2f} {baseline_fus:>7.2f}   "
          f"{map_only_sem:>7.2f} {map_only_mem:>7.2f} {map_only_fus:>7.2f}   "
          f"{delta_sem_str:>7} {delta_fus_str:>7}")

print("-" * 100)

# Calculate averages
avg_baseline_sem = baseline_df['semantic_i_roc'].mean()
avg_baseline_fus = baseline_df['fusion_i_roc'].mean()
avg_map_only_sem = map_only_df['semantic_i_roc'].mean()
avg_map_only_fus = map_only_df['fusion_i_roc'].mean()

delta_avg_sem = avg_map_only_sem - avg_baseline_sem
delta_avg_fus = avg_map_only_fus - avg_baseline_fus

print(f"{'AVERAGE':<20} {avg_baseline_sem:>7.2f} {'-':>7} {avg_baseline_fus:>7.2f}   "
      f"{avg_map_only_sem:>7.2f} {'-':>7} {avg_map_only_fus:>7.2f}   "
      f"{delta_avg_sem:>+7.2f} {delta_avg_fus:>+7.2f}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print()

if delta_avg_sem > 0:
    print(f"✅ MAP-only IMPROVES semantic performance: {delta_avg_sem:+.2f}% on average")
else:
    print(f"❌ MAP-only DEGRADES semantic performance: {delta_avg_sem:+.2f}% on average")

print()
print(f"Classes with improvement (>1%): {len(improvements)}")
if improvements:
    improvements.sort(key=lambda x: x[1], reverse=True)
    for cls, delta_sem, delta_fus in improvements[:5]:
        print(f"  • {cls:<20} Δ Semantic: {delta_sem:+.2f}%")

print()
print(f"Classes with degradation (<-1%): {len(degradations)}")
if degradations:
    degradations.sort(key=lambda x: x[1])
    for cls, delta_sem, delta_fus in degradations[:5]:
        print(f"  • {cls:<20} Δ Semantic: {delta_sem:+.2f}%")

print()
print("=" * 100)

# Save comparison
output_dir = f'./result/comparison/map_vs_baseline/{dataset}/k_{k_shot}'
import os
os.makedirs(output_dir, exist_ok=True)

comparison_df = pd.DataFrame({
    'baseline_semantic': baseline_df['semantic_i_roc'],
    'baseline_fusion': baseline_df['fusion_i_roc'],
    'map_only_semantic': map_only_df['semantic_i_roc'],
    'map_only_fusion': map_only_df['fusion_i_roc'],
    'delta_semantic': map_only_df['semantic_i_roc'] - baseline_df['semantic_i_roc'],
    'delta_fusion': map_only_df['fusion_i_roc'] - baseline_df['fusion_i_roc'],
})

comparison_df.to_csv(f'{output_dir}/comparison.csv', float_format='%.2f')
print(f"Detailed comparison saved to: {output_dir}/comparison.csv")

EOF

echo ""
echo "=========================================="
