#!/bin/bash

# Test Phase 2.1 on a single class to verify implementation
# This script will test on mvtec/screw (the class with highest oracle gain)

echo "=============================================="
echo "Phase 2.1 Test: Single Class (mvtec/screw)"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Load checkpoint from result/gate"
echo "  2. Compute per-sample reliability indicators"
echo "  3. Save to result/gate2/mvtec/k_4/per_sample/"
echo "  4. Run analysis with real data"
echo ""
echo "Expected time: ~5 minutes"
echo "=============================================="
echo ""

# Configuration
DEVICE="cuda:0"
DATASET="mvtec"
CLASS="screw"
K_SHOT=4
TASK="cls"
OUTPUT_DIR="result/gate2"
CHECKPOINT_DIR="result/gate"

# Step 1: Run inference with indicator computation
echo "[Step 1/2] Running gate experiment with indicator computation..."
python run_gate_experiment.py \
    --dataset $DATASET \
    --class_name $CLASS \
    --k-shot $K_SHOT \
    --task $TASK \
    --gpu-id 0 \
    --root-dir $OUTPUT_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --seed 111 \
    --backbone ViT-B-16-plus-240 \
    --pretrained_dataset laion400m_e32

if [ $? -ne 0 ]; then
    echo "❌ Gate experiment failed"
    exit 1
fi

echo ""
echo "✅ Inference complete!"
echo ""

# Step 2: Run analysis
echo "[Step 2/2] Running indicator-oracle correlation analysis..."

# Create a custom class list for single-class analysis
python -c "
import sys
sys.path.insert(0, '.')
from phase2_1_oracle_correlation import *
import argparse

# Override class lists to only include screw
class_lists = {
    'mvtec': ['screw'],
    'visa': []
}

# Create args
class Args:
    result_dir = 'result/gate2'
    output_dir = 'result/gate2/test_single'
    datasets = ['mvtec']
    k_shots = [4]
    task = 'cls'
    use_real_data = True

args = Args()

# Create output directory
import os
os.makedirs(args.output_dir, exist_ok=True)

print('='*80)
print('Phase 2.1 Analysis: mvtec/screw')
print('='*80)
print(f'Result directory: {args.result_dir}')
print(f'Output directory: {args.output_dir}')
print(f'Use real data: {args.use_real_data}')
print('')

# Load data
results_dict = {}
for dataset in args.datasets:
    for k_shot in args.k_shots:
        for class_name in class_lists.get(dataset, []):
            data = load_oracle_and_indicators(
                args.result_dir, dataset, k_shot, args.task, class_name,
                use_real_data=args.use_real_data
            )
            if data is not None:
                key = (dataset, k_shot, args.task, class_name)
                results_dict[key] = data

if not results_dict:
    print('❌ No data loaded. Check if per_sample file exists.')
    sys.exit(1)

print(f'✅ Loaded data for {len(results_dict)} class')

# Analyze
for key, data in results_dict.items():
    if not data.get('has_per_sample_data', False):
        print('❌ No per-sample data found')
        continue
    
    dataset, k_shot, task, class_name = key
    print(f'\n{'='*60}')
    print(f'{dataset}/{class_name} (k={k_shot})')
    print(f'{'='*60}')
    
    oracle_choices = data['oracle_choices']
    indicators = data['indicators']
    
    print(f'  Samples: {len(oracle_choices)}')
    print(f'  Semantic ratio: {(oracle_choices == 0).mean()*100:.1f}%')
    print(f'  Memory ratio: {(oracle_choices == 1).mean()*100:.1f}%')
    
    # Task 2: Distribution
    print(f'\n  📊 Task 2: Indicator Distribution')
    for name, values in indicators.items():
        sem_mask = oracle_choices == 0
        mem_mask = oracle_choices == 1
        sem_vals = values[sem_mask]
        mem_vals = values[mem_mask]
        
        if len(sem_vals) > 0 and len(mem_vals) > 0:
            from scipy import stats as sp_stats
            pooled_std = np.sqrt((sem_vals.std()**2 + mem_vals.std()**2) / 2)
            cohens_d = (mem_vals.mean() - sem_vals.mean()) / (pooled_std + 1e-8)
            t_stat, p_value = sp_stats.ttest_ind(sem_vals, mem_vals)
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f'    {name:20s}: d={cohens_d:+.3f}, p={p_value:.4f} {sig}')
    
    # Task 3: AUC
    print(f'\n  📊 Task 3: Oracle Predictability (AUC)')
    auc_scores = {}
    
    for name, values in indicators.items():
        if name.startswith('r_sem'):
            y_score = -values
        else:
            y_score = values
        
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(oracle_choices)) > 1 and len(np.unique(y_score)) > 1:
                auc = roc_auc_score(oracle_choices, y_score)
                auc_scores[name] = auc
                print(f'    {name:20s}: AUC = {auc:.3f}')
        except:
            pass
    
    # Average
    mem_auc = np.mean([auc_scores.get(f'r_mem_{x}', 0.5) for x in ['margin', 'entropy', 'centroid']])
    sem_auc = np.mean([auc_scores.get(f'r_sem_{x}', 0.5) for x in ['prompt_var', 'prompt_margin', 'extremity']])
    
    print(f'\n    Memory avg AUC:   {mem_auc:.3f}')
    print(f'    Semantic avg AUC: {sem_auc:.3f}')
    print(f'    Overall avg AUC:  {(mem_auc + sem_auc)/2:.3f}')
    
    print(f'\n{'='*60}')
    print('DECISION:')
    avg = (mem_auc + sem_auc) / 2
    if avg > 0.60:
        print('✅ AUC > 0.60: Indicators correlate with oracle!')
        print('   → Proceed to Phase 2.2 (adaptive gating)')
    elif avg > 0.55:
        print('⚠️  AUC 0.55-0.60: Weak correlation')
        print('   → Consider improving indicators')
    else:
        print('❌ AUC < 0.55: No correlation')
        print('   → Revisit indicator design')

print(f'\n{'='*80}')
print('Single Class Test Complete')
print(f'{'='*80}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test successful!"
    echo ""
    echo "Next steps:"
    echo "  1. If test passed → run all 5 classes: ./run_phase2_1_real.sh"
    echo "  2. Review results in result/gate2/test_single/"
else
    echo ""
    echo "❌ Test failed. Check error messages above."
fi
