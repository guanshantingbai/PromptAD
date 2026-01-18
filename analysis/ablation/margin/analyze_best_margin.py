#!/usr/bin/env python3
"""
Analyze best margin values for each class based on fusion (i_roc) and semantic metrics
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# Margin values
MARGINS = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2]

def analyze_best_margins(csv_path, dataset_name):
    """Analyze best margin for each class"""
    
    # Read CSV with proper handling of spaces in column names
    df = pd.read_csv(csv_path)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # Set first column as index
    df = df.set_index(df.columns[0])
    df.index.name = 'class_name'
    
    # Remove AVERAGE row for per-class analysis
    df_classes = df[df.index != 'AVERAGE']
    
    print(f"\n{'='*100}")
    print(f"{dataset_name.upper()} Dataset - Best Margin Analysis")
    print(f"{'='*100}\n")
    
    # Analyze Fusion (i_roc)
    print(f"{'Class':<15} {'Best Fusion Margin':<20} {'Fusion Value':<15} {'Best Semantic Margin':<22} {'Semantic Value'}")
    print("-" * 100)
    
    fusion_best_margins = []
    semantic_best_margins = []
    
    for class_name in df_classes.index:
        # Find best fusion margin
        fusion_values = {}
        semantic_values = {}
        
        for margin in MARGINS:
            fusion_col = f'margin_{margin}_fusion'
            semantic_col = f'margin_{margin}_semantic'
            
            if fusion_col in df_classes.columns:
                fusion_values[margin] = df_classes.loc[class_name, fusion_col]
            if semantic_col in df_classes.columns:
                semantic_values[margin] = df_classes.loc[class_name, semantic_col]
        
        # Best fusion
        best_fusion_margin = max(fusion_values.items(), key=lambda x: x[1])
        fusion_best_margins.append(best_fusion_margin[0])
        
        # Best semantic
        best_semantic_margin = max(semantic_values.items(), key=lambda x: x[1])
        semantic_best_margins.append(best_semantic_margin[0])
        
        print(f"{class_name:<15} {best_fusion_margin[0]:<20.1f} {best_fusion_margin[1]:<15.2f} "
              f"{best_semantic_margin[0]:<22.1f} {best_semantic_margin[1]:.2f}")
    
    print()
    
    # Statistics
    fusion_counter = Counter(fusion_best_margins)
    semantic_counter = Counter(semantic_best_margins)
    
    print(f"\n{'='*100}")
    print(f"Statistics Summary - {dataset_name.upper()}")
    print(f"{'='*100}\n")
    
    print("Fusion (i_roc) - Best Margin Distribution:")
    for margin in sorted(fusion_counter.keys()):
        count = fusion_counter[margin]
        percentage = count / len(fusion_best_margins) * 100
        bar = '█' * int(percentage / 5)
        print(f"  margin={margin:<4}  Count: {count:>2}/{len(fusion_best_margins)}  ({percentage:>5.1f}%)  {bar}")
    
    print()
    
    print("Semantic - Best Margin Distribution:")
    for margin in sorted(semantic_counter.keys()):
        count = semantic_counter[margin]
        percentage = count / len(semantic_best_margins) * 100
        bar = '█' * int(percentage / 5)
        print(f"  margin={margin:<4}  Count: {count:>2}/{len(semantic_best_margins)}  ({percentage:>5.1f}%)  {bar}")
    
    print()
    
    # Most common
    most_common_fusion = fusion_counter.most_common(1)[0]
    most_common_semantic = semantic_counter.most_common(1)[0]
    
    print(f"Most Common Best Margin:")
    print(f"  Fusion:   margin={most_common_fusion[0]} ({most_common_fusion[1]}/{len(fusion_best_margins)} classes, {most_common_fusion[1]/len(fusion_best_margins)*100:.1f}%)")
    print(f"  Semantic: margin={most_common_semantic[0]} ({most_common_semantic[1]}/{len(semantic_best_margins)} classes, {most_common_semantic[1]/len(semantic_best_margins)*100:.1f}%)")
    
    # Check if same
    if most_common_fusion[0] == most_common_semantic[0]:
        print(f"\n  ✓ Fusion and Semantic agree on best margin: {most_common_fusion[0]}")
    else:
        print(f"\n  ✗ Fusion and Semantic disagree (Fusion: {most_common_fusion[0]}, Semantic: {most_common_semantic[0]})")
    
    # Average performance at each margin
    print(f"\n{'='*100}")
    print(f"Average Performance at Each Margin - {dataset_name.upper()}")
    print(f"{'='*100}\n")
    
    # Find AVERAGE row (might have extra spaces)
    avg_idx = None
    for idx in df.index:
        if 'AVERAGE' in str(idx).upper().strip():
            avg_idx = idx
            break
    
    if avg_idx is None:
        print("Warning: AVERAGE row not found in dataset")
        return {
            'fusion_best_margins': fusion_best_margins,
            'semantic_best_margins': semantic_best_margins,
            'fusion_counter': fusion_counter,
            'semantic_counter': semantic_counter,
            'most_common_fusion': most_common_fusion,
            'most_common_semantic': most_common_semantic
        }
    
    avg_row = df.loc[avg_idx]
    
    print(f"{'Margin':<10} {'Fusion (i_roc)':<20} {'Semantic':<20} {'Rank (Fusion)':<18} {'Rank (Semantic)'}")
    print("-" * 90)
    
    # Calculate rankings
    fusion_avgs = [(m, avg_row[f'margin_{m}_fusion']) for m in MARGINS]
    semantic_avgs = [(m, avg_row[f'margin_{m}_semantic']) for m in MARGINS]
    
    fusion_avgs_sorted = sorted(fusion_avgs, key=lambda x: x[1], reverse=True)
    semantic_avgs_sorted = sorted(semantic_avgs, key=lambda x: x[1], reverse=True)
    
    fusion_ranks = {m: i+1 for i, (m, _) in enumerate(fusion_avgs_sorted)}
    semantic_ranks = {m: i+1 for i, (m, _) in enumerate(semantic_avgs_sorted)}
    
    for margin in MARGINS:
        fusion_val = avg_row[f'margin_{margin}_fusion']
        semantic_val = avg_row[f'margin_{margin}_semantic']
        f_rank = fusion_ranks[margin]
        s_rank = semantic_ranks[margin]
        
        marker = ''
        if f_rank == 1 and s_rank == 1:
            marker = ' ★★★'
        elif f_rank == 1:
            marker = ' ★F'
        elif s_rank == 1:
            marker = ' ★S'
        
        print(f"{margin:<10.1f} {fusion_val:<20.2f} {semantic_val:<20.2f} "
              f"{f_rank:<18} {s_rank}{marker}")
    
    print()
    
    return {
        'fusion_best_margins': fusion_best_margins,
        'semantic_best_margins': semantic_best_margins,
        'fusion_counter': fusion_counter,
        'semantic_counter': semantic_counter,
        'most_common_fusion': most_common_fusion,
        'most_common_semantic': most_common_semantic
    }

def main():
    base_dir = Path('analysis/ablation/margin')
    
    # Analyze MVTec
    mvtec_results = analyze_best_margins(
        base_dir / 'mvtec_margin_ablation.csv',
        'MVTec'
    )
    
    # Analyze ViSA
    visa_results = analyze_best_margins(
        base_dir / 'visa_margin_ablation.csv',
        'ViSA'
    )
    
    # Overall conclusion
    print(f"\n{'='*100}")
    print("OVERALL CONCLUSION")
    print(f"{'='*100}\n")
    
    print("MVTec Dataset (15 classes):")
    print(f"  Best for Fusion:   margin={mvtec_results['most_common_fusion'][0]} "
          f"({mvtec_results['most_common_fusion'][1]}/15 classes)")
    print(f"  Best for Semantic: margin={mvtec_results['most_common_semantic'][0]} "
          f"({mvtec_results['most_common_semantic'][1]}/15 classes)")
    print()
    
    print("ViSA Dataset (12 classes):")
    print(f"  Best for Fusion:   margin={visa_results['most_common_fusion'][0]} "
          f"({visa_results['most_common_fusion'][1]}/12 classes)")
    print(f"  Best for Semantic: margin={visa_results['most_common_semantic'][0]} "
          f"({visa_results['most_common_semantic'][1]}/12 classes)")
    print()
    
    # Combined analysis
    all_fusion_margins = mvtec_results['fusion_best_margins'] + visa_results['fusion_best_margins']
    all_semantic_margins = mvtec_results['semantic_best_margins'] + visa_results['semantic_best_margins']
    
    combined_fusion_counter = Counter(all_fusion_margins)
    combined_semantic_counter = Counter(all_semantic_margins)
    
    most_common_fusion_overall = combined_fusion_counter.most_common(1)[0]
    most_common_semantic_overall = combined_semantic_counter.most_common(1)[0]
    
    print("Combined Analysis (27 classes):")
    print(f"  Best for Fusion:   margin={most_common_fusion_overall[0]} "
          f"({most_common_fusion_overall[1]}/27 classes, {most_common_fusion_overall[1]/27*100:.1f}%)")
    print(f"  Best for Semantic: margin={most_common_semantic_overall[0]} "
          f"({most_common_semantic_overall[1]}/27 classes, {most_common_semantic_overall[1]/27*100:.1f}%)")
    print()
    
    # Final recommendation
    print(f"{'='*100}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*100}\n")
    
    if most_common_fusion_overall[0] == most_common_semantic_overall[0]:
        print(f"✓ UNANIMOUS: margin={most_common_fusion_overall[0]} is best for both Fusion and Semantic")
        print(f"  - Fusion: {most_common_fusion_overall[1]}/27 classes prefer this margin")
        print(f"  - Semantic: {most_common_semantic_overall[1]}/27 classes prefer this margin")
    else:
        print(f"✗ SPLIT DECISION:")
        print(f"  - Fusion prefers:   margin={most_common_fusion_overall[0]} ({most_common_fusion_overall[1]}/27 classes)")
        print(f"  - Semantic prefers: margin={most_common_semantic_overall[0]} ({most_common_semantic_overall[1]}/27 classes)")
        print()
        print(f"  Recommendation: Consider margin={most_common_fusion_overall[0]} as Fusion is typically the primary metric")
    
    print()
    print("Note: This analysis is based on per-class best performance.")
    print("For overall average performance, refer to the AVERAGE row in the CSV files.")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
