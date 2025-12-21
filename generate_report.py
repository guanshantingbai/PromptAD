#!/usr/bin/env python3
"""
Generate a comprehensive results report in Markdown format.
"""

import pandas as pd
from pathlib import Path
import sys


def generate_markdown_report(csv_file, output_file=None):
    """Generate Markdown report from consolidated CSV."""
    
    df = pd.read_csv(csv_file)
    
    if output_file is None:
        output_file = csv_file.replace('.csv', '_report.md')
    
    with open(output_file, 'w') as f:
        f.write("# Multi-Prototype PromptAD Results Report\n\n")
        f.write("## Overview\n\n")
        f.write("Training configuration:\n")
        f.write("- **Method**: Multi-Prototype Semantic Discrimination\n")
        f.write("- **Normal Prototypes**: 3 (multi-modal normal manifold)\n")
        f.write("- **Abnormal Prototypes**: 6 (structured abnormal directions)\n")
        f.write("- **Training**: 100 epochs, lr=0.002\n")
        f.write("- **Memory Bank**: Removed (pure semantic discrimination)\n\n")
        
        f.write("---\n\n")
        f.write("## Results Summary (Image-level AUROC %)\n\n")
        
        # Get average row
        avg_row = df[df['class'] == 'AVERAGE'].iloc[0]
        
        f.write("### Overall Performance\n\n")
        f.write("| K-Shot | Average AUROC |\n")
        f.write("|--------|---------------|\n")
        
        for col in df.columns:
            if col.startswith('i_roc_k'):
                k_shot = col.split('_k')[1]
                value = avg_row[col]
                f.write(f"| {k_shot} | {value:.2f}% |\n")
        
        f.write("\n")
        f.write("**Key Observations**:\n")
        f.write(f"- Performance improves with more shots: {avg_row['i_roc_k1']:.2f}% → {avg_row['i_roc_k2']:.2f}% → {avg_row['i_roc_k4']:.2f}%\n")
        f.write(f"- Best average: **{avg_row['i_roc_k4']:.2f}%** at k=4\n\n")
        
        f.write("---\n\n")
        f.write("## Detailed Results by Class\n\n")
        
        # Per-class table
        f.write("### Image-level AUROC (%) - All K-Shots\n\n")
        f.write("| Class | k=1 | k=2 | k=4 |\n")
        f.write("|-------|-----|-----|-----|\n")
        
        for _, row in df.iterrows():
            if row['class'] == 'AVERAGE':
                f.write("|-------|-----|-----|-----|\n")
            f.write(f"| **{row['class']}** | {row['i_roc_k1']:.2f} | {row['i_roc_k2']:.2f} | {row['i_roc_k4']:.2f} |\n")
        
        f.write("\n")
        
        # Top performers
        f.write("### Top Performing Classes (k=2)\n\n")
        df_classes = df[df['class'] != 'AVERAGE'].copy()
        df_classes = df_classes.sort_values('i_roc_k2', ascending=False)
        
        f.write("| Rank | Class | AUROC |\n")
        f.write("|------|-------|-------|\n")
        
        for idx, (_, row) in enumerate(df_classes.head(5).iterrows(), 1):
            f.write(f"| {idx} | {row['class']} | {row['i_roc_k2']:.2f}% |\n")
        
        f.write("\n")
        
        # Challenging classes
        f.write("### Most Challenging Classes (k=2)\n\n")
        f.write("| Rank | Class | AUROC |\n")
        f.write("|------|-------|-------|\n")
        
        for idx, (_, row) in enumerate(df_classes.tail(5).iloc[::-1].iterrows(), 1):
            f.write(f"| {idx} | {row['class']} | {row['i_roc_k2']:.2f}% |\n")
        
        f.write("\n")
        
        f.write("---\n\n")
        f.write("## Analysis\n\n")
        
        # Perfect scores
        perfect = df_classes[df_classes['i_roc_k2'] == 100.0]['class'].tolist()
        if perfect:
            f.write(f"### Perfect Performance (100% AUROC at k=2)\n")
            f.write(f"- {', '.join(perfect)}\n\n")
        
        # Above 95%
        excellent = df_classes[(df_classes['i_roc_k2'] >= 95) & (df_classes['i_roc_k2'] < 100)]['class'].tolist()
        if excellent:
            f.write(f"### Excellent Performance (≥95% at k=2)\n")
            f.write(f"- {', '.join(excellent)}\n\n")
        
        # Below 85%
        challenging = df_classes[df_classes['i_roc_k2'] < 85]['class'].tolist()
        if challenging:
            f.write(f"### Challenging Cases (<85% at k=2)\n")
            f.write(f"- {', '.join(challenging)}\n")
            f.write(f"- These may benefit from:\n")
            f.write(f"  - More training epochs\n")
            f.write(f"  - More prototypes (n_pro > 3)\n")
            f.write(f"  - Higher k-shot (see k=4 results)\n\n")
        
        f.write("---\n\n")
        f.write("## Multi-Prototype Benefits\n\n")
        f.write("1. **Multi-modal Normal Manifold**: 3 normal prototypes capture diverse normal states\n")
        f.write("2. **Structured Abnormal Directions**: 6 abnormal prototypes represent semantic defect types\n")
        f.write("3. **No Memory Bank**: Pure semantic discrimination (faster, no visual gallery)\n")
        f.write("4. **Scalability**: Performance improves with more shots\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated from: {csv_file}*\n")
    
    print(f"\n{'='*80}")
    print(f"Markdown report saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        csv_file = "result/prompt1/mvtec_consolidated_results.csv"
    else:
        csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)
    
    generate_markdown_report(csv_file)
