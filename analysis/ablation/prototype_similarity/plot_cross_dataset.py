#!/usr/bin/env python3
"""
Plot cross-dataset prototype similarity comparison (English version)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('analysis/ablation/prototype_similarity/cross_dataset_summary.csv')

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Color scheme
colors = ['#3498db', '#e74c3c', '#2ecc71']
dataset_labels = df['dataset'].tolist()

# ===== Subplot 1: Prototype Similarity =====
ax1 = axes[0]
bars1 = ax1.bar(range(len(df)), df['mean_sim'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(dataset_labels, fontsize=11, rotation=15, ha='right')
ax1.set_ylabel('Mean Similarity', fontsize=12, fontweight='bold')
ax1.set_title('Prototype Similarity Across Datasets', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, df['mean_sim'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ===== Subplot 2: Mean AUROC Performance =====
ax2 = axes[1]
bars2 = ax2.bar(range(len(df)), df['mean_auroc'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(dataset_labels, fontsize=11, rotation=15, ha='right')
ax2.set_ylabel('Mean AUROC (%)', fontsize=12, fontweight='bold')
ax2.set_title('Detection Performance', fontsize=13, fontweight='bold')
ax2.set_ylim([70, 100])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, df['mean_auroc'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ===== Subplot 3: Correlation Analysis =====
ax3 = axes[2]

# Bar chart for Pearson correlation
x = np.arange(len(df))
bars3 = ax3.bar(x, df['pearson_r'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax3.set_xticks(x)
ax3.set_xticklabels(dataset_labels, fontsize=11, rotation=15, ha='right')
ax3.set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
ax3.set_title('Similarity-Performance Correlation', fontsize=13, fontweight='bold')
ax3.set_ylim([-0.2, 0.8])
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels with p-value
for i, (bar, r, p) in enumerate(zip(bars3, df['pearson_r'], df['pearson_p'])):
    # Determine significance
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    elif p < 0.10:
        sig = '†'  # marginal
    else:
        sig = 'n.s.'
    
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if bar.get_height() > 0 else bar.get_height() - 0.05, 
             f'r={r:.3f}\n{sig}', ha='center', va='bottom' if bar.get_height() > 0 else 'top', 
             fontsize=9, fontweight='bold')

# Add legend for significance
legend_text = '*** p<0.001  ** p<0.01  * p<0.05\n† p<0.10  n.s. not significant'
ax3.text(0.98, 0.02, legend_text, transform=ax3.transAxes, 
         fontsize=8, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Cross-Dataset Prototype Similarity Validation', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('analysis/ablation/prototype_similarity/cross_dataset_validation.png', 
            dpi=300, bbox_inches='tight')
print('✅ Figure saved: analysis/ablation/prototype_similarity/cross_dataset_validation.png')
plt.close()

# Print summary
print()
print('=' * 80)
print('Cross-Dataset Prototype Similarity Summary')
print('=' * 80)
print()
for _, row in df.iterrows():
    print(f'{row["dataset"]:15s}: Similarity={row["mean_sim"]:.3f}, '
          f'AUROC={row["mean_auroc"]:.2f}%, r={row["pearson_r"]:.3f} (p={row["pearson_p"]:.3f})')
print()
print('Key Findings:')
print('  1. MVTec K=2 shows moderate positive correlation (r=0.504, p=0.055)')
print('  2. ViSA K=2 has higher similarity (0.746) but weaker correlation')
print('  3. Prototype consistency is more predictive on MVTec than ViSA')
print('=' * 80)
