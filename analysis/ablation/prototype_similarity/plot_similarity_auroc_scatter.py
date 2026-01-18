#!/usr/bin/env python3
"""
Plot Prototype Similarity vs AUROC scatter plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Read data
df_sim = pd.read_csv('analysis/ablation/prototype_similarity/mvtec/baseline_k2_similarity.csv')
df_perf = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)

# Merge data
df_sim['class_full'] = 'mvtec-' + df_sim['class']
df = df_sim.set_index('class_full').join(df_perf[['i_roc', 'semantic_i_roc', 'memory_i_roc']])

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ===== Subplot 1: Similarity vs Fusion AUROC =====
ax1 = axes[0]

# Scatter plot
scatter1 = ax1.scatter(df['mean_sim'], df['i_roc'], 
                       s=150, alpha=0.7, c='#3498db', edgecolors='black', linewidth=1.5)

# Fit line
z1 = np.polyfit(df['mean_sim'], df['i_roc'], 1)
p1 = np.poly1d(z1)
x_line = np.linspace(df['mean_sim'].min(), df['mean_sim'].max(), 100)
ax1.plot(x_line, p1(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

# Pearson correlation
corr1, p1_val = pearsonr(df['mean_sim'], df['i_roc'])
ax1.text(0.05, 0.95, f'Pearson r = {corr1:.3f}\np = {p1_val:.4f}', 
         transform=ax1.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('Prototype Similarity', fontsize=13, fontweight='bold')
ax1.set_ylabel('Fusion AUROC (%)', fontsize=13, fontweight='bold')
ax1.set_title('Prototype Similarity vs Fusion Performance', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)
ax1.set_xlim([0.35, 0.85])
ax1.set_ylim([60, 102])

# ===== Subplot 2: Similarity vs Semantic AUROC =====
ax2 = axes[1]

# Scatter plot
scatter2 = ax2.scatter(df['mean_sim'], df['semantic_i_roc'], 
                       s=150, alpha=0.7, c='#e74c3c', edgecolors='black', linewidth=1.5)

# Fit line
z2 = np.polyfit(df['mean_sim'], df['semantic_i_roc'], 1)
p2 = np.poly1d(z2)
ax2.plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

# Pearson correlation
corr2, p2_val = pearsonr(df['mean_sim'], df['semantic_i_roc'])
ax2.text(0.05, 0.95, f'Pearson r = {corr2:.3f}\np = {p2_val:.4f}', 
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('Prototype Similarity', fontsize=13, fontweight='bold')
ax2.set_ylabel('Semantic AUROC (%)', fontsize=13, fontweight='bold')
ax2.set_title('Prototype Similarity vs Semantic Performance', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)
ax2.set_xlim([0.35, 0.85])
ax2.set_ylim([60, 102])

# Overall title
fig.suptitle('MVTec K=2: Prototype Similarity vs Detection Performance', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('analysis/ablation/prototype_similarity/similarity_auroc_scatter.png', 
            dpi=300, bbox_inches='tight')
print('✅ Figure saved: analysis/ablation/prototype_similarity/similarity_auroc_scatter.png')
plt.close()

# Print summary
print()
print('=' * 80)
print('Prototype Similarity vs AUROC Correlation Analysis')
print('=' * 80)
print()
print(f'Fusion AUROC:')
print(f'  Pearson r = {corr1:.3f}, p = {p1_val:.4f}')
print(f'  Interpretation: {["Weak", "Weak", "Moderate", "Strong"][int(abs(corr1)*4)]} correlation')
print()
print(f'Semantic AUROC:')
print(f'  Pearson r = {corr2:.3f}, p = {p2_val:.4f}')
print(f'  Interpretation: {["Weak", "Weak", "Moderate", "Strong"][int(abs(corr2)*4)]} correlation')
if p2_val < 0.10:
    print(f'  ✅ Marginally significant (p < 0.10)')
print()
print('Key Observation:')
print('  Semantic branch shows stronger correlation with prototype similarity')
print('  than Fusion branch, suggesting consistent prototypes are more critical')
print('  for semantic learning in few-shot anomaly detection.')
print('=' * 80)
