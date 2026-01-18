#!/usr/bin/env python3
"""
Generate margin ablation tables for MVTec and ViSA datasets
"""

import pandas as pd
from pathlib import Path

# Margin values
MARGINS = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2]
base_path = Path('result/margin_ablation')
output_dir = Path('analysis/ablation/margin')

def generate_dataset_table(dataset_name):
    """Generate margin ablation table for a specific dataset"""
    
    # Collect data for all margins
    all_data = {}
    
    for margin in MARGINS:
        csv_path = base_path / f'margin_{margin}' / dataset_name / 'k_2' / 'csv' / 'Seed_111-results.csv'
        
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            
            for idx, row in df.iterrows():
                # Extract class name from index (e.g., 'mvtec-carpet' -> 'carpet')
                class_name = str(idx).split('-')[-1] if '-' in str(idx) else str(idx)
                
                if class_name not in all_data:
                    all_data[class_name] = {}
                
                # Store semantic and fusion values
                all_data[class_name][f'margin_{margin}_semantic'] = row['semantic_i_roc']
                all_data[class_name][f'margin_{margin}_fusion'] = row['i_roc']
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(all_data, orient='index')
    result_df.index.name = 'class_name'
    
    # Sort columns: group by margin, then semantic/fusion
    cols_sorted = []
    for margin in MARGINS:
        cols_sorted.append(f'margin_{margin}_semantic')
        cols_sorted.append(f'margin_{margin}_fusion')
    
    result_df = result_df[cols_sorted]
    
    # Add average row
    avg_row = result_df.mean()
    avg_row.name = 'AVERAGE'
    result_df = pd.concat([result_df, avg_row.to_frame().T])
    
    return result_df

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating margin ablation tables...")
    print()
    
    # Generate MVTec table
    print("Processing MVTec dataset...")
    mvtec_df = generate_dataset_table('mvtec')
    mvtec_output = output_dir / 'mvtec_margin_ablation.csv'
    mvtec_df.to_csv(mvtec_output)
    print(f"  Saved: {mvtec_output}")
    print(f"  Classes: {len(mvtec_df) - 1}")  # -1 for AVERAGE row
    print()
    
    # Generate ViSA table
    print("Processing ViSA dataset...")
    visa_df = generate_dataset_table('visa')
    visa_output = output_dir / 'visa_margin_ablation.csv'
    visa_df.to_csv(visa_output)
    print(f"  Saved: {visa_output}")
    print(f"  Classes: {len(visa_df) - 1}")  # -1 for AVERAGE row
    print()
    
    # Print preview
    print("=" * 100)
    print("MVTec Dataset Preview (first 3 classes):")
    print("=" * 100)
    print(mvtec_df.head(3).to_string())
    print()
    
    print("=" * 100)
    print("ViSA Dataset Preview (first 3 classes):")
    print("=" * 100)
    print(visa_df.head(3).to_string())
    print()
    
    # Print average comparison
    print("=" * 100)
    print("Average Performance Comparison:")
    print("=" * 100)
    print()
    
    print("MVTec AVERAGE:")
    mvtec_avg = mvtec_df.loc['AVERAGE']
    for margin in MARGINS:
        semantic = mvtec_avg[f'margin_{margin}_semantic']
        fusion = mvtec_avg[f'margin_{margin}_fusion']
        print(f"  margin={margin:<4}  Semantic: {semantic:>6.2f}%  Fusion: {fusion:>6.2f}%")
    print()
    
    print("ViSA AVERAGE:")
    visa_avg = visa_df.loc['AVERAGE']
    for margin in MARGINS:
        semantic = visa_avg[f'margin_{margin}_semantic']
        fusion = visa_avg[f'margin_{margin}_fusion']
        print(f"  margin={margin:<4}  Semantic: {semantic:>6.2f}%  Fusion: {fusion:>6.2f}%")
    print()
    
    print("✓ Tables generated successfully!")

if __name__ == "__main__":
    main()
