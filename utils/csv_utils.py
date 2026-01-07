import pandas as pd
import os


def write_results(results:dict, cur_class, total_classes, csv_path, alpha=None):
    keys_ = list(results.keys())
    
    # Determine which keys to use based on what's in results
    # For CLS task: i_roc, semantic_i_roc, memory_i_roc (no p_roc)
    # For SEG task: only p_roc
    if 'p_roc' in results and 'i_roc' not in results:
        # Segmentation task: only pixel-level metric
        keys = ['p_roc']
    elif 'p_roc' in results and 'i_roc' in results:
        # Both image and pixel-level metrics
        keys = ['i_roc', 'p_roc', 'semantic_i_roc', 'memory_i_roc']
    else:
        # Classification task: only image-level metrics
        keys = ['i_roc', 'semantic_i_roc', 'memory_i_roc']

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys_:
        # For fusion_i_roc, create alpha-specific column if alpha is provided
        if k == 'fusion_i_roc' and alpha is not None:
            col_name = f'fusion_alpha_{alpha:.2f}'
            df.loc[cur_class, col_name] = results[k]
        else:
            df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def write_semantic_sweep(semantic_roc, cur_class, total_classes, csv_path, alpha):
    """
    Write semantic_i_roc for different alpha values to a separate CSV.
    
    Args:
        semantic_roc: float, the semantic_i_roc value
        cur_class: str, current class name (e.g., 'mvtec-carpet')
        total_classes: list, all class names
        csv_path: str, path to semantic sweep CSV
        alpha: float, semantic weight value
    """
    # Create CSV if not exists
    if not os.path.exists(csv_path):
        df_all = pd.DataFrame(0.0, index=total_classes, columns=[])
        df_all.to_csv(csv_path, header=True, float_format='%.2f')
    
    # Load existing CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Add column for this alpha if not exists
    col_name = f'semantic_alpha_{alpha:.2f}'
    if col_name not in df.columns:
        df[col_name] = 0.0
    
    # Update value
    df.loc[cur_class, col_name] = semantic_roc
    
    # Save
    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name, dataset, csv_path, semantic_weight=None):
    # if dataset != 'mvtec':
    for indx in range(len(total_classes)):
        total_classes[indx] = f"{dataset}-{total_classes[indx]}"
    class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path, alpha=semantic_weight)
    
    # If semantic_weight is provided, also write to semantic sweep CSV
    if semantic_weight is not None and 'semantic_i_roc' in metrics:
        # Create semantic sweep CSV path
        csv_dir = os.path.dirname(csv_path)
        csv_base = os.path.basename(csv_path).replace('-results.csv', '-semantic-sweep.csv')
        semantic_csv_path = os.path.join(csv_dir, csv_base)
        
        # Write semantic score
        write_semantic_sweep(
            semantic_roc=metrics['semantic_i_roc'],
            cur_class=class_name,
            total_classes=total_classes,
            csv_path=semantic_csv_path,
            alpha=semantic_weight
        )
