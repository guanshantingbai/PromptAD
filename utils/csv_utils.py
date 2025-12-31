import pandas as pd
import os


def write_results(results:dict, cur_class, total_classes, csv_path):
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
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name, dataset, csv_path):
    # if dataset != 'mvtec':
    for indx in range(len(total_classes)):
        total_classes[indx] = f"{dataset}-{total_classes[indx]}"
    class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path)
