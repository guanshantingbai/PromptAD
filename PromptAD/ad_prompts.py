

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut"
}


state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

#
class_state_abnormal = {
    'bottle': ['{} with large breakage', '{} with small breakage', '{} with contamination'],
    'toothbrush': ['{} with defect', '{} with anomaly'],
    'carpet': ['{} with hole', '{} with color stain', '{} with metal contamination', '{} with thread residue', '{} with thread', '{} with cut'],
    'hazelnut': ['{} with crack', '{} with cut', '{} with hole', '{} with print'],
    'leather': ['{} with color stain', '{} with cut', '{} with fold', '{} with glue', '{} with poke'],
    'cable': ['{} with bent wire', '{} with missing part', '{} with missing wire', '{} with cut', '{} with poke'],
    'capsule': ['{} with crack', '{} with faulty imprint', '{} with poke', '{} with scratch', '{} squeezed with compression'],
    'grid': ['{} with breakage',  '{} with thread residue', '{} with thread', '{} with metal contamination', '{} with glue', '{} with a bent shape'],
    'pill': ['{} with color stain', '{} with contamination', '{} with crack', '{} with faulty imprint', '{} with scratch', '{} with abnormal type'],
    'transistor': ['{} with bent lead', '{} with cut lead', '{} with damage', '{} with misplaced transistor'],
    'metal_nut': ['{} with a bent shape ', '{} with color stain', '{} with a flipped orientation', '{} with scratch'],
    'screw': ['{} with manipulated front',  '{} with scratch neck', '{} with scratch head'],
    'zipper': ['{} with broken teeth', '{} with fabric border', '{} with defect fabric', '{} with broken fabric', '{} with split teeth', '{} with squeezed teeth'],
    'tile': ['{} with crack', '{} with glue strip', '{} with gray stroke', '{} with oil', '{} with rough surface'],
    'wood': ['{} with color stain', '{} with hole', '{} with scratch', '{} with liquid'],

    'candle': ['{} with melded wax', '{} with foreign particals', '{} with extra wax', '{} with chunk of wax missing', '{} with weird candle wick', '{} with damaged corner of packaging', '{} with different colour spot'],
    'capsules': ['{} with scratch', '{} with discolor', '{} with misshape', '{} with leak', '{} with bubble'],
    # 'capsules': [],
    'cashew': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with spot'],
    'chewinggum': ['{} with corner missing', '{} with scratches', '{} with chunk of gum missing', '{} with colour spot', '{} with cracks'],
    'fryum': ['{} with breakage', '{} with scratches', '{} with burnt', '{} with colour spot', '{} with fryum stuck together', '{} with colour spot'],
    'macaroni1': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'macaroni2': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'pcb1': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb2': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb3': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb4': ['{} with scratch', '{} with extra', '{} with missing', '{} with wrong place', '{} with damage', '{} with burnt', '{} with dirt'],
    'pipe_fryum': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with colour spot', '{} with cracks']}


def get_full_manual_prompts(classname):
    """
    获取指定类别的完整静态prompt列表
    
    Args:
        classname: 类别名称
        
    Returns:
        full_prompts: 完整的prompt模板列表 (通用 + 类别特定)
        prompt_info: 每个prompt的详细信息
    """
    # 映射类别名称
    display_name = class_mapping.get(classname, classname)
    
    # 通用异常状态
    generic_prompts = state_anomaly.copy()
    
    # 类别特定异常状态
    specific_prompts = class_state_abnormal.get(classname, [])
    
    # 合并
    full_prompts = generic_prompts + specific_prompts
    
    # 构建详细信息
    prompt_info = []
    for i, prompt_template in enumerate(full_prompts):
        info = {
            'index': i,
            'template': prompt_template,
            'text': prompt_template.format(display_name),
            'type': 'generic' if i < len(generic_prompts) else 'specific',
            'classname': classname,
            'display_name': display_name
        }
        prompt_info.append(info)
    
    return full_prompts, prompt_info


def load_prompts_from_table(classname, table_path='prompts/manual_prompts_master_table.csv'):
    """
    从主Prompt表格加载指定类别的prompts
    直接返回full_text列（已填充类别名），不需要后续format
    
    Args:
        classname: 类别名称
        table_path: Prompt表格路径
        
    Returns:
        full_texts: 完整的prompt文本列表（只包含enabled=True的）
        prompt_info: 详细信息列表
    """
    import pandas as pd
    import os
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_path = os.path.join(project_root, table_path)
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        print(f"Warning: Prompt table not found at {full_path}")
        print(f"Falling back to hard-coded prompts from ad_prompts.py")
        return get_full_manual_prompts(classname)
    
    # 读取CSV
    df = pd.read_csv(full_path)
    
    # 过滤该类别且enabled=True的prompts
    class_prompts = df[(df['class'] == classname) & (df['enabled'] == True)]
    
    if len(class_prompts) == 0:
        print(f"Warning: No enabled prompts found for class '{classname}' in table")
        print(f"Falling back to hard-coded prompts")
        return get_full_manual_prompts(classname)
    
    # 按index_in_class排序
    class_prompts = class_prompts.sort_values('index_in_class')
    
    # 直接使用full_text列（已填充类别名的完整文本）
    if 'full_text' in class_prompts.columns:
        full_texts = class_prompts['full_text'].tolist()
    else:
        # 如果没有full_text列，从template生成（向后兼容）
        templates = class_prompts['template'].tolist()
        display_name = class_prompts['display_name'].iloc[0]
        full_texts = [t.format(display_name) for t in templates]
    
    prompt_info = []
    for _, row in class_prompts.iterrows():
        info = {
            'prompt_id': row['prompt_id'],
            'index': row['index_in_class'],
            'template': row['template'],
            'text': row['full_text'] if 'full_text' in row else row['template'],
            'type': row['type'],
            'classname': classname,
            'display_name': row['display_name'],
            'enabled': row['enabled'],
            'manual_score': row['manual_score'] if pd.notna(row['manual_score']) else None,
            'relevance': row['relevance'] if pd.notna(row['relevance']) else None,
            'notes': row['notes'] if pd.notna(row['notes']) else None,
        }
        prompt_info.append(info)
    
    print(f"✓ Loaded {len(full_texts)} prompts for '{classname}' from table (using full_text)")
    
    return full_texts, prompt_info


def get_all_classes_manual_prompts():
    """
    获取所有类别的完整静态prompt表
    
    Returns:
        dict: {classname: {'prompts': [...], 'info': [...]}}
    """
    all_classes = list(class_state_abnormal.keys())
    result = {}
    
    for classname in all_classes:
        prompts, info = get_full_manual_prompts(classname)
        result[classname] = {
            'prompts': prompts,
            'info': info,
            'num_generic': len(state_anomaly),
            'num_specific': len(class_state_abnormal.get(classname, [])),
            'num_total': len(prompts)
        }
    
    return result


def print_manual_prompts_table(classname=None):
    """
    打印静态prompt表格，便于查看和分析
    
    Args:
        classname: 指定类别名称，None则打印所有类别
    """
    if classname is not None:
        prompts, info = get_full_manual_prompts(classname)
        print(f"\n{'='*80}")
        print(f"Class: {classname} (Display: {info[0]['display_name']})")
        print(f"Total: {len(prompts)} prompts (Generic: {len(state_anomaly)}, Specific: {len(prompts) - len(state_anomaly)})")
        print(f"{'='*80}")
        
        for item in info:
            type_tag = '[G]' if item['type'] == 'generic' else '[S]'
            print(f"{item['index']:3d} {type_tag} {item['template']:40s} -> {item['text']}")
    else:
        all_prompts = get_all_classes_manual_prompts()
        print(f"\n{'='*80}")
        print(f"All Classes Manual Prompts Summary")
        print(f"{'='*80}")
        print(f"{'Class':<20} {'Generic':<10} {'Specific':<10} {'Total':<10}")
        print(f"{'-'*80}")
        
        for classname, data in all_prompts.items():
            print(f"{classname:<20} {data['num_generic']:<10} {data['num_specific']:<10} {data['num_total']:<10}")
