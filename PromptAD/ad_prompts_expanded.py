"""
Expanded static prompts for multi-prototype learning.
Each class has a list of fully formatted prompts (no .format() needed).
"""

# Class name mapping (from ad_prompts.py)
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

# Generic LAP (Learnable Anomaly Prompts) - from Purge1
generic_lap_prompts = [
    "damaged {}",
    "{} with damage"
]

# Class-specific MAP (Most Anomalous Prompts) - from Purge3
# Fully expanded: no need for .format(classname)
class_specific_map_prompts = {
    'bottle': [
        'bottle with large breakage',
        'bottle with small breakage',
        'bottle with contamination'
    ],
    
    'toothbrush': [
        'toothbrush with defect',
        'toothbrush with anomaly'
    ],
    
    'carpet': [
        'carpet with hole',
        'carpet with color stain',
        'carpet with metal contamination',
        'carpet with thread residue',
        'carpet with thread',
        'carpet with cut'
    ],
    
    'hazelnut': [
        'hazelnut with crack',
        'hazelnut with cut',
        'hazelnut with hole',
        'hazelnut with print'
    ],
    
    'leather': [
        'leather with color stain',
        'leather with cut',
        'leather with fold',
        'leather with glue',
        'leather with poke'
    ],
    
    'cable': [
        'cable with bent wire',
        'cable with cut',
        'cable with poke'
    ],
    
    'capsule': [
        'capsule with crack',
        'capsule with faulty imprint',
        'capsule with poke',
        'capsule with scratch',
        'capsule squeezed with compression'
    ],
    
    'grid': [
        'grid with breakage',
        'grid with thread residue',
        'grid with thread',
        'grid with metal contamination',
        'grid with glue',
        'grid with a bent shape'
    ],
    
    'pill': [
        'pill with color stain',
        'pill with contamination',
        'pill with crack',
        'pill with faulty imprint',
        'pill with abnormal type'
    ],
    
    'transistor': [
        'transistor with bent lead',
        'transistor with cut lead'
    ],
    
    'metal_nut': [
        'metal nut with color stain',
        'metal nut with scratch'
    ],
    
    'screw': [
        'screw with manipulated front',
        'screw with scratch neck',
        'screw with scratch head'
    ],
    
    'zipper': [
        'zipper with broken teeth',
        'zipper with fabric border',
        'zipper with defect fabric',
        'zipper with broken fabric',
        'zipper with split teeth',
        'zipper with squeezed teeth'
    ],
    
    'tile': [
        'tile with crack',
        'tile with glue strip',
        'tile with gray stroke',
        'tile with oil',
        'tile with rough surface'
    ],
    
    'wood': [
        'wood with color stain',
        'wood with hole',
        'wood with scratch',
        'wood with liquid'
    ],
    
    # VisA dataset
    'candle': [
        'candle with melded wax',
        'candle with foreign particals',
        'candle with extra wax',
        'candle with chunk of wax missing',
        'candle with weird candle wick',
        'candle with damaged corner of packaging',
        'candle with different colour spot'
    ],
    
    'capsules': [
        'capsules with scratch',
        'capsules with discolor',
        'capsules with misshape',
        'capsules with leak',
        'capsules with bubble'
    ],
    
    'cashew': [
        'cashew with breakage',
        'cashew with small scratches',
        'cashew with burnt',
        'cashew with stuck together',
        'cashew with spot'
    ],
    
    'chewinggum': [
        'chewing gum with corner missing',
        'chewing gum with scratches',
        'chewing gum with chunk of gum missing',
        'chewing gum with colour spot',
        'chewing gum with cracks'
    ],
    
    'fryum': [
        'fryum with breakage',
        'fryum with scratches',
        'fryum with burnt',
        'fryum with colour spot',
        'fryum with fryum stuck together',
        'fryum with colour spot'
    ],
    
    'macaroni1': [
        'macaroni with color spot',
        'macaroni with small chip around edge',
        'macaroni with small scratches',
        'macaroni with breakage',
        'macaroni with cracks'
    ],
    
    'macaroni2': [
        'macaroni with color spot',
        'macaroni with small chip around edge',
        'macaroni with small scratches',
        'macaroni with breakage',
        'macaroni with cracks'
    ],
    
    'pcb1': [
        'printed circuit board with bent',
        'printed circuit board with scratch',
        'printed circuit board with missing',
        'printed circuit board with melt'
    ],
    
    'pcb2': [
        'printed circuit board with bent',
        'printed circuit board with scratch',
        'printed circuit board with missing',
        'printed circuit board with melt'
    ],
    
    'pcb3': [
        'printed circuit board with bent',
        'printed circuit board with scratch',
        'printed circuit board with missing',
        'printed circuit board with melt'
    ],
    
    'pcb4': [
        'printed circuit board with scratch',
        'printed circuit board with extra',
        'printed circuit board with missing',
        'printed circuit board with wrong place',
        'printed circuit board with damage',
        'printed circuit board with burnt',
        'printed circuit board with dirt'
    ],
    
    'pipe_fryum': [
        'pipe fryum with breakage',
        'pipe fryum with small scratches',
        'pipe fryum with burnt',
        'pipe fryum with stuck together',
        'pipe fryum with colour spot',
        'pipe fryum with cracks'
    ]
}


def get_all_prompts_for_class(classname, use_lap=True):
    """
    Get all prompts for a specific class.
    
    Args:
        classname: The class name (e.g., 'bottle', 'metal_nut')
        use_lap: Whether to include generic LAP prompts
    
    Returns:
        list of strings: All prompts for this class
    """
    # Apply class name mapping first
    display_name = class_mapping.get(classname, classname)
    
    prompts = []
    
    # Add LAP prompts (if enabled)
    if use_lap:
        lap_prompts = [template.format(display_name) for template in generic_lap_prompts]
        prompts.extend(lap_prompts)
    
    # Add MAP prompts (class-specific)
    if classname in class_specific_map_prompts:
        prompts.extend(class_specific_map_prompts[classname])
    
    return prompts


def print_prompt_table(classname=None):
    """Print prompt table for visualization."""
    if classname:
        classes = [classname]
    else:
        classes = sorted(class_specific_map_prompts.keys())
    
    print("\n" + "="*80)
    print("EXPANDED PROMPT TABLE (Multi-Prototype Ready)")
    print("="*80)
    
    for cls in classes:
        display_name = class_mapping.get(cls, cls)
        
        print(f"\n[{cls.upper()}] (display: '{display_name}')")
        print(f"  LAP (Generic): {len(generic_lap_prompts)} prompts")
        for i, template in enumerate(generic_lap_prompts, 1):
            print(f"    {i}. {template.format(display_name)}")
        
        if cls in class_specific_map_prompts:
            map_prompts = class_specific_map_prompts[cls]
            print(f"  MAP (Specific): {len(map_prompts)} prompts")
            for i, prompt in enumerate(map_prompts, 1):
                print(f"    {i}. {prompt}")
        
        print(f"  Total: {len(get_all_prompts_for_class(cls, use_lap=True))} prompts")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test: print all prompts
    print_prompt_table()
    
    # Test: get prompts for specific class
    print("\n\nTest: Getting prompts for 'metal_nut'")
    prompts = get_all_prompts_for_class('metal_nut', use_lap=True)
    for i, p in enumerate(prompts, 1):
        print(f"{i}. {p}")
