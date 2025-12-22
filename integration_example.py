"""
Example: How to integrate modular scoring into existing train_cls.py

This shows minimal changes needed to support --score-mode flag.
"""

# ============= BEFORE (Original train_cls.py) =============
"""
def main(args):
    # ... setup code ...
    
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    # ... training loop ...
    
    metrics = fit(model, args, test_dataloader, device, ...)
"""

# ============= AFTER (With modular scoring support) =============
"""
def main(args):
    # ... setup code ...
    
    # Create base model (same as before)
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    # ADDED: Optional modular scoring wrapper
    if hasattr(args, 'score_mode') and args.score_mode != 'default':
        from PromptAD.model_modular import PromptADModular
        model = PromptADModular(model, score_mode=args.score_mode)
        print(f"Using modular scorer: {args.score_mode}")
    
    # ... rest unchanged ...
    
    metrics = fit(model, args, test_dataloader, device, ...)
"""

# ============= ARGUMENT PARSER UPDATE =============
"""
def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    
    # ... existing args ...
    
    # ADDED: Score mode selection
    parser.add_argument('--score-mode', type=str, default='default',
                        choices=['default', 'semantic', 'memory', 'max', 'harmonic', 'oracle'],
                        help='Scoring strategy: default uses current model.forward(), '
                             'others use modular scorer')
    
    args = parser.parse_args()
    return args
"""

# ============= USAGE EXAMPLES =============
"""
# Default behavior (unchanged)
python train_cls.py --class_name carpet

# Semantic-only evaluation
python train_cls.py --class_name carpet --score-mode semantic

# Memory-only evaluation  
python train_cls.py --class_name carpet --score-mode memory

# Max fusion (should match current default)
python train_cls.py --class_name carpet --score-mode max

# Oracle upper bound
python train_cls.py --class_name carpet --score-mode oracle
"""

# ============= ALTERNATIVE: Separate Script =============
"""
Instead of modifying train_cls.py, you can also just use demo_modular_scoring.py:

python demo_modular_scoring.py \\
    --dataset mvtec \\
    --class_name carpet \\
    --k-shot 4 \\
    --task cls

This evaluates all modes (semantic, memory, max, harmonic, oracle) in one run.
"""

print("Integration example created!")
print("\nThree ways to use modular scoring:")
print("1. Minimal integration: Add --score-mode to train_cls.py (shown above)")
print("2. Standalone script: Use demo_modular_scoring.py (already created)")
print("3. Direct API: Import PromptADModular in your own scripts")
