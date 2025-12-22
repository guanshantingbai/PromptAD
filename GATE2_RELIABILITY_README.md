# Gate2: Reliability-Based Branch Analysis

**Branch**: `gate2-spatial-prompts`  
**Status**: Phase 1 - Analysis Mode  
**Goal**: Validate reliability indicators can predict oracle choices (WITHOUT building gating yet)

---

## 🎯 Design Philosophy

### Phase 1: Analysis Mode (Current)

**What we DO:**
- ✅ Compute reliability indicators from prior information
- ✅ Output separate indicators: `r_mem_margin`, `r_mem_entropy`, `r_sem_prompt_var`, etc.
- ✅ Analyze correlation with oracle selection patterns
- ✅ Validate hypothesis: "Can prior-based indicators predict which branch is more reliable?"

**What we DON'T do:**
- ❌ Aggregate indicators into single reliability score
- ❌ Build adaptive gating mechanism
- ❌ Evaluate as a "method" with AUROC
- ❌ Introduce learnable parameters or dataset-specific tuning

**Purpose**: Avoid premature optimization. First prove the indicators are informative.

---

## 📂 File Structure

### Core Implementation

```
PromptAD/
├── reliability.py              # Reliability indicator computation
│   ├── ReliabilityEstimator
│   │   ├── calibrate_on_support()       # Pre-compute support stats
│   │   ├── compute_memory_reliability()  # Memory indicators
│   │   ├── compute_semantic_reliability()# Semantic indicators
│   │   └── compare_branch_reliability()  # Output grouped indicators
│   └── [NO adaptive_weights in Phase 1]
```

### Analysis Scripts

```
analyze_oracle_vs_reliability.py  # Phase 1 oracle pattern analysis
├── load_gate_results()           # Load gate experiment outputs
├── analyze_oracle_pattern()      # Statistics on oracle choices
└── plot_oracle_analysis()        # Visualization
```

---

## 🔬 Reliability Indicators

### Memory Branch (`r_mem_*`)

| Indicator | Definition | Interpretation |
|-----------|------------|----------------|
| `r_mem_margin` | NN margin (d2 - d1), z-scored | Higher = clearer nearest neighbor = more reliable |
| `r_mem_entropy` | Neighborhood entropy, z-scored, negated | Higher = lower entropy = neighbors agree = more reliable |
| `r_mem_centroid` | Similarity to support centroid, z-scored | Higher = closer to normal = more reliable for normal samples |

**Normalization**: All z-scored using support set leave-one-out statistics (median + MAD).

### Semantic Branch (`r_sem_*`)

| Indicator | Definition | Interpretation |
|-----------|------------|----------------|
| `r_sem_prompt_var` | Prompt variance, z-scored, negated | Higher = lower variance = prompts agree = more reliable |
| `r_sem_prompt_margin` | Margin between top-2 prompts, z-scored | Higher = stronger semantic preference = more reliable |
| `r_sem_extremity` | Distance from 0.5, z-scored | Higher = more extreme score = more confident |

**Normalization**: All z-scored using support set statistics (median + MAD).

---

## 🚀 Phase 1 Workflow

### Step 1: Analyze Oracle Selection Patterns

**Goal**: Understand when oracle prefers semantic vs memory.

```bash
cd /home/zju/codes/AD/PromptAD

# Analyze MVTec AD k=4 CLS
python analyze_oracle_vs_reliability.py \
    --dataset mvtec \
    --k_shot 4 \
    --task cls \
    --output_dir result/gate2/analysis

# Output:
# - result/gate2/analysis/oracle_analysis_mvtec_k4_cls.json
# - result/gate2/analysis/oracle_selection_mvtec_k4_cls.png
```

**Expected insights:**
1. Is oracle selection balanced (50/50) or biased?
2. Does selection pattern vary significantly across classes?
3. Is score difference (semantic - memory) a good predictor?

### Step 2: Integrate Reliability Indicators (TODO)

**Prerequisites**: Modify `PromptAD/model.py` to expose per-prompt scores.

```python
# Example usage (not yet integrated)
from PromptAD.reliability import ReliabilityEstimator

estimator = ReliabilityEstimator(temperature=0.07)

# Calibrate on support set
estimator.calibrate_on_support(
    support_features=...,
    gallery=...,
    prompt_scores_support=...
)

# Compute indicators for test samples
memory_rel = estimator.compute_memory_reliability(query_features, gallery)
semantic_rel = estimator.compute_semantic_reliability(prompt_scores)

# Get grouped indicators (NO aggregation)
indicators = estimator.compare_branch_reliability(memory_rel, semantic_rel)
# Returns: {r_mem_margin, r_mem_entropy, ..., r_sem_prompt_var, ...}
```

### Step 3: Correlation Analysis (TODO)

Extend `analyze_oracle_vs_reliability.py` to:
- Compute reliability indicators for all test samples
- Calculate correlation with oracle choices
- Visualize: `indicator_value` vs `oracle_selection`

**Success criteria:**
- Strong correlation (|r| > 0.5) → proceed to Phase 2
- Weak correlation (|r| < 0.3) → rethink indicator design

---

## 📊 Expected Outputs

### Phase 1 Deliverables

1. **Oracle Selection Report**
   - JSON: detailed statistics per class
   - Plot: bar chart of semantic vs memory selection ratios
   - Insight: is oracle deterministic or random?

2. **Reliability Indicator Dataset** (TODO)
   - Per-sample indicators for all 162 tasks
   - Format: `{class, k_shot, task, r_mem_*, r_sem_*, oracle_choice}`

3. **Correlation Analysis** (TODO)
   - Pearson/Spearman correlation matrices
   - Visualization: scatter plots, heatmaps
   - Decision: proceed to Phase 2 or pivot?

---

## ⚠️ Important Constraints

### What Phase 1 Must NOT Do

1. **No score fusion**: Don't combine semantic + memory with adaptive weights
2. **No AUROC evaluation**: Phase 1 is not a "method", just analysis
3. **No aggregation**: Keep indicators separate for interpretability
4. **No tuning**: All normalization from support set only

### Rationale

- Prevent bias: Avoid treating Phase 1 as "proposed method"
- Scientific rigor: Validate hypothesis before building mechanism
- Reviewer clarity: Clearly separate analysis from contribution

---

## 🔄 Transition to Phase 2

**Only proceed if Phase 1 shows:**
- ✅ Strong correlation between indicators and oracle choices
- ✅ Indicators are class-agnostic (generalize across MVTec/VisA)
- ✅ Clear ranking: some indicators consistently outperform others

**Phase 2 would add:**
- Weighted combination of indicators
- Adaptive gating: `score = w_sem * semantic + w_mem * memory`
- Full evaluation on 162 tasks
- Comparison: Adaptive vs Max vs Oracle

---

## 📝 Current Status

- [x] Implement `ReliabilityEstimator` class
- [x] Implement memory reliability indicators
- [x] Implement semantic reliability indicators
- [x] Implement support-set normalization
- [x] Create oracle analysis script
- [ ] **Next: Run Phase 1 experiments**
  - [ ] Analyze oracle patterns (all 162 tasks)
  - [ ] Integrate reliability computation into model
  - [ ] Compute indicators + oracle correlation
  - [ ] Make go/no-go decision for Phase 2

---

## 🎓 Research Question

**Central hypothesis**: Reliability indicators computed from prior information (few-shot, prompt ensemble, memory bank) can predict which branch Oracle would select, WITHOUT using ground-truth labels or dataset-specific tuning.

**If true**: We can build a principled adaptive gating mechanism.  
**If false**: Need better indicators or different approach (e.g., meta-learning).

---

**Last Updated**: 2025-12-22  
**Branch**: gate2-spatial-prompts  
**Contact**: See commit history
