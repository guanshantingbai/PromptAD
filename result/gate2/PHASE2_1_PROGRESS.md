# Phase 2.1 Analysis Progress Report

**Date**: 2025-12-22  
**Branch**: gate2-spatial-prompts  
**Status**: Partial - Framework complete, awaiting per-sample data

---

## ✅ Completed

### 1. Analysis Framework Implementation

**File**: `phase2_1_oracle_correlation.py`

- ✅ Task 1: Oracle selection summary (aggregated statistics)
- ✅ Task 4: Hard-case identification
- ⏸️ Task 2: Indicator distribution analysis (needs per-sample data)
- ⏸️ Task 3: Indicator AUC computation (needs per-sample data)

### 2. Key Findings (Aggregated Level)

**Overall Statistics** (27 classes, MVTec + VisA, k=4, CLS):
- Semantic selection ratio: **48.7% ± 29.8%**
- Oracle gain over best branch: **8.58% ± 9.89%**
- Range: 0% - 33.82%

**Hard Cases Identified** (5 classes):

| Dataset | Class | Semantic Ratio | Oracle Gain |
|---------|-------|----------------|-------------|
| MVTec | capsule | 39.4% | +7.94% |
| MVTec | screw | 25.6% | **+33.82%** |
| VisA | capsules | 85.2% | **+29.72%** |
| VisA | macaroni2 | 90.5% | **+26.72%** |
| VisA | pcb2 | 90.6% | **+24.32%** |

**Insights**:
1. Oracle selection is **NOT random** (large variance across classes)
2. Hard cases show **extreme semantic preference** or **extreme memory preference**
3. Oracle gain is **largest on hard cases** (up to 33.82%)
4. This validates the research motivation: adaptive gating has potential

### 3. Deliverables

**CSV Files**:
- ✅ `result/gate2/phase2_1_analysis/oracle_selection_summary.csv`
- ✅ `result/gate2/phase2_1_analysis/hard_cases_summary.csv`

**Visualizations**:
- ✅ `result/gate2/phase2_1_analysis/oracle_selection_histogram.png`

---

## ⏸️ Blocked: Per-Sample Data Needed

### Current Limitation

Gate experiment results (`result/gate/`) only contain **aggregated statistics**:
```json
{
    "oracle": {
        "i_roc": 100.0,
        "oracle_semantic_ratio": 18.07,  // ✅ Have this
        "oracle_memory_ratio": 81.93     // ✅ Have this
    }
}
```

**Missing**: Per-sample data needed for Tasks 2 & 3:
```json
// Need to add:
{
    "oracle": {
        ...
        "per_sample_data": [
            {
                "sample_id": 0,
                "gt_label": 1,
                "semantic_score": 0.85,
                "memory_score": 0.92,
                "oracle_choice": 1,  // 0=semantic, 1=memory
                "reliability_indicators": {
                    "r_mem_margin": 1.23,
                    "r_mem_entropy": -0.45,
                    "r_sem_prompt_var": -0.78,
                    "r_sem_prompt_margin": 0.56,
                    ...
                }
            },
            ...
        ]
    }
}
```

### Why Per-Sample Data is Critical

**Task 2** (Indicator Distribution Analysis):
- Need to split samples by oracle choice
- Plot boxplot/violin for each indicator
- Compute effect size and p-values

**Task 3** (Oracle Predictability):
- Need per-sample indicator values as "prediction score"
- Need per-sample oracle choices as "labels"
- Compute ROC-AUC for each indicator individually

**Without per-sample data**: Can only analyze aggregated statistics (already done).

---

## 🚧 Next Steps

### Option A: Modify Gate Experiment (Recommended)

**Pros**: Clean, complete solution  
**Cons**: Need to re-run experiments (time-consuming)

**Steps**:
1. Modify `run_gate_experiment.py`:
   - Integrate `ReliabilityEstimator` from `PromptAD/reliability.py`
   - Compute indicators for each sample during evaluation
   - Save per-sample data in metadata JSON

2. Re-run gate experiments:
   ```bash
   python run_gate_experiment.py \
       --dataset mvtec --k_shot 4 --task cls \
       --save_per_sample_data  # New flag
   ```

3. Update `phase2_1_oracle_correlation.py`:
   - Load per-sample data from enhanced metadata
   - Complete Tasks 2 & 3

**Estimated time**: 1-2 days for full re-run

### Option B: Prototype with Subset (Quick Validation)

**Pros**: Fast, validates approach before full re-run  
**Cons**: Incomplete, only for proof-of-concept

**Steps**:
1. Pick 3-5 representative classes
2. Run inference to get per-sample scores and features
3. Compute indicators using `ReliabilityEstimator`
4. Manually create per-sample JSON
5. Test Phase 2.1 Tasks 2 & 3 on subset

**Estimated time**: 4-6 hours

### Option C: Approximate Analysis (Current Limitation)

**Pros**: Uses existing data  
**Cons**: Cannot validate core hypothesis

**Steps**:
1. Use class-level statistics as proxy
2. Treat each class as one "sample"
3. Compute correlation between oracle ratio and branch AUROC difference

**Limitation**: Only 27 data points (classes), not robust

---

## 📊 Expected Results (After Per-Sample Data)

### Task 2: Indicator Distribution Analysis

**Example for `r_mem_margin`**:
```
Group A (oracle→semantic): mean=0.85, median=0.92
Group B (oracle→memory):   mean=1.23, median=1.18
Effect size: -0.38 (memory more reliable when oracle chooses memory)
p-value: 0.001 (statistically significant)
```

**Hypothesis**: 
- When oracle chooses memory → `r_mem_margin` higher (memory more reliable)
- When oracle chooses semantic → `r_sem_prompt_var` higher (semantic more reliable)

### Task 3: Indicator AUC

**Example results**:

| Indicator | AUC (Full) | AUC (Hard Cases) | Interpretation |
|-----------|------------|------------------|----------------|
| `r_mem_margin` | 0.68 | 0.72 | Moderate predictive power |
| `r_mem_entropy` | 0.62 | 0.65 | Weak but meaningful |
| `r_sem_prompt_var` | 0.71 | 0.75 | **Best predictor** |
| `r_sem_prompt_margin` | 0.66 | 0.69 | Moderate predictive power |

**Success criteria**: Any AUC > 0.60 validates that oracle is partially predictable.

**Failure criteria**: All AUC ~0.50 (random) → indicators don't capture reliability.

---

## 🎯 Research Implications

### If AUC ~0.6-0.7 (Expected)

**Interpretation**: 
- Oracle decisions are **partially** explainable by reliability indicators
- Indicators capture **some but not all** factors affecting branch performance
- This is **sufficient** to motivate adaptive gating (Phase 2.2)

**Next phase**: 
- Build simple adaptive gating using top-performing indicators
- Evaluate if adaptive > max fusion (even if adaptive < oracle)

### If AUC ~0.5 (Unexpected Failure)

**Interpretation**:
- Current indicators don't capture oracle decision factors
- May need different indicators or features

**Pivot options**:
1. Add more indicators (patch-level variance, spatial consistency, etc.)
2. Use different normalization (percentile instead of z-score)
3. Consider meta-learning approach instead of hand-crafted indicators

---

## 📁 File Organization

```
result/gate2/
├── phase2_1_analysis/
│   ├── oracle_selection_summary.csv          # ✅ Done
│   ├── hard_cases_summary.csv                # ✅ Done
│   ├── oracle_selection_histogram.png        # ✅ Done
│   │
│   ├── indicator_auc_summary.csv             # ⏸️ Needs per-sample data
│   ├── indicator_r_mem_margin_distribution.png   # ⏸️ Needs data
│   ├── indicator_r_sem_prompt_var_distribution.png  # ⏸️ Needs data
│   └── ...
│
└── PHASE2_1_PROGRESS.md                      # This file
```

---

## 💡 Recommendations

**Immediate Action** (Choose one):

1. **If research timeline is tight** → Option B (Prototype with subset)
   - Quick validation (4-6 hours)
   - Decide go/no-go before full implementation
   
2. **If confident in approach** → Option A (Full implementation)
   - Most robust solution
   - Enables full analysis on 162 tasks
   
3. **If uncertain** → Option C (Class-level approximation)
   - Quick sanity check with existing data
   - If promising → proceed to Option A

**My recommendation**: **Option B** first, then Option A if results are promising.

---

**Status**: Awaiting decision on next steps.  
**Blocker**: Need per-sample oracle choices and reliability indicators.  
**Estimated completion**: 1-2 days after data availability.
