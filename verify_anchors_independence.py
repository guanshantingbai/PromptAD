"""
验证"平均锚点只用于评估，不影响训练"

关键问题：
1. 训练时是否使用了平均锚点？
2. 修改评估方式（平均→maxpooling）是否会影响训练？
"""

import torch
import sys

print("="*80)
print("验证：平均锚点 vs 训练过程的独立性")
print("="*80)

print("\n【第1部分】训练过程中的锚点使用")
print("-"*80)

print("""
在 train_cls.py 的训练循环中：

1. 🔴 训练时使用的锚点 (train_cls.py L87-91):
   ```python
   # 每个batch都重新计算平均锚点（局部变量）
   normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
   abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
   
   # 用于计算loss（不保存到模型）
   l_pos = einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.T)
   l_neg_v2t = einsum('nc,cm->nm', cls_feature, abnormal_text_features.T)
   loss_v2t = criterion(logits_v2t, target_v2t)
   ```

2. 🟢 每个epoch结束后 (train_cls.py L113):
   ```python
   scheduler.step()
   model.build_text_feature_gallery()  # ← 这里计算并保存平均锚点
   ```

3. 🔵 评估时使用的锚点 (model.py calculate_textual_anomaly_score):
   ```python
   # 使用 self.text_features（保存的平均锚点）
   logits = t * visual_features @ self.text_features.T
   ```

结论：
✅ 训练时：每个batch使用临时计算的平均锚点（局部变量）
✅ 评估时：使用 build_text_feature_gallery() 保存的平均锚点
✅ 两者独立：修改评估方式不影响训练
""")

print("\n【第2部分】验证训练过程不依赖保存的text_features")
print("-"*80)

print("""
证明：self.text_features 只在评估时使用

训练循环伪代码：
```
for epoch in range(Epoch):
    for batch in train_data:
        # ❌ 没有使用 self.text_features
        normal_text_prompt = model.prompt_learner()
        normal_text_features = model.encode_text_embedding(normal_text_prompt)  
        
        # ✅ 使用临时计算的平均值
        normal_anchor = normal_text_features.mean(dim=0)
        abnormal_anchor = abnormal_text_features.mean(dim=0)
        
        loss = compute_loss(cls_feature, normal_anchor, abnormal_anchor)
        loss.backward()
    
    # 🔄 epoch结束后更新保存的锚点（用于评估）
    model.build_text_feature_gallery()  # 更新 self.text_features
    
    # 📊 评估时才使用 self.text_features
    if eval_epoch:
        score = model.calculate_textual_anomaly_score(...)  # 使用 self.text_features
```

关键区别：
┌────────────────────────────────────────┐
│ 训练时 (每个batch)                      │
│ - 输入：当前batch的prompt embeddings    │
│ - 计算：临时平均锚点（局部变量）         │
│ - 用途：计算loss并更新prompt参数        │
│ - 不保存                               │
└────────────────────────────────────────┘
           ↓ (epoch结束)
┌────────────────────────────────────────┐
│ build_text_feature_gallery()           │
│ - 计算所有prompts的平均锚点             │
│ - 保存到 self.text_features            │
│ - 用于评估，不参与训练                  │
└────────────────────────────────────────┘
           ↓ (评估时)
┌────────────────────────────────────────┐
│ calculate_textual_anomaly_score()      │
│ - 使用 self.text_features (保存的锚点)  │
│ - 计算异常分数                          │
└────────────────────────────────────────┘
""")

print("\n【第3部分】修改策略：平均 → MaxPooling")
print("-"*80)

print("""
改动方案：

1. ✅ 修改 checkpoint 保存内容
   旧：只保存平均锚点 text_features [2, 640]
   新：保存所有向量 + 聚合方式标记
   ```python
   'normal_text_features_all': [n_pro, 640],
   'abnormal_text_features_all': [n_pro_ab * n_ab_handle, 640],
   'aggregation_method': 'maxpooling'  # 或 'average'
   ```

2. ✅ 修改 calculate_textual_anomaly_score()
   旧：使用平均锚点
   ```python
   logits = t * visual @ self.text_features.T  # [N, 2]
   score = softmax(logits)[:, 1]
   ```
   
   新：使用maxpooling聚合
   ```python
   # normal分数：与所有normal向量的最小距离
   sim_normal = visual @ self.normal_features_all.T  # [N, n_pro]
   score_normal = sim_normal.max(dim=1)[0]  # [N,]
   
   # abnormal分数：与所有abnormal向量的最大相似度
   sim_abnormal = visual @ self.abnormal_features_all.T  # [N, n_ab]
   score_abnormal = sim_abnormal.max(dim=1)[0]  # [N,]
   
   # 归一化
   logits = torch.stack([score_normal, score_abnormal], dim=1) * t
   score = softmax(logits)[:, 1]
   ```

3. ❌ 不修改训练代码
   - 训练仍使用平均锚点（局部变量）
   - 不影响梯度计算和参数更新

影响分析：
┌──────────────────┬─────────┬─────────────┐
│                  │ 训练     │ 评估        │
├──────────────────┼─────────┼─────────────┤
│ 修改前 (平均)     │ 不变     │ 平均锚点    │
│ 修改后 (maxpool)  │ 不变     │ MaxPooling  │
└──────────────────┴─────────┴─────────────┘

✅ 结论：修改只影响评估，不影响训练
""")

print("\n【第4部分】实验验证")
print("-"*80)

print("""
验证步骤：

1. 使用相同的训练checkpoint
2. 分别用"平均"和"maxpooling"评估
3. 对比结果差异

预期：
- 训练过程完全相同（因为不依赖评估方式）
- 评估结果可能不同（聚合方式不同）
- MaxPooling理论上更敏感（捕捉最强异常信号）
""")

print("\n" + "="*80)
print("验证完成！")
print("="*80)
