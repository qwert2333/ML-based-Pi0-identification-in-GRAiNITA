# GATr 项目与第一次 GPU 训练结果总结

## 1. 文档范围

本文只总结项目中的 GATr 路线，不讨论 BDT 模型。训练结果只采用第一次正式 GPU 训练：

```text
trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/
```

CPU smoke test 不用于性能评价。本文的主要内容包括：

1. 当前 GATr 数据、模型、训练和评估流程；
2. 第一次正式 GPU 训练的分类与回归性能；
3. 当前结果的适用范围和主要问题；
4. 下一步模型与训练优化方向；
5. 向分类 head 加入绝对空间尺度，同时抑制能量依赖的具体方案。

---

## 2. 当前项目目标

该项目使用 GATr（Geometric Algebra Transformer）处理 GRAiNITA 量能器中的 CLUE cluster，目标是识别双光子形成的合并 shower，并同时回归母粒子质量和衰变点。

当前 GATr 是一个 cluster-level 多任务网络，包含三个主要任务：

- **分类**：区分单光子 cluster 和合并在同一个 cluster 中的 $A/\pi^0\rightarrow\gamma\gamma$；
- **质量回归**：预测母粒子质量 $m_A$；
- **衰变点回归**：预测母粒子的三维衰变位置 $(x,y,z)$。

主要代码位置：

- 数据预处理：[`src/preprocessing/gatr_data_prep.py`](src/preprocessing/gatr_data_prep.py)
- GATr 模型和 ROOT streaming dataset：[`src/models/gatr_models.py`](src/models/gatr_models.py)
- 多任务训练：[`src/training/train_gatr.py`](src/training/train_gatr.py)
- 训练入口：[`scripts/run_training.py`](scripts/run_training.py)
- 70k 样本预处理：[`scripts/preprocess_Ayy_gamma_70k.py`](scripts/preprocess_Ayy_gamma_70k.py)
- Condor GPU 提交：[`scripts/submit_gatr_condor.py`](scripts/submit_gatr_condor.py)

---

## 3. 数据处理和标签定义

### 3.1 输入和预处理

预处理以 CLUE cluster 为基本样本单位。每个输出 row 对应一个 cluster，保存：

- 最多 256 个 hit；
- hit 的 $x,y,z$ 坐标；
- hit energy、time 和 layer；
- cluster energy、cluster position 和 hit 数量；
- 母粒子质量、能量、动量、产生点和衰变点等 truth；
- 两个 daughter photon 的运动学信息；
- 分类和回归有效性标记。

当一个 cluster 超过 256 个 hit 时，只保留能量最高的 256 个 hit，并记录保留的能量比例。

数据按 `event_uid` 分组后划分为 training、validation 和 testing。同一事件的多个 cluster 不会跨 split，因此避免了直接的 event leakage。

### 3.2 当前分类标签的物理含义

当前标签定义的核心逻辑是：

```python
is_llp = bool(truth["event_label"])
class_valid = (not is_llp) or len(ids) == 1
label = 1 if is_llp and len(ids) == 1 else 0
```

因此：

- 单光子背景产生的 cluster 是有效背景，`label = 0`；
- $A/\pi^0\rightarrow\gamma\gamma$ 只有在两个光子被 CLUE 合并成一个 cluster 时才是有效信号，`label = 1`；
- 信号事件如果被重建为两个或更多 cluster，则分类标签无效，不参加当前分类训练和评估。

所以当前分类任务准确地说是：

> 在已经得到单个 CLUE cluster 的条件下，区分 merged diphoton cluster 和 single-photon cluster。

当前结果不是完整的 event-level $\pi^0$ identification efficiency，也不包括 CLUE 已经成功把两个光子分开的信号事件。

### 3.3 `Ayy_gamma_70k` 数据统计

数据摘要保存在：

```text
data/GATr/dataset_Ayy_gamma_70k/summary.json
```

| Split | Events | Cluster rows | 有效信号 | 有效背景 | 无效分类标签 | 有效回归样本 |
|---|---:|---:|---:|---:|---:|---:|
| Training | 40,713 | 52,456 | 8,206 | 21,114 | 23,136 | 8,206 |
| Validation | 13,571 | 17,454 | 2,717 | 7,020 | 7,717 | 2,717 |
| Testing | 13,571 | 17,429 | 2,765 | 7,046 | 7,618 | 2,765 |

Testing split 中只有 $2,765+7,046=9,811$ 个 cluster 参与分类评价；其余 7,618 个无效标签约占全部测试 cluster rows 的 43.7%。

信号覆盖多个质量和参考衰变长度：

- 质量约为 0.05、0.10、0.135、0.20、0.50、1.0、2.0 GeV；
- 参考衰变长度约为 0、100、300、800、1600 mm；
- 粒子能量覆盖约 1--80 GeV，不同质量样本的最低能量不同。

---

## 4. 当前模型结构

### 4.1 Hit-level 输入

当前 forward 实际使用的 hit 信息是：

- $x,y,z$；
- hit raw energy；
- hit mask。

虽然预处理和 dataset 还读取了 hit layer，并保存了 hit time，但当前模型没有把 layer 或 time 输入 GATr。

每个 cluster 的坐标首先独立进行中心化和各向同性缩放：

$$
\mathbf r_i^{\mathrm{norm}}
=
\frac{\mathbf r_i-\mathbf r_{\mathrm{center}}}
{R_{\mathrm{cluster}}}.
$$

归一化坐标通过 `embed_point()` 转换为 PGA multivector。两个 scalar input channels 是：

1. hit energy fraction；
2. 广播到每个有效 hit 的 `log(1 + total_cluster_energy)`。

### 4.2 Pooling 和输出 heads

GATr 输出经过两种 pooling：

- 对有效 hit 的普通平均；
- 按 hit energy fraction 加权的平均。

当前 scalar head 输入为：

```text
[pooled_mean, pooled_energy, log_total_energy]
```

分类 head 和质量回归 head 分别作用在相同的 pooled scalar representation 上。衰变点由 energy-weighted multivector pooling 后的 point representation 给出。

### 4.3 第一次 GPU 训练所用模型配置

第一次 GPU 训练实际使用的是 `src/training/train_gatr.py` 中的默认配置，而不是 `configs/GATr_config.py` 中的 `SCALE_UP_CONFIG`：

| 参数 | 数值 |
|---|---:|
| GATr blocks | 2 |
| Hidden multivector channels | 8 |
| Hidden scalar channels | 32 |
| Output scalar channels | 16 |
| Output multivector channels | 1 |
| Scalar-head hidden dimension | 32（默认值） |
| Dropout | 0.1（默认值） |
| 参数总数 | 49,731 |

这是一个相对小的 GATr 模型。配置文件中的 `num_heads` 当前没有传给 `SelfAttentionConfig()`，因此该配置项没有实际控制本次训练。

---

## 5. 多任务损失

分类使用 binary cross entropy with logits。质量目标变换为：

$$
y_m=\log(1+m/\mathrm{GeV}),
$$

并使用 Smooth-L1 loss。衰变点坐标除以 1000 mm 后使用 Smooth-L1 loss。总损失为：

$$
L_{\mathrm{total}}
=
w_cL_{\mathrm{classification}}
+w_mL_{\mathrm{mass}}
+w_dL_{\mathrm{decay}}
+w_rL_{\mathrm{point\ reg}}.
$$

第一次 GPU 训练中：

```text
w_classification = 1.0
w_mass           = 1.0
w_decay          = 1.0
w_point_reg      = 1.0e-3
decay scale      = 1000 mm
```

分类在所有有效分类 cluster 上计算；质量和衰变点损失只在 merged signal 且 truth 有效的 cluster 上计算。

需要特别说明：最终 ROC AUC 只使用分类 head 的 sigmoid score。质量和衰变点预测不会在评估时拼入分类 score，但回归任务通过共享 GATr backbone 和联合损失间接影响分类表示。

---

## 6. 第一次正式 GPU 训练设置

训练目录：

```text
trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/
```

训练环境和超参数：

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA H100L-2-24C MIG 2g.24GB |
| PyTorch | 2.2.1+cu121 |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | $10^{-4}$ |
| Weight decay | $10^{-4}$ |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 |
| Early-stopping patience | 5 |
| Best checkpoint | Epoch 20 |

完整日志：

```text
trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/training.log
```

---

## 7. GPU 训练过程

训练过程存在明显的阶段性：

1. **Epoch 1--2**：分类和衰变点基本停留在初始水平；
2. **Epoch 3--14**：总损失下降主要来自衰变点回归，分类损失长期接近 0.58；
3. **Epoch 15--20**：分类任务开始快速改善，验证分类损失从 0.5768 降到 0.4494。

部分关键 epoch：

| Epoch | Train loss | Validation loss | Validation classification loss | Mass MAE [GeV] | Decay-point MAE [mm] |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1172 | 1.1109 | 0.5917 | 0.1725 | 1962.6 |
| 4 | 0.7728 | 0.7215 | 0.5873 | 0.1709 | 690.1 |
| 12 | 0.6675 | 0.6944 | 0.5842 | 0.1783 | 581.6 |
| 14 | 0.6609 | 0.6933 | 0.5768 | 0.1765 | 615.3 |
| 17 | 0.5822 | 0.6107 | 0.5010 | 0.1787 | 591.1 |
| 19 | 0.5501 | 0.5861 | 0.4811 | 0.1735 | 564.0 |
| 20 | 0.5344 | **0.5592** | **0.4494** | **0.1718** | **593.0** |

第 20 epoch 仍然是最低 validation loss，并且最后数个 epoch 的分类损失仍在快速下降。因此本次训练更像是在 20 epoch 人为截止，而不是已经达到收敛平台。

最终 train/validation loss 差距较小，没有明显的严重过拟合迹象。不过 train loss 是在 dropout 开启且模型参数持续更新的整个 epoch 中累计的，所以与 epoch 末固定模型的 validation loss 只能作近似比较。

训练曲线还显示三个任务的优化不同步：早期主要学习衰变点，分类直到后期才明显改善。这可能来自多任务梯度竞争、损失尺度不匹配或当前小模型容量不足。

---

## 8. GPU 性能结果

### 8.1 Validation checkpoint 指标

Epoch 20 checkpoint 保存的 validation 指标：

| 指标 | 结果 |
|---|---:|
| Total loss | 0.5592 |
| Classification loss | 0.4494 |
| Classification AUC | 0.7958 |
| Mass MAE | 0.1718 GeV |
| Mass RMSE | 0.3390 GeV |
| Decay-point 3D-distance MAE | 593.0 mm |
| Decay-point 3D-distance RMSE | 791.3 mm |

### 8.2 Independent testing split

GPU 训练结束后，代码重新加载最佳 checkpoint 并在 Testing split 上评价：

| 指标 | 结果 |
|---|---:|
| Total loss | 0.5635 |
| Classification loss | 0.4557 |
| Mass loss | 0.0207 |
| Decay-point loss | 0.0864 |
| Classification ROC AUC | **0.797426** |
| Mass MAE | **0.1787 GeV** |
| Decay-point 3D-distance MAE | **571.3 mm** |

Validation 和 testing 结果非常接近，说明模型在当前相同生成分布内的随机 event split 上表现稳定。

### 8.3 AUC 的准确解释

测试 AUC 0.797426 是纯分类 score 的 ROC AUC：

$$
s_{\mathrm{class}}
=
\sigma(\mathrm{classification\ logits}).
$$

ROC 只使用 `s_class` 和二分类标签。质量预测和衰变点预测没有直接加入 score，也不参与工作点判定。

但是，这不是一个“只使用分类 loss 训练”的模型。分类、质量和衰变点任务共享 GATr backbone，因此两个回归损失可能通过共享特征间接改善或损害分类 AUC。需要多任务消融实验才能量化这种影响。

### 8.4 Working point = 0.5

测试集混淆矩阵：

|  | 预测背景 | 预测信号 |
|---|---:|---:|
| 真实背景 | 6,974 | 72 |
| 真实信号 | 1,823 | 942 |

由此得到：

| 指标 | 数值 |
|---|---:|
| Signal efficiency / recall | 34.07% |
| Background false-positive rate | 1.02% |
| Background rejection | 98.98% |
| Signal precision / purity | 约 92.9% |
| Accuracy | 约 80.7% |
| F1 score | 约 49.9% |

因此阈值 0.5 是一个偏高纯度的工作点：背景污染很低，但会损失约三分之二的有效 merged signal。

> 更正：本节初版把混淆矩阵信号行的两列读反，误写为 signal efficiency 65.93%。65.93% 实际是信号被判为背景的比例（1,823/2,765）。用 CPU 重新推理 best checkpoint 得到 TN/FP/FN/TP = 6976/70/1819/946，确认阈值 0.5 下信号效率约 34%。

测试图：

- [`gatr_eval_score_distribution.pdf`](trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/plots/gatr_eval_score_distribution.pdf)
- [`gatr_eval_roc_linear.pdf`](trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/plots/gatr_eval_roc_linear.pdf)
- [`gatr_eval_roc_log.pdf`](trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/plots/gatr_eval_roc_log.pdf)
- [`gatr_eval_contamination_matrix.pdf`](trained_models/GATr/model_Ayy_gamma_70k_gpu_20260925_221006/plots/gatr_eval_contamination_matrix.pdf)

### 8.5 回归结果的限制

质量样本跨越 0.05--2 GeV，而总 MAE 约为 0.18 GeV。这个 aggregate MAE 对低质量点可能大于真实质量本身，对高质量点则可能较合理。Validation RMSE 0.339 GeV 明显高于 MAE 0.172 GeV，提示存在长尾或少量严重失败事件。

因此质量回归需要按真实质量点分别报告：

- prediction mean 和 bias；
- MAE、RMSE 和相对分辨率；
- predicted-vs-true response；
- 各能量、寿命和 opening-angle 区间的性能。

衰变点 MAE 从初期约 1.96 m 降至测试集 0.57 m，表明模型确实学习了该目标。但该误差仍与 0--1600 mm 的参考长度范围处于相似量级。在与 cluster centroid、常数预测或简单几何外推基线比较前，不能确认该回归是否具有足够的物理增益。

---

## 9. 当前主要问题和建议优化方向

### 9.1 首先建立正确的评价体系

整体 AUC 不能充分描述本任务。下一轮应固定输出：

- ROC AUC 和 precision-recall AUC；
- signal efficiency at fixed background efficiencies；
- background efficiency at fixed signal efficiencies；
- AUC、signal efficiency、background efficiency 随能量的变化；
- 按质量、寿命、opening angle、衰变长度和 hit multiplicity 分区的结果；
- score calibration；
- merged-signal selection efficiency 与条件分类效率分开报告。

最终 event-level 性能应分解为：

$$
\epsilon_{\mathrm{total}}
=
\epsilon_{\mathrm{reconstruction/merge\ selection}}
\times
\epsilon_{\mathrm{classifier}\mid\mathrm{valid\ cluster}}.
$$

当前报告的 34.07%（阈值 0.5）只对应第二项。

### 9.2 做多任务消融

至少训练以下四组相同 backbone 和数据划分的模型：

1. classification only；
2. classification + mass；
3. classification + decay point；
4. classification + mass + decay point。

建议设置：

```text
classification only:
    mass_loss_weight = 0
    decay_point_loss_weight = 0

classification + mass:
    mass_loss_weight > 0
    decay_point_loss_weight = 0

classification + decay:
    mass_loss_weight = 0
    decay_point_loss_weight > 0
```

这能判断回归任务对分类 AUC 的真实影响，也能解释当前分类直到 epoch 15 才明显学习的现象。

### 9.3 延长训练并改进调度

因为最佳 checkpoint 出现在最后一个 epoch，建议：

- 把最大 epoch 提高到 50--100；
- 保留 validation early stopping；
- 保存 best classification AUC 和 best total loss 两套 checkpoint；
- 记录每个 epoch 的 AUC，而不只记录 loss；
- 尝试 warmup + cosine decay，或者更灵敏的 ReduceLROnPlateau；
- 每个 epoch 重新 shuffle 数据。

对分类研究而言，仅按 total multitask loss 保存 checkpoint 未必得到最佳分类 AUC，应允许按目标指标选择模型。

### 9.4 调整多任务权重

当前三项损失直接等权相加，但三个任务具有不同样本数、数值尺度和收敛速度。可考虑：

- 手工扫描 loss weights；
- 分类 warmup：先训练分类，再逐步加入回归；
- uncertainty weighting；
- GradNorm；
- PCGrad 或其他 gradient-conflict 方法；
- 分离部分 task-specific layers，减少 head 间竞争。

### 9.5 加入尚未使用的信息

当前 preprocessing 已保存但模型未使用的信息包括：

- hit time；
- detector layer；
- cluster-level hit count；
- cluster absolute spatial scale；
- transverse/longitudinal shower moments。

其中 timing、layer 和 transverse shower width 都可能帮助区分单光子与双光子合并 shower。

### 9.6 模型容量和配置一致性

当前模型只有约 5 万参数。应在建立可靠 baseline 后逐步扫描：

- GATr block 数量；
- hidden multivector/scalar channels；
- attention 配置；
- dropout；
- head hidden dimension。

同时应统一 `configs/GATr_config.py` 和训练脚本中的配置来源，避免配置文件参数看似存在但实际没有用于正式训练。

### 9.7 测试真正的外推能力

当前 split 从所有生成条件中随机抽取 event，所以 train、validation 和 test 都包含相同质量与寿命网格。这验证的是同分布插值能力。

还应设计：

- leave-one-mass-out；
- leave-one-lifetime-out；
- 独立能量区间测试；
- 不同模拟或重建设置测试；
- 不同 detector/noise 条件测试。

---

## 10. 加入绝对空间尺度

### 10.1 为什么当前模型缺少绝对尺度

当前 `_normalise_cluster()` 返回：

```python
coordinates_normalized, centre, coordinate_scale
```

归一化后的坐标进入 GATr；`coordinate_scale` 只用于把衰变点预测转换回绝对坐标。分类 head 看不到该尺度。

因此分类器可以学习归一化后的 shower shape，但无法直接知道 cluster 的绝对宽度。双光子 opening 和 shower separation 可能使 merged cluster 在给定能量下更宽，所以绝对尺度是合理的分类信息。

### 10.2 最小实现

可以在 pooling 后把尺度作为 global scalar 拼接到分类 head：

```python
log_coordinate_scale = torch.log(
    coordinate_scale.reshape(-1, 1).clamp_min(self.eps)
)

pooled_scalars = torch.cat(
    (
        pooled_mean,
        pooled_energy,
        log_total_energy,
        log_coordinate_scale,
    ),
    dim=-1,
)
```

对应的 head 输入维度从：

```python
pooled_dim = 2 * out_s + 1
```

改为：

```python
pooled_dim = 2 * out_s + 2
```

这种信息属于每个 cluster 一个值的 global information，直接加入 pooled representation 比广播给每个 hit 更清晰高效。

### 10.3 比单一 isotropic scale 更好的尺度

当前 `coordinate_scale` 把三个方向混成一个 RMS，而且是按 hit 数量而不是 hit energy 加权。纵向 shower depth 可能掩盖真正用于双光子分离的横向宽度。

建议至少构造：

- energy-weighted transverse RMS；
- energy-weighted longitudinal RMS；
- PCA 第一、第二、第三主轴尺度；
- 主轴尺度比值；
- `log(n_hits)`；
- 能量集中度，例如最高能 hit 所占比例或核心能量比例。

为保持旋转不变性，这些量应使用距离、特征值、范数或标量积构造，而不是直接把某个固定坐标方向的宽度作为输入。

---

## 11. 为什么直接加入总能量不能保证 energy independence

当前模型已经两次向分类路径提供 `log_total_energy`：

1. 作为每个 hit 的第二个 scalar channel 广播进入共享 GATr；
2. pooling 后直接拼接给分类和质量 heads。

如果信号和背景能谱不完全一致，分类器很容易利用能量与标签的相关性。简单地同时加入空间尺度和总能量，不能让模型“理解尺度的能量依赖后主动忽略能量”。网络只会选择最容易降低训练 loss 的信息。

此外，即使完全删除 raw energy，空间尺度本身也随能量变化，分类 score 仍可能间接依赖能量。

因此目标应更准确地定义为：

> 分类器判断一个 cluster 相对于同能量单光子 shower 是否异常宽，而不是利用信号和背景总体能谱的差异。

---

## 12. 推荐的 energy-decorrelated scale 方案

### 12.1 用单光子背景建立能量条件化尺度

首先只使用 training split 中的单光子背景，拟合：

$$
\mu_\gamma(E)
=
\mathbb E[\log R\mid E,\gamma],
$$

$$
\sigma_\gamma(E)
=
\mathrm{Std}[\log R\mid E,\gamma].
$$

然后定义 scale pull：

$$
R_{\mathrm{pull}}
=
\frac{\log R-\mu_\gamma(E)}
{\sigma_\gamma(E)}.
$$

它表示当前 cluster 相对同能量单光子 shower 的尺度偏离。分类 head 使用 `R_pull`，而不是 raw scale 或 raw energy。

对 transverse 和 longitudinal scale 可以分别构造：

$$
R_{T,\mathrm{pull}},\qquad R_{L,\mathrm{pull}}.
$$

背景基线可以使用：

- energy bin 中的均值和标准差；
- median 和 16%--84% 分位区间；
- spline/interpolation；
- 在 background training sample 上训练的小型回归器。

所有基线参数只能在 training split 上拟合，然后冻结并应用于 validation/test，避免泄漏。

### 12.2 推荐的分类与回归分支

建议将分类和回归的 global inputs 分开：

```text
normalized hit geometry + hit energy fractions
                     │
                   GATr
                     │
            pooled shower shape
                     │
          ┌──────────┴──────────┐
          │                     │
 classification head      regression heads
          │                     │
 scale pulls, n_hits       raw energy, raw scales
          │                     │
     classification       mass and decay point
```

示意代码：

```python
shape_features = torch.cat(
    (pooled_mean, pooled_energy),
    dim=-1,
)

classification_features = torch.cat(
    (
        shape_features,
        transverse_scale_pull,
        longitudinal_scale_pull,
        torch.log1p(n_hits),
    ),
    dim=-1,
)

regression_features = torch.cat(
    (
        shape_features,
        log_total_energy,
        log_transverse_scale,
        log_longitudinal_scale,
    ),
    dim=-1,
)

classification_logits = self.classifier(classification_features)
mass_log1p = self.mass_regressor(regression_features)
```

如果 raw `log_total_energy` 继续广播进入共享 GATr，分类 head 仍可能从 `s_out` 恢复能量信息。对严格的 energy-decorrelated classification，推荐：

- GATr backbone 只接收 hit energy fractions，不接收 raw total energy；
- raw total energy 只在 pooling 后提供给 regression heads；
- classification head 只接收 energy-conditioned scale pulls 和 shower-shape representation。

但即使如此，其他 shower 特征仍可能携带能量信息，所以还需要数据层和 loss 层的约束。

### 12.3 匹配信号和背景能谱

训练分类器前，应让信号和背景具有相同或近似相同的能量分布。可选择：

- 按能量 bin 对信号和背景 reweight；
- 每个能量 bin 抽取相同数量的两类样本；
- 使用 energy-stratified batch sampler；
- 在 BCE 中加入 event weights。

能谱匹配是避免能量成为标签捷径的基础。即使采用 adversarial decorrelation，也建议先完成这一步。

### 12.4 显式 decorrelation loss

如果能谱匹配和 scale pull 后仍存在明显的 score-energy correlation，可以增加 decorrelation loss：

$$
L
=
L_{\mathrm{classification}}
+\lambda_{\mathrm{decor}}
L_{\mathrm{decorrelation}}.
$$

可选方法包括：

- distance correlation / DisCo；
- adversarial energy regressor + gradient reversal；
- mutual-information penalty；
- energy-bin 间 score-distribution matching。

如果物理目标主要是保持稳定的 background rejection，可以只在背景样本上施加：

$$
L_{\mathrm{decorrelation}}
=
\mathrm{dCorr}
\left(s_{\mathrm{class}},\log E\right)_{y=0}.
$$

这样直接约束单光子误判率不要随能量显著变化。若要求两类 score 都与能量无关，则应分别在信号和背景上计算 penalty。

`lambda_decor` 需要扫描。约束过强通常会降低 AUC，因为真实 shower morphology 本身确实随能量变化。

### 12.5 后处理选择

如果只需要固定 background efficiency 在不同能量下保持稳定，也可以采用类似 DDT 的后处理：

$$
s_{\mathrm{DDT}}
=
s-q_\gamma(E),
$$

其中 $q_\gamma(E)$ 是背景 score 在给定能量下的目标分位数。这不会使原始模型表示完全 energy-independent，但可以获得随能量稳定的工作点。

---

## 13. Energy decorrelation 的验证标准

加入绝对尺度后，不能只比较 overall AUC。每个模型都应报告：

1. overall ROC AUC；
2. 每个能量 bin 的 ROC AUC；
3. 固定阈值下 signal efficiency vs energy；
4. 固定阈值下 background efficiency vs energy；
5. 固定 background efficiency 下 signal efficiency vs energy；
6. signal 和 background 的 mean score vs energy；
7. score 与 energy 的 Pearson/Spearman correlation；
8. background score 和 energy 的 distance correlation；
9. decorrelation 前后的 AUC 损失；
10. scale、scale pull 和 score 的二维分布。

应重点避免一种假象：overall AUC 提高，但只是因为模型学到了信号和背景不同的能谱，而不是双光子 shower 的空间结构。

---

## 14. 建议的实施顺序

### Phase 1：建立可靠 baseline

- 固定当前 dataset split 和随机种子；
- 增加每 epoch AUC 记录；
- 输出能量、质量、寿命和 opening-angle 分区指标；
- 做 classification-only 多任务消融；
- 延长训练并保存 best-AUC checkpoint。

### Phase 2：只加入绝对尺度

- 保持其他设置不变；
- 加入 `log_coordinate_scale`；
- 再加入 transverse/longitudinal energy-weighted scales；
- 比较 overall 和 per-energy AUC。

这一步用于确认绝对尺度是否真的提供额外判别力。

### Phase 3：消除明显能谱捷径

- 对信号和背景做 energy reweight；
- 从分类 head 去掉 raw `log_total_energy`；
- 不再把 raw total energy 广播到分类共享表示；
- raw energy 只保留在 regression heads。

### Phase 4：使用能量条件化尺度

- 在 background training sample 上拟合 $\mu_\gamma(E)$ 和 $\sigma_\gamma(E)$；
- 向分类 head 输入 scale pulls；
- 比较 raw scale 和 conditional scale 两种方案。

### Phase 5：必要时加入显式 decorrelation

- 优先尝试 background-only DisCo；
- 扫描 `lambda_decor`；
- 画出 AUC 与 energy correlation 的 trade-off；
- 根据最终物理工作点选择模型，而不是只选择最高 overall AUC。

---

## 15. 总结

第一次正式 GPU 训练证明当前 GATr pipeline 能完整完成数据读取、多任务训练、checkpoint 保存和独立 testing：

- 条件式 merged-cluster 分类测试 AUC 为 **0.797426**；
- 在 threshold 0.5 下，signal efficiency 为 **34.07%**，background rejection 为 **98.98%**（初版误写为 65.93%，见 8.4 节更正）；
- validation 和 testing 一致，没有明显分布内过拟合；
- 最佳 checkpoint 出现在最后一个 epoch，训练尚未显示清晰收敛；
- mass 和 decay-point heads 学到了一定信息，但需要分区指标和简单基线才能判断物理价值。

下一步最重要的不是直接扩大模型，而是先明确评价范围、完成多任务消融，并检查分类性能对能量的依赖。

对于绝对尺度，推荐保留归一化坐标作为 GATr 的 hit-level 输入，同时把 transverse/longitudinal absolute scales 作为 pooling 后的 global scalars。为了避免分类器利用能谱差异，应优先采用 energy reweighting 和基于单光子背景的 energy-conditioned scale pulls；若仍有明显相关性，再加入 DisCo 或 adversarial decorrelation。这样才能让分类器主要学习“相对于同能量单光子 shower 的异常空间尺度”，而不是把总能量当作标签捷径。
