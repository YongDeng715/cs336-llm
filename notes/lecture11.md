# Lecture-11 Scaling Law（整体综述 v2）

> 本文沿用“实践→理论→实践→未来”四段框架，但定位为 **Scaling Law 的系统综述**，不是课堂教学设计。重点回答：在给定资源约束下，如何统一理解参数量（N）、数据量（D）与计算量（C）对模型损失与能力的影响。

## 1. 实践起点（案例 + 经验驱动）

### 1.1 经验事实：大模型训练的三元约束
在主流 dense Transformer 预训练中，训练计算量常被近似写为：

$$
C \approx 6ND
$$

其中：
- \(N\)：可训练参数量（parameters）
- \(D\)：训练 token 总量（dataset size in tokens）
- \(C\)：训练 FLOPs（compute budget）

该近似是 Scaling Law 实践化的出发点：同一 \(C\) 下，提升 \(N\) 会压缩可训练 \(D\)，反之亦然。

### 1.2 两代主线结论：Kaplan → Chinchilla
- **Kaplan 系列（2020）**：首先系统给出 \(L\) 随 \(N\)、\(D\)、\(C\) 的幂律缩放关系，奠定“可由小规模外推大规模”的方法论。
- **Chinchilla（2022）**：在更充分网格实验上提出联合形式
  $$
  L(N,D)=E + A N^{-\alpha}+B D^{-\beta}
  $$

  并给出典型指数（公开报告常引用）($\alpha\approx0.34,\beta\approx0.28$)，说明先前许多模型在固定训练计算量下偏向“参数过大、数据不足”。
- **工程后果**：行业从“盲目增大参数”逐步转向“参数与 token 更平衡”的策略。常见经验区间由 Chinchilla 风格的 token/param 比例（约 20:1 量级）引导，而不是极低 token 覆盖。

### 1.3 Iso-FLOPs 与 sweet spot 的可视化含义
将 ($D=C/(6N)$) 代入后，同一 FLOPs 预算下的损失曲线通常呈“先降后升”的 U 形：
- 左侧：\(N\) 太小，容量受限（model-limited）
- 右侧：\(N\) 太大但 \(D\) 不足，数据受限（data-limited）
- 底部：compute-optimal sweet spot（\(N^*,D^*\)）

这也是公开 Iso-FLOPs 图里“scale-up”与“train-more”路线差异的本质。

### 1.4 跨模态侧证：语言模型与视觉模型的一致性
- 在 LLM 实践中，许多后续开源/闭源报告都显示：当 token 预算、数据质量与训练稳定性提高时，性能更接近幂律可预测区间。
- 在 ViT 与 DINO 系列中，虽然数据单位从“文本 token”变为“图像 patch token + 多视角增强样本”，但仍存在同样的 compute/data/capacity 资源分配问题。

---

## 2. 理论抽象（从现象到隐式原理）

### 2.1 联合幂律与最优分配公式
给定

$$
L(N,D)=E + A N^{-\alpha}+B D^{-\beta},\quad C\approx 6ND
$$

可得在固定 \(C\) 下的最优分配（忽略常数细节后）
$$
N_{opt}(C)\propto C^{\frac{\beta}{\alpha+\beta}},\qquad
D_{opt}(C)\propto C^{\frac{\alpha}{\alpha+\beta}}
$$
并且最优损失随计算量下降为
$$
L_{opt}(C)=E+K C^{-\gamma},\qquad
\gamma=\frac{\alpha\beta}{\alpha+\beta}
$$
这解释了“加算力会稳定降损失，但边际收益递减”的经验事实。

### 2.2 偏导视角：何时该扩参数，何时该扩数据

a) 参数方向边际收益：
$$
\left|\frac{\partial L}{\partial N}\right|=A\alpha N^{-(\alpha+1)}
$$

b) 数据方向边际收益：
$$
\left|\frac{\partial L}{\partial D}\right|=B\beta D^{-(\beta+1)}
$$

当其中一侧边际收益显著更大时，资源应优先投向该侧；当两侧接近时，系统位于近似 compute-optimal 区域。该判据在自动化训练预算分配中非常实用。

### 2.3 与 bias/variance、double descent 的关系
- ($N^{-\alpha}$) 项可理解为近似误差（capacity/bias）衰减。
- ($D^{-\beta}$) 项可理解为估计误差（data/variance）衰减。
- 在插值阈值附近，部分下游离散指标可能出现非单调或“跃迁”现象；但在 token-level loss 上，幂律常更稳定。

### 2.4 为什么同一公式会在现实中“看起来失效”
常见原因并非幂律本身错误，而是“有效变量”被替换：
- **有效数据量**不等于原始 token 数（去重、质量过滤、分布匹配会改变 \(D_{eff}\)）
- **有效计算量**小于账面 FLOPs（利用率、通信开销、失稳回滚会折损 \(C_{eff}\)）
- **目标函数变化**（从预训练 loss 到指令跟随/推理准确率）会改变可观测指数

---

## 3. 再回到实践（跨模型类比与训练工程）

### 3.1 一条可执行的 scaling 工作流
1. **小规模试验网格**：在多个 \((N,D)\) 点采样，记录验证损失与训练稳定性。
2. **拟合联合模型**：估计 \(A,B,\alpha,\beta,E\)，并构造 Iso-FLOPs / Iso-Loss 图。
3. **预算反推**：给定目标 \(C\) 与交付时间，解出 \((N^*,D^*)\) 与容错区间。
4. **在线校准**：训练中持续比对“预测 loss 曲线 vs 实际曲线”，出现系统偏离时重估参数。

### 3.2 训练工程变量如何改变“有效 C、D、N”
- **batch size 与学习率日程**：决定优化噪声与收敛效率，直接影响 \(C_{eff}\)。
- **gradient accumulation / checkpoint / 并行策略**：影响吞吐、稳定性与内存边界，进而改变可达 sweet spot。
- **数据管线质量**：高重复、低信息密度数据会显著降低 \(D_{eff}\)。

### 3.3 ViT / DINO 语境中的映射关系
视觉预训练可将“数据规模”重写为
$$
D_{eff} \sim \text{图像数} \times \text{每图 token 数} \times \text{有效增强视角} \times \text{质量权重}
$$
因此 patch size、分辨率、增广策略、数据多样性都会影响 scaling 曲线斜率。DINOv2/v3 这类系统中的关键经验是：**数据质量提升与模型扩展必须协同**，否则会出现“算力增加但收益钝化”。

### 3.4 稀疏与混合架构下的变体（MoE 等）
对 MoE，单步训练计算更接近活跃参数 \($N_{act}$\)，而模型容量又受总参数 \($N_{tot}$\) 影响。实践中需要从 $L(N,D)$ 扩展到更高维的
$$
L(N_{tot},N_{act},D,\text{quality})
$$
这也是“同等训练 FLOPs 下 MoE 可能更优，但推理成本与路由稳定性更复杂”的理论来源。

---

## 4. 未来问题与发展（实践-理论闭环）

### 4.1 从数量扩展转向“质量扩展”
下一阶段的核心不只是增大 \(D\)，而是提高 \(D_{eff}\)：去重、重加权、课程学习与合成数据混配策略，可能重写传统指数并改变最优配比。

### 4.2 从训练最优转向全生命周期最优
经典 scaling law 优化的是训练损失；产业部署更关心
$$
C_{total}=C_{train}+\lambda C_{infer}
$$
当推理流量巨大时，最优 \($(N,D)$\) 可能偏离 Chinchilla 训练最优点，形成 inference-aware scaling 新问题。

### 4.3 多模态与长上下文的统一度量
文本、图像、音频 token 的信息密度与计算开销差异巨大，未来需要统一的“跨模态等效 token / 等效 compute”标尺，才能让 iso-loss 预测可迁移。

### 4.4 后训练与测试时计算的 scaling law
SFT、RLHF、DPO、LoRA、检索增强与 test-time compute（如推理时搜索/自一致性）正在形成“第二阶段 scaling”：
- 预训练决定基础斜率
- 后训练决定能力投影方向
- 测试时计算决定可兑现上限

未来高价值问题不再是单一 \(L(C)\)，而是多阶段、多目标的联合 scaling。

---

## 参考资料（公开论文与技术报告）
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* / Chinchilla (2022)
- Henighan et al., *Scaling Laws for Autoregressive Generative Modeling* (2020)
- DINOv2 及后续公开技术报告（视觉自监督扩展实践）
- 近年开源/闭源大模型公开训练报告（用于验证 token/param 与 compute 调度趋势）

```mermaid
flowchart LR
    Practice1[实践起点
    （经验事实）] --> Theory[理论抽象]
    Theory --> Practice2[工程回投与跨模型类比]
    Practice2 --> Future[未来问题与新范式]
    Future --> Practice1
