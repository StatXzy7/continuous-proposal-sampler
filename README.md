# continuous-proposal-sampler

Linux/CPU 上的连续状态**批量候选采样器**（NumPy 实现，无模型下载）。
目标核与草案核是 1–16 维仿射高斯转移核

```
X_{t+1} = A_t X_t + b_t + N(0, sigma2_t * I_d)
```

两核均值函数 `(A, b)` 可不同，但每一步共享**严格正的相同球形方差** `sigma2_t`。

- 目标逐步基线：每步独立抽目标核。
- 批量前瞻模式：一次生成 `L` 个草案并**批量计算目标均值**，按顺序用密度比
  逐个接受；接受的草案原样提交；首次拒绝时通过**高斯反射最大耦合**
  （平分超平面反射）给出纠偏状态，其后草案全部作废，从纠偏状态继续。
  拒绝后绝不改成独立抽目标核。

## 安装

```bash
pip install -r requirements.txt   # 仅 numpy
```

## 接口

```python
import numpy as np
from gauss_batch import sample_target, sample_coupled_batches, analytic_moments

d, T, L = 3, 10, 4
A, b, s2 = 0.6 * np.eye(d), np.full(d, 0.1), 0.5
Ap, bp = 0.3 * np.eye(d), np.zeros(d)          # 不同均值函数，同一方差
x0 = np.zeros(d)

traj_base = sample_target(x0, (A, b, s2), T, n_traj=100, seed=0)
# (100, T+1, d)

res = sample_coupled_batches(x0, (A, b, s2), (Ap, bp, s2), T, L,
                             n_traj=100, seed=0)
res.trajectories                 # (100, T+1, d)，永远只提交 T 个新状态
res.accept_run_lengths           # 每条轨迹每批接受的草案数
res.rejection_steps              # 每条轨迹首次拒绝位置（-1 = 从未拒绝）
res.n_target_evaluations         # 目标均值评估的【状态数】（含作废草案）
res.n_batches                    # 批量目标均值计算的【批次数】
res.n_rejections                 # 耦合纠偏（拒绝）总次数
res.target_evaluations_per_traj  # 分轨迹计数
res.batches_per_traj

means, covs = analytic_moments(x0, (A, b, s2), T)  # 解析均值/协方差
```

`A/b/sigma2` 均支持单步形状（全程恒定）或长度 `T` 的逐步数组；`x0` 为
确定性初始状态。生成前会拒绝：非正/非有限方差、非有限参数、维度冲突、
非法步数/前瞻长度/轨迹数。极小方差（测试到 `1e-300`）下接受判断在对数
空间用点积完成，避免密度下溢与除零。

## 测试与报告

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_report.py     # 打印经验分布与批处理统计表
```

测试包含：种子重现、接受/拒绝后状态与反射几何、首次拒绝后草案作废、
尾批不多提交、状态计数与批次数不混淆、参数生成前校验、极小方差数值安全，
以及用**预先固定的样本量与误差阈值**（N=10000，|z|<4.5，6 方向 KS D<0.023）
对目标解析均值/协方差与多个方向经验分布的检验。算法推导与实测数据见
[REPORT.md](REPORT.md)。
