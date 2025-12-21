import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# --- 配置学术风格 ---
# 如果运行报错提示没有 LaTeX，把 usetex 改为 False
# --- 修改后的配置块 ---
plt.rcParams.update({
    "text.usetex": False,        # 禁用外部 LaTeX，解决 FileNotFoundError
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"], # DejaVu 兼容性更好
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})

# --- 下方的绘图标签也建议做微调以兼容 Mathtext ---
# 例如将 r"\textbf{DA-Q (Ours)}" 改为 r"$\mathbf{DA-Q\ (Ours)}$"

def silu(x):
    return x * torch.sigmoid(x)

# --- 1. 准备数据 ---
x = torch.linspace(-6, 6, 2000, requires_grad=True) #稍微拉宽一点范围看看边缘
y = silu(x)

# 计算导数权重
grad1, = torch.autograd.grad(y.sum(), x, create_graph=True)
grad2, = torch.autograd.grad(grad1.sum(), x)
# 原始导数权重 (Pure Gradient Base)
w_grad_raw = 0.7 * grad1.abs() + 0.3 * grad2.abs()

x_np = x.detach().numpy()
y_np = y.detach().numpy()
w_grad_np = w_grad_raw.detach().numpy()

# --- 关键修正：归一化后再混合，确保视觉效果 ---
# 1. 归一化导数权重，使其和为1
w_grad_norm = w_grad_np / np.sum(w_grad_np)
# 2. 创建归一化的均匀权重，和也为1
w_uniform_norm = np.ones_like(w_grad_norm) / len(x_np)

# --- 2. 模拟三种策略 (4-bit, 16 levels) ---
n_bits = 4
n_levels = 2**n_bits
target_cdf = np.linspace(0, 1, n_levels)

# 策略A: Uniform (基准)
levels_uniform = np.linspace(x_np.min(), x_np.max(), n_levels)

# 策略B: Pure Grad (纯导数，alpha=1.0)
cdf_pure = np.cumsum(w_grad_norm)
levels_pure = np.interp(target_cdf, cdf_pure, x_np)

# 策略C: DA-Q Hybrid (混合，alpha=0.7 用于演示)
# 现在的混合是真正的 70% 导数分布 + 30% 均匀分布
alpha_visual = 0.7 
w_hybrid_norm = alpha_visual * w_grad_norm + (1 - alpha_visual) * w_uniform_norm
cdf_hybrid = np.cumsum(w_hybrid_norm)
levels_daq = np.interp(target_cdf, cdf_hybrid, x_np)


# --- 3. 精美绘图 ---
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.5, 1]})

# --- 上图：函数与权重 ---
# 画 SiLU 曲线
line_f, = ax1.plot(x_np, y_np, color='#2c3e50', lw=2.5, zorder=10, label=r'SiLU Activation $f(x)$')

# 画灵敏度权重区域 (为了好看，稍微放大一点显示的幅度)
ax1.fill_between(x_np, y_np.min(), w_grad_np * 2.5 + y_np.min(), color='#3498db', alpha=0.15, label=r'Sensitivity $\omega(x)$ (Grad-based)')

# 画 DA-Q 的量化点垂线
for i, l in enumerate(levels_daq):
    label = r'DA-Q Levels (Ours)' if i == 0 else None
    # 用一点渐变色增加高级感
    ax1.axvline(x=l, color='#e74c3c', alpha=0.4, linestyle='--', lw=1.2, label=label)

ax1.set_title(r"Derivative-Aware Quantization: Sensitivity-to-Grids Mapping", fontsize=18, pad=15)
ax1.set_ylabel(r"$f(x)$", fontsize=18)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
ax1.set_xlim(x_np.min(), x_np.max())

# --- 下图：Rug Plot 对比 ---
# 使用不同的颜色和高度来区分
# Uniform
ax2.eventplot(levels_uniform, lineoffsets=2.5, linelengths=0.6, color='#95a5a6', alpha=0.8)
ax2.text(x_np.min() + 0.2, 2.5, "Uniform\n(Baseline)", ha='left', va='center', color='#7f8c8d', fontsize=11)

# Pure Grad
ax2.eventplot(levels_pure, lineoffsets=1.5, linelengths=0.6, color='#f1c40f')
ax2.text(x_np.min() + 0.2, 1.5, "Pure Grad\n($\\alpha=1.0$)", ha='left', va='center', color='#d35400', fontsize=11)

# DA-Q (Ours)
ax2.eventplot(levels_daq, lineoffsets=0.5, linelengths=0.8, color='#e74c3c', lw=2)
ax2.text(x_np.min() + 0.2, 0.5, r"DA-Q (Ours)"+"\n($\\alpha=0.7$)", ha='left', va='center', color='#c0392b', fontsize=11)


ax2.set_yticks([])
ax2.set_xlabel(r"Input Activation Value $x$", fontsize=18)
ax2.set_xlim(x_np.min(), x_np.max())
# 去掉下图多余的边框线
for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)
ax2.grid(False) # Rug plot 不需要网格

plt.tight_layout()
# 保存高清图，适合放在 README
plt.savefig("daq_concept_hd.png", dpi=300, bbox_inches='tight')
print("新的高清可视化图表已生成：daq_concept_hd.png。现在应该能看出区别了。")
plt.show()