import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})

np.random.seed(42)
t = np.arange(150)

# 基础趋势和季节性
trend = 0.0005 * (t - 75)**2 + 0.02 * t
seasonality = 1.5 * np.sin(0.1 * t) * (1 + 0.01 * t) + 1.0 * np.sin(0.35 * t)

# 计算滑动标准差函数
def rolling_std(x, window=10):
    return np.array([np.std(x[max(i - window + 1, 0):i + 1]) for i in range(len(x))])

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 1. 仅趋势 + 季节性
axs[0].plot(t, trend, linestyle='--', color='gray', label='Nonlinear Trend')
axs[0].plot(t, seasonality, linestyle=':', color='#e67e22', label='Seasonality')
axs[0].set_title('1. Trend + Seasonality')
axs[0].legend()
axs[0].grid(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# 2. 加入噪声
noise_std2 = 0.2 + 0.005 * t
noise2 = np.random.normal(0, noise_std2)
y2 = trend + seasonality + noise2
axs[1].plot(t, y2, marker='o', markersize=2, color='#1f77b4', label='Observed with Noise')
axs[1].plot(t, trend, linestyle='--', color='gray')
axs[1].plot(t, seasonality, linestyle=':', color='#e67e22')
axs[1].set_title('2. Add Noise')
axs[1].legend()
axs[1].grid(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# 3. 加入不稳定性区间（滑动std）
window = 10
rolling_std3 = rolling_std(y2, window)
upper3 = y2 + rolling_std3
lower3 = y2 - rolling_std3
axs[2].plot(t, y2, marker='o', markersize=2, color='#1f77b4', label='Observed with Noise')
axs[2].fill_between(t, lower3, upper3, color='#1f77b4', alpha=0.15, label='Instability Range')
axs[2].plot(t, trend, linestyle='--', color='gray')
axs[2].plot(t, seasonality, linestyle=':', color='#e67e22')
axs[2].set_title('3. Instability Range by Rolling Std')
axs[2].legend()
axs[2].grid(False)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

# 4. 小样本 + 异常点示例
# 取150个点中的一小段
small_t = t[50:80]
small_trend = trend[50:80]
small_seasonality = seasonality[50:80]
small_noise_std = 0.2 + 0.005 * small_t
small_noise = np.random.normal(0, small_noise_std)
small_y = small_trend + small_seasonality + small_noise
# 插入异常点
outlier_indices = [5, 20]
small_y[outlier_indices] += np.array([4, -3])

# 计算小样本不稳定区间
rolling_std4 = rolling_std(small_y, window=5)
upper4 = small_y + rolling_std4
lower4 = small_y - rolling_std4

axs[3].plot(small_t, small_y, marker='o', markersize=4, color='#1f77b4', label='Small Sample with Outliers')
axs[3].fill_between(small_t, lower4, upper4, color='#1f77b4', alpha=0.15, label='Instability Range')
axs[3].plot(small_t, small_trend, linestyle='--', color='gray')
axs[3].plot(small_t, small_seasonality, linestyle=':', color='#e67e22')
axs[3].set_title('4. Small Sample & Outliers')
axs[3].legend()
axs[3].grid(False)
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)

# x轴统一设置
axs[3].set_xlabel('Time')
for ax in axs:
    ax.set_ylabel('Value')

# 避免x轴标签过密
axs[3].set_xticks(np.arange(0, 151, 10))

plt.tight_layout()
plt.show()
