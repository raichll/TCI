import numpy as np
import matplotlib.pyplot as plt

def generate_data_with_band(T, shift=0, noise_level=1, seed=0):
    np.random.seed(seed)
    mean_trend = np.linspace(0, 10, T) + shift
    noise = np.random.normal(0, noise_level, T)
    data = mean_trend + noise

    # 为每个点生成上下边界 ± std 或固定范围
    band_width = 2.0
    upper = data + band_width
    lower = data - band_width
    return data, upper, lower

# 设置图像
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
T1, T2, T3 = 50, 50, 8  # 小样本保持 8

# 图① 非稳定性（含动态带）
x = np.arange(T1)
y1, upper1, lower1 = generate_data_with_band(T1, shift=0)
axs[0].plot(x, y1, label='Data', color='blue')
axs[0].fill_between(x, lower1, upper1, color='blue', alpha=0.2, label='Dynamic Region')
axs[0].axvline(x=T1//2, color='red', linestyle='--', label='Distribution Shift')
axs[0].set_title("① Non-stationarity + Dynamic Band")
axs[0].legend()
axs[0].grid(True)

# 图② 噪声（噪声更强，虚区间更宽）
y2, upper2, lower2 = generate_data_with_band(T2, shift=2, noise_level=4)
axs[1].plot(x, y2, label='Noisy Data', color='green')
axs[1].fill_between(x, lower2, upper2, color='green', alpha=0.2, label='Noise Band')
axs[1].set_title("② Noise with Fluctuating Band")
axs[1].legend()
axs[1].grid(True)

# 图③ 小样本（更少数据，仍带动态带）
x3 = np.arange(T3)
y3, upper3, lower3 = generate_data_with_band(T3, shift=5)
axs[2].plot(x3, y3, label='Few Samples', color='purple', marker='o')
axs[2].fill_between(x3, lower3, upper3, color='purple', alpha=0.2, label='Uncertainty Band')
axs[2].set_title("③ Few-shot with Band")
axs[2].legend()
axs[2].grid(True)

plt.suptitle("Dynamic Virtual Zones Along Line Charts", fontsize=16)
plt.tight_layout()
plt.show()
