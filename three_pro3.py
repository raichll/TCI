import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generate_base_wave(wave_type, T=100):
    x = np.arange(T)
    if wave_type == 'sine':
        return np.sin(0.2 * x)
    elif wave_type == 'trend':
        return 0.05 * x
    elif wave_type == 'jump':
        y = np.zeros(T)
        y[:50] = 0
        y[50:] = 4
        return y
    elif wave_type == 'noise':
        return np.zeros(T)
    else:
        return np.zeros(T)

def generate_multiple_series(base, n_series=30, noise_scale=0.3, seed=0):
    np.random.seed(seed)
    T = len(base)
    series = []
    for _ in range(n_series):
        noise = np.random.normal(0, noise_scale, T)
        series.append(base + noise)
    return np.array(series)

# Setup plot
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
types = ['sine', 'trend', 'jump', 'noise']
titles = [
    "1. Sine-Like Fluctuations",
    "2. Upward Trend with Noise",
    "3. Sudden Distribution Shift",
    "4. Pure Random Noise"
]

n_series = 30
cmap = cm.get_cmap('Blues', n_series)  # Blue color gradient

for ax, wave_type, title in zip(axs.flat, types, titles):
    base = generate_base_wave(wave_type)
    series_set = generate_multiple_series(base, n_series=n_series, noise_scale=0.3)

    for i, s in enumerate(series_set):
        ax.plot(s, color=cmap(i), linewidth=1)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(False)

plt.suptitle("Time Series in Blue Tones (30 per Type)", fontsize=16)
plt.tight_layout()
plt.show()
