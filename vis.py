import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def remove_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]

folder_path = 'Time-Series-Library/dataset_refine'
output_folder = 'trend_plots_png_no_outliers'

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
valid_files = []

# 筛选包含'OT'列的有效文件
for file in files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path)
        if 'OT' in df.columns:
            valid_files.append(file)
        else:
            print(f"文件 {file} 不包含 'OT' 列，跳过")
    except Exception as e:
        print(f"读取文件 {file} 出错: {e}")

# 计算子图行列数
n = len(valid_files)
cols = 3  # 每行最多3个子图
rows = math.ceil(n / cols)

# 创建子图
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
fig.suptitle("Trend of OT (Outliers Removed)", fontsize=16)

for idx, file in enumerate(valid_files):
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    ot_data = df['OT'].dropna().reset_index(drop=True)
    ot_filtered = remove_outliers_iqr(ot_data)

    r, c = divmod(idx, cols)
    ax = axes[r][c]
    ax.plot(ot_filtered.reset_index(drop=True), color='blue')
    ax.set_title(file, fontsize=10)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("OT Value")

# 移除多余的子图框
for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    fig.delaxes(axes[r][c])

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(output_folder, 'all_trends_no_outliers.png')
plt.savefig(save_path)
print(f"已保存汇总图: {save_path}")
plt.show()
