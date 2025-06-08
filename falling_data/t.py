import pandas as pd

# 读取原始 Excel 文件
df = pd.read_excel('standardized_data.xlsx')

# 修改列名：将 'time' 改为 'date'（如果存在），最后一列改为 'OT'
if 'time' in df.columns:
    df.rename(columns={'time': 'date'}, inplace=True)
df.rename(columns={df.columns[-1]: 'OT'}, inplace=True)

# 将数据复制 10 次（纵向堆叠）
df_repeated = pd.concat([df] * 9, ignore_index=True)

# 保存为 CSV 文件
df_repeated.to_csv('standardized_data_strength_0.csv', index=False)

print("数据已复制 10 份并保存为 standardized_data_new.csv。")
