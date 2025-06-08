import os
import pandas as pd

folder_path = r'Time-Series-Library/dataset'  # 原始文件夹路径
output_folder = r'Time-Series-Library/dataset_refine'  # 保存路径

# 如果输出文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_csv(file_path)
        
        keep_rows = max(1, len(df) // 4)  
        df_cut = df.iloc[:keep_rows]
        
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}.csv"
        new_path = os.path.join(output_folder, new_filename)
        
        df_cut.to_csv(new_path, index=False)
        print(f"{filename} 处理完成，保存为 {new_filename}")
