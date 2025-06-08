import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import os # <--- 确保导入了 os 模块

# ==============================================================================
# 这里应该是你脚本中已经存在的其他函数定义，例如：
# SimpleTransformerEncoder 类, 
# _generate_positional_encoding 方法,
# compute_attention_scores 函数,
# generate_augmented_time_series_with_transformer 函数,
# roulette_selection 函数,
# add_noise_neighborhood 函数
# 请确保它们都在这个 main 函数之前定义好了。
# 我会从你的上一个代码版本中复制这些，以确保完整性，但你只需要替换 main 函数。
# ==============================================================================

# (此处省略了 SimpleTransformerEncoder, _generate_positional_encoding, 
#  compute_attention_scores, roulette_selection, add_noise_neighborhood,
#  generate_augmented_time_series_with_transformer 函数的定义，
#  假设它们已经存在于你的脚本中并且是正确的)

# --- 你需要将下面的 main 函数替换掉你脚本中原来的 main 函数 ---
def main():
    # 1. 读取输入文件路径 (与之前逻辑类似)
    filepath = r"Time-Series-Library\\dataset\\ETTh1.csv" # <--- 请确保这是你实际的输入文件路径
                                            # 例如："e:\project\TCI\data\standardized_data.xlsx"
                                            # 你可能需要根据实际情况调整此路径或使其动态获取
    try:
        # 尝试使用 'openpyxl' 引擎读取 .csv 文件
        input_df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误：文件 '{filepath}' 未找到。请检查路径。")
        print("为演示目的，正在创建一个虚拟 DataFrame。")
        L_dummy, N_dummy_vars = 50, 4 
        N_dummy_cols = N_dummy_vars + 1 
        data_dummy = np.random.rand(L_dummy, N_dummy_cols)
        data_dummy[:,0] = np.arange(1, L_dummy + 1) 
        input_df = pd.DataFrame(data_dummy, columns=["Time"] + [f"Var{i}" for i in range(1,N_dummy_vars+1)])
    except ValueError as e:
        if "Excel file format cannot be determined" in str(e):
            print(f"错误：无法确定 Excel 文件格式 '{filepath}'。请确保它是有效的 .xlsx 或 .xls 文件，并已安装相应引擎 (如 openpyxl 或 xlrd)。")
        else:
            print(f"读取 Excel 文件时发生错误: {e}")
        return # 无法继续，则退出
    input_begin= input_df.copy()  # 保留原始 DataFrame 的副本
    # 2. 处理第一列 "Time" (与之前逻辑类似)
    try:
        if not pd.api.types.is_numeric_dtype(input_df.iloc[:, 0]) or input_df.iloc[:,0].isnull().any():
            print(f"警告：第一列数据类型为 {input_df.iloc[:, 0].dtype} 或包含空值，正在转换为整数序列。")
            input_df.iloc[:, 0] = range(1, len(input_df) + 1)
        else:
            # 如果已经是数字且没有空值，确保它是1到L的序列（如果这是核心假设）
            input_df.iloc[:, 0] = range(1, len(input_df) + 1)
    except Exception as e:
        print(f"处理 DataFrame 第一列时出错: {e}")
        print("回退到简单赋值方式处理第一列。")
        input_df.iloc[:, 0] = range(1, len(input_df) + 1)

    # 3. 清理数据列并转换为数字类型 (与之前逻辑类似)
    print("正在尝试清理数据列并将其转换为数字类型...")
    for col_idx in range(1, input_df.shape[1]):
        col_name = input_df.columns[col_idx]
        if input_df[col_name].dtype == 'object':
            print(f"列 '{col_name}' 的数据类型是 object。正在尝试将其转换为数字类型。")
            input_df[col_name] = pd.to_numeric(input_df[col_name], errors='coerce')
            if input_df[col_name].isnull().any():
                 print(f"警告：列 '{col_name}' 中存在无法转换为数字的值，这些值已被转换为 NaN。")

    if input_df.iloc[:, 1:].isnull().any().any():
        print("警告：在尝试数字转换后，数据列中发现 NaN。正在用 0 填充 NaN。")
        fill_values = {col_name: 0 for col_name in input_df.columns[1:] if input_df[col_name].isnull().any()}
        input_df.fillna(value=fill_values, inplace=True)
    
    input_data = input_df.to_numpy()
    input_data = input_data.astype(np.float32)  # 确保转换为 float32

    if input_data.dtype == np.object_:
        print("警告：input_data 的整体数据类型仍然是 'object'。正在确保数据部分为 float32。")
        try:
            if input_data.shape[1] > 1:
                 data_part = input_data[:, 1:].astype(np.float32)
                 first_col = input_data[:, 0].reshape(-1, 1)
                 input_data = np.hstack((first_col, data_part))
            elif input_data.shape[1] == 1 and input_data.dtype == np.object_:
                 input_data = input_data.astype(np.float32)
            print("已成功将 input_data 的数据部分转换为 float32。")
        except ValueError as e:
            print(f"严重错误：无法将 input_data 的数据部分转换为 float32: {e}")
            print("请清理你的 Excel 文件中数据列（从第二列开始）的非数字值。")
            return # 无法继续，则退出
            
    print(f"清理后的原始数据形状: {input_data.shape}, 数据类型: {input_data.dtype}")
    if input_data.shape[1] > 1:
        print(f"清理后的 input_data[:, 1:] 数据类型: {input_data[:, 1:].dtype}")


    # 4. 生成数据：只生成一个新的增强样本 (n=1)
    n_new_samples_to_generate = 1 
    print(f"正在生成 {n_new_samples_to_generate} 个新的增强数据样本...")
    # 假设 generate_augmented_time_series_with_transformer 函数已在脚本中定义
    df_all_samples = generate_augmented_time_series_with_transformer(
        input_data, 
        s=3, 
        n=n_new_samples_to_generate, # <--- 设置 n=1
        noise_std=0.01, 
        seed=42
    )

    # 5. 提取新生成的增强样本并调整
    df_to_save = pd.DataFrame() # 初始化一个空的 DataFrame
    if not df_all_samples.empty:
        # 新生成的增强样本的 "Sample" ID 应该是 1 
        # (因为原始样本是0, n_new_samples_to_generate=1 生成的是第1个增强样本)
        if 1 in df_all_samples["Sample"].unique():
            df_newly_generated = df_all_samples[df_all_samples["Sample"] == 1].copy()
            
            if not df_newly_generated.empty:
                print("已成功提取新生成的增强样本。")
                # “替换”原来的数据：在新文件中，这个样本可以被认为是主要的，将其 "Sample" ID 设置为 0
                df_newly_generated["Sample"] = 0
                df_to_save = df_newly_generated
            else:
                print("错误：未能提取到新生成的增强样本 (Sample ID 为 1 的样本为空)。")
        else:
            print("错误：在生成的数据中未找到 Sample ID 为 1 的新增强样本。")
            print(f"所有生成的样本ID为: {df_all_samples['Sample'].unique()}")
    else:
        print("错误：`generate_augmented_time_series_with_transformer` 函数返回了空的 DataFrame。")

    # 6. 保存到新的文件名
    if not df_to_save.empty:
        # 从输入文件路径创建新的输出文件名
        base_filepath, ext_filepath = os.path.splitext(filepath)
        new_output_filename = base_filepath + "_new" + ext_filepath
        df_to_save = df_to_save.drop(columns=df_to_save.columns[-1])
        df_to_save = df_to_save.reset_index(drop=True)
     
        df_to_save['Time'] = input_begin['date']  # 保留原始数据列]
        df_to_save.columns= input_begin.columns  # 保留原始列名，去掉 "Sample" 列
        
        try:
            df_to_save.to_csv(new_output_filename, index=False)
            print(f"已将新生成的单个增强数据样本保存到: {new_output_filename}")
            print(f"保存的数据形状: {df_to_save.shape}")
        except Exception as e:
            print(f"保存文件 '{new_output_filename}' 时出错: {e}")
    else:
        print("没有生成或提取到有效的新数据样本可供保存。")

if __name__ == "__main__":
    # ==============================================================================
    # 在此之下，你需要确保你的脚本中已经定义了以下所有函数和类：
    # - class SimpleTransformerEncoder(nn.Module): ...
    # - def compute_attention_scores(data_np, mode='row'): ...
    # - def generate_augmented_time_series_with_transformer(...): ...
    #   (包含 roulette_selection 和 add_noise_neighborhood 的定义或调用)
    #
    # 例如，你需要从之前的代码中复制这些函数的定义到这个脚本的全局作用域，
    # 或者确保它们是通过 import 导入的。
    # 为了这个示例能独立运行（假设你把函数定义放在同一个文件），
    # 我会象征性地添加占位符，实际使用时你需要填入完整的函数定义。
    # ==============================================================================

    # --- 以下是示例占位符，你需要用你实际的函数定义替换 ---
    class SimpleTransformerEncoder(nn.Module):
        def __init__(self, d_model, nhead):
            super().__init__()
            if d_model % nhead != 0: raise ValueError("d_model must be divisible by nhead")
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4 if d_model > 0 else 2048) # d_model*4 might be 0
            if d_model == 0 : self.encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first=True, dim_feedforward=4) # fallback for d_model =0
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        def _generate_positional_encoding(self, length, d_model):
            if d_model == 0: d_model = 1 # Avoid division by zero if d_model somehow is 0
            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1 and pe[:, 1::2].size(1) > 0:
                 pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        def forward(self, x):
            if x.shape[2] == 0: # if d_model is 0 from input
                print("Warning: Input d_model is 0 in SimpleTransformerEncoder.forward. Adjusting.")
                # This case should ideally be prevented by input validation or design
                # For now, just pass through or return error
                return x 
            seq_len, d_model_input = x.shape[1], x.shape[2]
            pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
            x = x + pe.unsqueeze(0)
            return self.encoder(x)

    def compute_attention_scores(data_np, mode='row'):
        device = torch.device("cpu")
        d_model_transformer = 1
        if data_np.size == 0: # Handle empty input array
            print(f"Warning: compute_attention_scores received empty data_np for mode='{mode}'. Returning empty scores.")
            return np.array([])

        if mode == 'row':
            _data_np_reshaped = data_np.reshape(-1, 1)
            data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0)
        elif mode == 'col':
            data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError("mode must be 'row' or 'col'")

        if data_tensor.shape[2] == 0 : # d_model is 0
             print(f"Warning: data_tensor for transformer has d_model=0 in compute_attention_scores (mode='{mode}'). Returning empty scores.")
             return np.array([])


        transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)
        with torch.no_grad():
            output = transformer(data_tensor)
            attention_scores = output.squeeze(0).squeeze(-1).cpu().numpy()
        return attention_scores

    def generate_augmented_time_series_with_transformer(original_data, s=2, n=10, noise_std=0.01, seed=42):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if original_data.ndim < 2 or original_data.shape[0] == 0 or original_data.shape[1] == 0 :
            print("Warning: original_data is empty or not 2D. Returning empty DataFrame.")
            return pd.DataFrame()

        L, N_cols = original_data.shape

        def roulette_selection(scores, k):
            if scores.size == 0: return np.array([], dtype=int) # Handle empty scores
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            stable_scores = scores - np.max(scores)
            probs = np.exp(stable_scores)
            probs_sum = probs.sum()
            if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
                probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
            else:
                probs /= probs_sum
            if probs.size == 0: return np.array([], dtype=int)
            probs = np.maximum(0, probs)
            probs_sum_final = probs.sum()
            if probs_sum_final == 0 :
                 probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
            else:
                probs /= probs_sum_final
            if probs.size == 0: return np.array([], dtype=int)
            
            k_actual = min(k, len(scores))
            if k_actual == 0 : return np.array([], dtype=int)
            if np.any(np.isnan(probs)) or not np.isclose(np.sum(probs), 1.0): # Final sanity check for probs
                # print(f"Warning: Probabilities are invalid in roulette_selection: {probs}. Using uniform.")
                probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
                if probs.size == 0: return np.array([], dtype=int)

            return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)

        def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
            length = len(data_array)
            if length == 0: return data_array
            noisy_array = data_array.copy()
            start_idx = max(0, center_idx - (window_size - 1) // 2)
            end_idx = min(length, start_idx + window_size)
            if end_idx == length: start_idx = max(0, end_idx - window_size)
            actual_len_to_noise = end_idx - start_idx
            if actual_len_to_noise > 0:
                noise = np.random.normal(0, noise_std_val, actual_len_to_noise)
                noisy_array[start_idx:end_idx] += noise
            return noisy_array

        augmented_list = []
        if L == 0: # No time points
            print("Warning: Original data has L=0 time points. Cannot generate meaningful augmentations.")
            # Add original (empty) and one attempt at 'augmented' (also empty structure)
            augmented_list.append(original_data.copy())
            if n > 0 : augmented_list.append(original_data.copy()) # Add one 'empty' augmented sample
        else:
            for i in range(n + 1): # n+1 to include original if L > 0
                if i == 0:
                    augmented_list.append(original_data.copy())
                else:
                    new_data = augmented_list[0].copy()
                    num_actual_variables = N_cols - 1
                    if num_actual_variables <= 0: k_vars = 0
                    else: k_vars = max(1, math.ceil(num_actual_variables / 10.0))

                    if k_vars > 0 and num_actual_variables > 0:
                        for row_idx in range(L):
                            row_variables_data = new_data[row_idx, 1:]
                            if row_variables_data.size == 0: continue
                            scores_vars = compute_attention_scores(row_variables_data, mode='row')
                            if scores_vars.size == 0: continue
                            var_indices_in_row_data = roulette_selection(scores_vars, k_vars)
                            for var_idx_local in var_indices_in_row_data:
                                actual_col_idx = var_idx_local + 1
                                new_data[row_idx, actual_col_idx] += np.random.normal(0, noise_std)
                    
                    k_times = max(1, math.ceil(L / 10.0))
                    if k_times > 0 and L > 0 and N_cols > 1:
                        for col_idx in range(1, N_cols):
                            current_variable_timeseries = new_data[:, col_idx].copy()
                            if current_variable_timeseries.size == 0: continue
                            col_data_for_attention = current_variable_timeseries.reshape(-1, 1)
                            scores_times = compute_attention_scores(col_data_for_attention, mode='col')
                            if scores_times.size == 0: continue
                            time_indices_in_col = roulette_selection(scores_times, k_times)
                            for t_idx in time_indices_in_col:
                                current_variable_timeseries = add_noise_neighborhood(current_variable_timeseries, t_idx, s, noise_std)
                            new_data[:, col_idx] = current_variable_timeseries
                    augmented_list.append(new_data)
        
        if not augmented_list: # If L=0 and n=0, augmented_list might be empty
            return pd.DataFrame()

        combined_df = pd.DataFrame()
        col_names = ["Time"] + [f"Var{i}" for i in range(1, N_cols)] if N_cols > 0 else ["Time"]
        if N_cols == 0: col_names = [] # Handle case with no columns at all

        for i_sample, data_sample in enumerate(augmented_list):
            if data_sample.size == 0 and not col_names: # Empty data and no columns
                df = pd.DataFrame()
            elif data_sample.size == 0 and col_names: # Empty data but columns defined (e.g. L=0)
                 df = pd.DataFrame(columns=col_names)
            else:
                df = pd.DataFrame(data_sample, columns=col_names)
            df["Sample"] = i_sample
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df
    # --- 占位符结束 ---

    main()