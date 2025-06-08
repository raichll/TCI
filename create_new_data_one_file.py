import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import os
import traceback # 用于打印详细错误信息

# --- 1. Transformer 编码器定义 ---
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        if d_model <= 0: d_model = 1 # 基本保护，防止d_model为0或负数
        if nhead <= 0: nhead = 1   # 基本保护
        if d_model % nhead != 0:
            # print(f"警告: d_model ({d_model}) 不能被 nhead ({nhead}) 整除。将 nhead 调整为 1。")
            nhead = 1 # 如果不能整除，则强制nhead为1（d_model必须能被nhead整除）
        
        # 确保 dim_feedforward > 0
        dim_feedforward = d_model * 4
        if dim_feedforward <= 0:
            dim_feedforward = 2048 # TransformerEncoderLayer中的默认值，或至少是一个正值
            if d_model > 0 : dim_feedforward = d_model * 4 # 如果d_model调整后大于0，重新计算
            else: dim_feedforward = 4 # 最后的备用值

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def _generate_positional_encoding(self, length, d_model):
        if d_model == 0: d_model = 1 # 防止 d_model 为 0 导致除零错误
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term_denominator = d_model
        if d_model == 0 : div_term_denominator = 1 # 再次防止除零
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / div_term_denominator)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1 and pe[:, 1::2].size(1) > 0: # 确保cos部分有空间且div_term不为空/不越界
            if div_term.size(0) > pe[:, 1::2].size(1) :
                 pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
            elif div_term.size(0) == pe[:,1::2].size(1) and div_term.size(0) > 0 : # div_term不为空
                 pe[:, 1::2] = torch.cos(position * div_term)
            # else: div_term 对于 cos 部分来说太短或为空 (例如 d_model=1 时)
        return pe

    def forward(self, x):
        if x.shape[2] == 0: # d_model (特征维度) 为 0
            # print("警告: SimpleTransformerEncoder.forward 中的输入 d_model 为 0。")
            return x # 或者抛出错误，取决于期望行为
        seq_len, d_model_input = x.shape[1], x.shape[2]
        pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
        x = x + pe.unsqueeze(0)
        return self.encoder(x)

# --- 2. 计算注意力分数函数 ---
def compute_attention_scores(data_np, mode='row'):
    device = torch.device("cpu")
    d_model_transformer = 1 # 每个时间点/变量的特征维度被视为1

    if data_np.size == 0:
        # print(f"警告: compute_attention_scores 为 mode='{mode}' 接收到空 data_np。返回空分数。")
        return np.array([])

    if mode == 'row':
        # data_np 期望是1D (num_variables,) 或 2D (1, num_variables)
        # 需要 reshape 成 (num_variables, 1)
        _data_np_reshaped = data_np.reshape(-1, 1)
    elif mode == 'col':
        # data_np 期望是1D (num_timesteps,) 或 2D (num_timesteps, 1)
        # 需要 reshape 成 (num_timesteps, 1)
        _data_np_reshaped = data_np.reshape(-1, 1)
    else:
        raise ValueError("mode 必须是 'row' 或 'col'")
    
    if _data_np_reshaped.size == 0: # Reshape后仍为空
        return np.array([])

    data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0) # [1, seq_len, 1]

    if data_tensor.shape[1] == 0 or data_tensor.shape[2] == 0: # seq_len 或 d_model 为 0
         # print(f"警告: 为 Transformer 准备的 data_tensor 维度无效 (mode='{mode}')。返回空分数。")
         return np.array([])

    transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)
    with torch.no_grad():
        output = transformer(data_tensor) # output: [1, seq_len, d_model_transformer]
        attention_scores = output.squeeze(0).squeeze(-1).cpu().numpy() # [seq_len]
    return attention_scores

# --- 3. 主要的时间序列增强函数 ---
def generate_augmented_time_series_with_transformer(original_data_with_target, s=2, n=10, noise_std=0.01, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not isinstance(original_data_with_target, np.ndarray):
        print("错误：original_data_with_target 必须是一个 NumPy 数组。")
        return pd.DataFrame()

    if original_data_with_target.ndim < 2 or original_data_with_target.shape[0] <= 1:
        print(f"警告：输入数据行数 ({original_data_with_target.shape[0]}) 不足（至少需要1行特征和1行目标）。")
        # 按原样返回，并标记为样本0
        _L_full, _N_cols_full = original_data_with_target.shape if original_data_with_target.ndim >=2 else (0,0)
        if _L_full == 0 or _N_cols_full == 0: return pd.DataFrame()
        
        _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
        _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
        _df["Sample"] = 0
        return _df

    # 分离特征和目标行
    original_features = original_data_with_target[:-1, :]
    target_row = original_data_with_target[-1:, :] # 保持为2D数组以便后续 vstack

    if original_features.shape[0] == 0:
        print("警告：没有特征行可供增强（只有目标行）。")
        _L_full, _N_cols_full = original_data_with_target.shape
        _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
        _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
        _df["Sample"] = 0
        return _df

    L_features, N_cols = original_features.shape

    # --- 内部辅助函数定义 ---
    def roulette_selection(scores, k):
        if not isinstance(scores, np.ndarray) or scores.size == 0: return np.array([], dtype=int)
        
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        if np.all(np.isneginf(scores)): # 如果所有分数都是负无穷大
            probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
        else:
            stable_scores = scores - np.max(scores[np.isfinite(scores)], initial=-np.inf) # 处理全是-inf的情况
            probs = np.exp(stable_scores)
        
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum) or len(scores) == 0:
            probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
        else:
            probs /= probs_sum
        
        if probs.size == 0: return np.array([], dtype=int)
        probs = np.maximum(0, probs) # 确保概率非负
        probs_sum_final = probs.sum()
        if probs_sum_final == 0 and len(scores) > 0 :
             probs = np.ones_like(scores) / len(scores)
        elif probs_sum_final > 0 : # 重新归一化
            probs /= probs_sum_final
        else: # len(scores) == 0
            return np.array([], dtype=int)

        k_actual = min(k, len(scores))
        if k_actual == 0: return np.array([], dtype=int)
        
        # 最终检查NaN或总和问题
        if np.any(np.isnan(probs)) or (probs.size > 0 and not np.isclose(np.sum(probs), 1.0, atol=1e-5)):
            # print(f"  轮盘赌选择概率警告: {probs}。使用均匀分布。")
            probs = np.ones(len(scores)) / len(scores) if len(scores) > 0 else np.array([])
            if probs.size == 0: return np.array([], dtype=int)
        
        try:
            return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)
        except ValueError: # 如果概率仍然有问题
            return np.random.choice(len(scores), size=k_actual, replace=False) # 无p，均匀选择


    def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
        length = len(data_array)
        if length == 0: return data_array
        noisy_array = data_array.copy()
        start_idx = max(0, center_idx - (window_size - 1) // 2)
        end_idx = min(length, start_idx + window_size)
        # 如果窗口在末尾被裁剪，调整start_idx以保持窗口大小（如果可能）
        if end_idx == length and (end_idx - start_idx) < window_size : 
            start_idx = max(0, end_idx - window_size)
        
        actual_len_to_noise = end_idx - start_idx
        if actual_len_to_noise > 0:
            noise = np.random.normal(0, noise_std_val, actual_len_to_noise)
            noisy_array[start_idx:end_idx] += noise
        return noisy_array
    # --- 内部辅助函数定义结束 ---

    augmented_samples_full_data = []
    augmented_samples_full_data.append(original_data_with_target.copy()) # Sample 0: 原始数据

    for _ in range(n): # 生成 n 个新的增强样本
        current_augmented_features = original_features.copy()

        # --- 步骤 1: 变量扰动 (在 current_augmented_features 上操作) ---
        num_variables_to_select_from = N_cols - 1 # 假设第一列是时间索引，不参与变量选择
        if num_variables_to_select_from > 0 and L_features > 0:
            k_vars = max(1, math.ceil(num_variables_to_select_from / 10.0))
            for row_idx in range(L_features):
                row_variables_data_slice = current_augmented_features[row_idx, 1:]
                if row_variables_data_slice.size == 0: continue

                scores_vars = compute_attention_scores(row_variables_data_slice, mode='row')
                if scores_vars.size == 0: continue
                
                var_indices_in_slice = roulette_selection(scores_vars, k_vars)
                for var_idx_local in var_indices_in_slice:
                    actual_col_idx_in_features = var_idx_local + 1
                    current_augmented_features[row_idx, actual_col_idx_in_features] += np.random.normal(0, noise_std)
        
        # --- 步骤 2: 时间点扰动 (在 current_augmented_features 上操作) ---
        if L_features > 0 and N_cols > 1: # 必须有时间点和变量才能进行时间点扰动
            k_times = max(1, math.ceil(L_features / 10.0))
            for col_idx_in_features in range(1, N_cols): # 对每个变量列（跳过第0列的时间索引）
                variable_timeseries_features = current_augmented_features[:, col_idx_in_features].copy()
                if variable_timeseries_features.size == 0: continue

                col_data_for_attention = variable_timeseries_features.reshape(-1, 1)
                scores_times = compute_attention_scores(col_data_for_attention, mode='col')
                if scores_times.size == 0: continue

                time_indices_in_col = roulette_selection(scores_times, k_times)
                for t_idx in time_indices_in_col:
                    variable_timeseries_features = add_noise_neighborhood(
                        variable_timeseries_features, t_idx, s, noise_std
                    )
                current_augmented_features[:, col_idx_in_features] = variable_timeseries_features
        
        final_augmented_sample_with_target = np.vstack((current_augmented_features, target_row))
        augmented_samples_full_data.append(final_augmented_sample_with_target)

    # --- DataFrame 组装 ---
    _L_full_ignored, N_cols_full = original_data_with_target.shape
    internal_col_names = [f"InternalCol{j}" for j in range(N_cols_full)]
    
    combined_df_list = []
    for i_sample, data_sample_np in enumerate(augmented_samples_full_data):
        df = pd.DataFrame(data_sample_np, columns=internal_col_names)
        df["Sample"] = i_sample
        combined_df_list.append(df)
    
    if not combined_df_list:
        return pd.DataFrame() # 如果列表为空，返回空DataFrame
        
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    return combined_df

# --- 4. 主函数 ---
def main():
    # 用户配置部分
    source_csv_directory = "Time-Series-Library/dataset_refine"  # <--- 修改这里：你的原始CSV文件所在的文件夹
    output_main_dir = "dataset_new_"      # <--- 修改这里：新的输出文件夹
    num_augmentations_per_file = 5
    s_param = 3
    noise_std_param = 0.01
    seed_param = 42
    # --- 配置结束 ---

    os.makedirs(output_main_dir, exist_ok=True)

    if not os.path.isdir(source_csv_directory):
        print(f"错误：源文件夹 '{source_csv_directory}' 未找到。请创建该文件夹或检查路径。")
        return

    try:
        all_files_in_source_dir = os.listdir(source_csv_directory)
    except Exception as e:
        print(f"错误：无法读取源文件夹 '{source_csv_directory}' 中的文件列表: {e}")
        return
    
    csv_files_to_process = [f for f in all_files_in_source_dir if f.lower().endswith('.csv')]

    if not csv_files_to_process:
        print(f"在文件夹 '{source_csv_directory}' 中未找到 CSV 文件。")
        return

    print(f"将在文件夹 '{source_csv_directory}' 中处理以下 CSV 文件: {csv_files_to_process}")
    print(f"每个文件将生成 {num_augmentations_per_file} 个增强版本，并与原始数据合并保存到 '{output_main_dir}' 文件夹中。")

    for input_filename_short in csv_files_to_process:
        filepath = os.path.join(source_csv_directory, input_filename_short)
        print(f"\n----------------------------------------------------------------")
        print(f"正在处理文件: {filepath} ...")
        
        try:
            input_df_original_state = pd.read_csv(filepath)
            processing_df = input_df_original_state.copy()

            original_column_names = input_df_original_state.columns.tolist()
            if not input_df_original_state.empty:
                original_first_column_values = input_df_original_state.iloc[:, 0].copy()
            else:
                print(f"警告: 输入文件 {filepath} 为空。已跳过。")
                continue
            
            print(f"  成功读取文件: {input_filename_short} (形状: {processing_df.shape})")

            if not processing_df.empty:
                processing_df.iloc[:, 0] = range(1, len(processing_df) + 1)
            else:
                print(f"  错误: processing_df 为空 {filepath}。已跳过。")
                continue

            for col_idx in range(1, processing_df.shape[1]):
                col_name = original_column_names[col_idx]
                if str(processing_df[col_name].dtype) == 'object': # 更可靠的dtype检查
                    processing_df[col_name] = pd.to_numeric(processing_df[col_name], errors='coerce')
                    if processing_df[col_name].isnull().any():
                         print(f"  警告：文件 '{input_filename_short}' 的列 '{col_name}' 中存在无法转换为数字的值，已被转换为 NaN。")

            if processing_df.iloc[:, 1:].isnull().any().any():
                print(f"  警告：文件 '{input_filename_short}' 的数据列中发现 NaN。正在用 0 填充 NaN。")
                for col_idx_to_fill in range(1, processing_df.shape[1]):
                    col_name_to_fill = original_column_names[col_idx_to_fill]
                    if processing_df[col_name_to_fill].isnull().any(): # 检查是否真的有NaN需要填充
                        processing_df[col_name_to_fill].fillna(0, inplace=True)
            
            input_data = processing_df.to_numpy().astype(np.float32)
            print(f"  内部处理数据形状: {input_data.shape}, 数据类型: {input_data.dtype}")

            df_all_samples = generate_augmented_time_series_with_transformer(
                input_data, s=s_param, n=num_augmentations_per_file, 
                noise_std=noise_std_param, seed=seed_param
            )

            if df_all_samples.empty or "Sample" not in df_all_samples.columns:
                print(f"  错误: 文件 '{input_filename_short}' 的数据增强失败或返回格式不正确。已跳过。")
                continue

            print(f"  为 {input_filename_short} 生成了包含原始数据和 {num_augmentations_per_file} 个增强样本的 DataFrame。正在格式化...")
            
            final_dfs_to_combine = []
            for sample_id in sorted(df_all_samples["Sample"].unique()):
                one_sample_internal_df = df_all_samples[df_all_samples["Sample"] == sample_id].copy()
                data_part_df = one_sample_internal_df.drop(columns=["Sample"], errors='ignore')
                
                formatted_sample_df = pd.DataFrame(data_part_df.values, columns=original_column_names)
                
                if len(formatted_sample_df) == len(original_first_column_values):
                    # 确保 original_first_column_values 的索引不会导致问题
                    formatted_sample_df.iloc[:, 0] = original_first_column_values.reset_index(drop=True).values
                else:
                    print(f"    警告: 文件 '{input_filename_short}' (样本ID {sample_id}) 的长度 "
                          f"({len(formatted_sample_df)}) 与原始时间列长度 ({len(original_first_column_values)}) "
                          f"不匹配，无法完全恢复原始时间列值。")
                         
                formatted_sample_df["Sample"] = sample_id
                final_dfs_to_combine.append(formatted_sample_df)

            if final_dfs_to_combine:
                df_to_save_final = pd.concat(final_dfs_to_combine, ignore_index=True)
                if "Sample" in df_to_save_final.columns:
                    df_to_save_final = df_to_save_final.drop(columns=["Sample"])
                    print(f"  已从最终输出中移除 'Sample' 列。")
                else:
                    print(f"  警告: 在最终输出中未找到 'Sample' 列进行移除。")
                name_part, ext_part = os.path.splitext(input_filename_short)
                new_combined_filename = f"{name_part}_new{ext_part}"
                output_file_path = os.path.join(output_main_dir, new_combined_filename)

                try:
                    df_to_save_final.to_csv(output_file_path, index=False)
                    print(f"  成功将 '{input_filename_short}' 的组合增强数据保存到: {output_file_path} (形状: {df_to_save_final.shape})")
                except Exception as e:
                    print(f"    错误: 保存组合文件 '{output_file_path}' 时出错: {e}")
            else:
                print(f"  文件 '{input_filename_short}' 没有可供合并和保存的数据。")

        except FileNotFoundError:
            print(f"错误：输入文件未找到: {filepath}。已跳过。")
        except Exception as e:
            print(f"处理文件 {filepath} 时发生意外错误: {e}")
            traceback.print_exc()

    print("\n----------------------------------------------------------------")
    print("所有源文件处理完毕。")

# --- 5. 脚本执行入口 ---
if __name__ == "__main__":
    main()