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
        if d_model <= 0: d_model = 1 
        if nhead <= 0: nhead = 1   
        if d_model % nhead != 0:
            nhead = 1 
        
        dim_feedforward = d_model * 4
        if dim_feedforward <= 0:
            dim_feedforward = 2048 
            if d_model > 0 : dim_feedforward = d_model * 4 
            else: dim_feedforward = 4 

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def _generate_positional_encoding(self, length, d_model):
        if d_model == 0: d_model = 1 
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term_denominator = d_model
        if d_model == 0 : div_term_denominator = 1 
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / div_term_denominator)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1 and pe[:, 1::2].size(1) > 0: 
            if div_term.size(0) > pe[:, 1::2].size(1) : # div_term for cos part might be shorter if d_model is odd
                 pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
            elif div_term.size(0) == pe[:,1::2].size(1) and div_term.size(0) > 0 : 
                 pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        if x.shape[2] == 0: 
            return x 
        seq_len, d_model_input = x.shape[1], x.shape[2]
        pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
        x = x + pe.unsqueeze(0)
        return self.encoder(x)

# --- 2. 计算注意力分数函数 ---
def compute_attention_scores(data_np, mode='row'):
    device = torch.device("cpu")
    d_model_transformer = 1 

    if not isinstance(data_np, np.ndarray) or data_np.size == 0:
        return np.array([])

    if mode == 'row':
        _data_np_reshaped = data_np.reshape(-1, 1)
    elif mode == 'col':
        _data_np_reshaped = data_np.reshape(-1, 1)
    else:
        raise ValueError("mode 必须是 'row' 或 'col'")
    
    if _data_np_reshaped.size == 0: 
        return np.array([])

    data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0) 

    if data_tensor.shape[1] == 0 or data_tensor.shape[2] == 0: 
         return np.array([])

    transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)
    with torch.no_grad():
        output = transformer(data_tensor) 
        attention_scores = output.squeeze(0).squeeze(-1).cpu().numpy() 
    return attention_scores

# --- 3. 主要的时间序列增强函数 (处理目标行不变) ---
#不包含元素数据，最后一行也扰动
'''
def generate_augmented_time_series_with_transformer(original_data, s=2, n=10, noise_std=0.3, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not isinstance(original_data, np.ndarray):
        print("错误：original_data 必须是一个 NumPy 数组。")
        return pd.DataFrame()

    if original_data.ndim < 2 or original_data.shape[0] < 1:
        print("错误：输入数据维度不足。")
        return pd.DataFrame()

    original_features = original_data  # 所有变量都可扰动（含目标）

    L_features, N_cols = original_features.shape

    def roulette_selection(scores, k):
        if not isinstance(scores, np.ndarray) or scores.size == 0:
            return np.array([], dtype=int)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        finite_scores = scores[np.isfinite(scores)]
        max_score = np.max(finite_scores) if finite_scores.size > 0 else -np.inf

        if np.all(np.isneginf(scores)) or max_score == -np.inf:
            probs = np.ones_like(scores) / len(scores)
        else:
            stable_scores = scores - max_score
            probs = np.exp(stable_scores)
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones_like(scores) / len(scores)

        k_actual = min(k, len(scores))
        try:
            return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)
        except ValueError:
            return np.random.choice(len(scores), size=k_actual, replace=False)

    def add_noise_neighborhood(data_array, center_idx, window_size):
        length = len(data_array)
        if length == 0:
            return data_array
        noisy_array = data_array.copy()
        start_idx = max(0, center_idx - (window_size - 1) // 2)
        end_idx = min(length, start_idx + window_size)
        if end_idx == length and (end_idx - start_idx) < window_size:
            start_idx = max(0, end_idx - window_size)

        orig_segment = noisy_array[start_idx:end_idx]
        scales = np.random.uniform(0.05, noise_std, size=orig_segment.shape)
        noise = np.random.normal(0, np.abs(orig_segment) * scales)
        noisy_array[start_idx:end_idx] += noise
        return noisy_array

    combined_df_list = []

    for sample_id in range(1, n + 1):  # 从 Sample 1 开始，不包括原始数据
        current_augmented_features = original_features.copy()

        if L_features > 0:
            k_vars = max(1, math.ceil(N_cols / 10.0))
            for row_idx in range(L_features):
                row_data = current_augmented_features[row_idx, :]
                scores_vars = compute_attention_scores(row_data, mode='row')
                var_indices = roulette_selection(scores_vars, k_vars)
                for col_idx in var_indices:
                    val = current_augmented_features[row_idx, col_idx]
                    scale = np.random.uniform(0.05, noise_std)
                    noise = np.random.normal(0, abs(val) * scale)
                    current_augmented_features[row_idx, col_idx] += noise

        if L_features > 0:
            k_times = max(1, math.ceil(L_features / 10.0))
            for col_idx in range(N_cols):
                col_series = current_augmented_features[:, col_idx].copy()
                scores_time = compute_attention_scores(col_series.reshape(-1, 1), mode='col')
                time_indices = roulette_selection(scores_time, k_times)
                for t_idx in time_indices:
                    col_series = add_noise_neighborhood(col_series, t_idx, s)
                current_augmented_features[:, col_idx] = col_series

        df = pd.DataFrame(current_augmented_features, columns=[f"InternalCol{j}" for j in range(N_cols)])
        df["Sample"] = sample_id
        combined_df_list.append(df)

    return pd.concat(combined_df_list, ignore_index=True) if combined_df_list else pd.DataFrame()
'''
#不包含原始数据,最后一行不扰动

def generate_augmented_time_series_with_transformer(original_data_with_target, s=2, n=50, noise_std=0.01, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not isinstance(original_data_with_target, np.ndarray):
        print("错误：original_data_with_target 必须是一个 NumPy 数组。")
        return pd.DataFrame()

    if original_data_with_target.ndim < 2 or original_data_with_target.shape[0] <= 1:
        print(f"警告：输入数据行数 ({original_data_with_target.shape[0]}) 不足（至少需要1行特征和1行目标）。")
        _L_full, _N_cols_full = original_data_with_target.shape if original_data_with_target.ndim >=2 else (0,0)
        if _L_full == 0 or _N_cols_full == 0: return pd.DataFrame()
        
        _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
        _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
        _df["Sample"] = 0
        return _df

    original_features = original_data_with_target 

    if original_features.shape[0] == 0:
        print("警告：没有特征行可供增强（只有目标行）。")
        _L_full, _N_cols_full = original_data_with_target.shape
        _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
        _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
        _df["Sample"] = 0
        return _df

    L_features, N_cols = original_features.shape

    def roulette_selection(scores, k):
        if not isinstance(scores, np.ndarray) or scores.size == 0: return np.array([], dtype=int)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        
        # 检查是否所有分数都是-inf
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0: # 如果没有有限分数（例如，全是-inf或空）
            max_score = -np.inf
        else:
            max_score = np.max(finite_scores)

        if np.all(np.isneginf(scores)) or max_score == -np.inf :
            probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
        else:
            stable_scores = scores - max_score 
            probs = np.exp(stable_scores)
        
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum) or len(scores) == 0:
            probs = np.ones_like(scores) / len(scores) if len(scores) > 0 else np.array([])
        else:
            probs /= probs_sum
        
        if probs.size == 0: return np.array([], dtype=int)
        probs = np.maximum(0, probs) 
        probs_sum_final = probs.sum()
        if probs_sum_final == 0 and len(scores) > 0 :
             probs = np.ones_like(scores) / len(scores)
        elif probs_sum_final > 0 : 
            probs /= probs_sum_final
        else: 
            return np.array([], dtype=int)

        k_actual = min(k, len(scores))
        if k_actual == 0: return np.array([], dtype=int)
        
        if np.any(np.isnan(probs)) or (probs.size > 0 and not np.isclose(np.sum(probs), 1.0, atol=1e-5)):
            probs = np.ones(len(scores)) / len(scores) if len(scores) > 0 else np.array([])
            if probs.size == 0: return np.array([], dtype=int)
        
        try:
            return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)
        except ValueError: 
            return np.random.choice(len(scores), size=k_actual, replace=False)


    def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
        length = len(data_array)
        if length == 0: return data_array
        noisy_array = data_array.copy()
        start_idx = max(0, center_idx - (window_size - 1) // 2)
        end_idx = min(length, start_idx + window_size)
        if end_idx == length and (end_idx - start_idx) < window_size : 
            start_idx = max(0, end_idx - window_size)
        
        actual_len_to_noise = end_idx - start_idx
        if actual_len_to_noise > 0:
            noise = np.random.normal(0, noise_std_val, actual_len_to_noise)
            noisy_array[start_idx:end_idx] += noise
        return noisy_array

    augmented_samples_full_data = []
    #是否增添原本原始数据
    #augmented_samples_full_data.append(original_data_with_target.copy()) # Sample 0

    for _ in range(n): 
        current_augmented_features = original_features.copy()

        num_variables_to_select_from = N_cols - 1 
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
        
        if L_features > 0 and N_cols > 1: 
            k_times = max(1, math.ceil(L_features / 10.0))
            for col_idx_in_features in range(1, N_cols): 
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
        
       # final_augmented_sample_with_target = np.vstack((current_augmented_features, target_row))
        final_augmented_sample_with_target = current_augmented_features
        augmented_samples_full_data.append(final_augmented_sample_with_target)

    _L_full_ignored, N_cols_full = original_data_with_target.shape
    internal_col_names = [f"InternalCol{j}" for j in range(N_cols_full)]
    
    combined_df_list = []
    for i_sample, data_sample_np in enumerate(augmented_samples_full_data):
        df = pd.DataFrame(data_sample_np, columns=internal_col_names)
        df["Sample"] = i_sample
        combined_df_list.append(df)
    #加上原始数据
    '''if not combined_df_list: return pd.DataFrame()
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    return combined_df'''
    return pd.concat(combined_df_list, ignore_index=True) if combined_df_list else pd.DataFrame()


#包含原始数据,最后一行扰动
'''def generate_augmented_time_series_with_transformer(original_data, s=2, n=10, noise_std=0.05, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if not isinstance(original_data, np.ndarray):
        print("错误：original_data 必须是一个 NumPy 数组。")
        return pd.DataFrame()

    if original_data.ndim < 2 or original_data.shape[0] < 1:
        print("错误：输入数据维度不足。")
        return pd.DataFrame()

    original_features = original_data  # ✅ 不再分离目标行

    L_features, N_cols = original_features.shape

    def roulette_selection(scores, k):
        if not isinstance(scores, np.ndarray) or scores.size == 0:
            return np.array([], dtype=int)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        finite_scores = scores[np.isfinite(scores)]
        max_score = np.max(finite_scores) if finite_scores.size > 0 else -np.inf

        if np.all(np.isneginf(scores)) or max_score == -np.inf:
            probs = np.ones_like(scores) / len(scores)
        else:
            stable_scores = scores - max_score
            probs = np.exp(stable_scores)
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones_like(scores) / len(scores)

        k_actual = min(k, len(scores))
        try:
            return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)
        except ValueError:
            return np.random.choice(len(scores), size=k_actual, replace=False)

    def add_noise_neighborhood(data_array, center_idx, window_size):
        length = len(data_array)
        if length == 0:
            return data_array
        noisy_array = data_array.copy()
        start_idx = max(0, center_idx - (window_size - 1) // 2)
        end_idx = min(length, start_idx + window_size)
        if end_idx == length and (end_idx - start_idx) < window_size:
            start_idx = max(0, end_idx - window_size)

        orig_segment = noisy_array[start_idx:end_idx]
        scales = np.random.uniform(0.05, 0.3, size=orig_segment.shape)
        noise = np.random.normal(0, np.abs(orig_segment) * scales)
        noisy_array[start_idx:end_idx] += noise
        return noisy_array

    augmented_samples_full_data = []
    augmented_samples_full_data.append(original_data.copy())  # 原始样本

    for _ in range(n):  # n 个增强样本
        current_augmented_features = original_features.copy()

        if L_features > 0:
            k_vars = max(1, math.ceil(N_cols / 10.0))
            for row_idx in range(L_features):
                row_data = current_augmented_features[row_idx, :]
                scores_vars = compute_attention_scores(row_data, mode='row')
                var_indices = roulette_selection(scores_vars, k_vars)
                for col_idx in var_indices:
                    val = current_augmented_features[row_idx, col_idx]
                    scale = np.random.uniform(0.05, noise_std)  # 变量扰动比例
                    noise = np.random.normal(0, abs(val) * scale)
                    current_augmented_features[row_idx, col_idx] += noise

        if L_features > 0:
            k_times = max(1, math.ceil(L_features / 10.0))
            for col_idx in range(N_cols):
                col_series = current_augmented_features[:, col_idx].copy()
                scores_time = compute_attention_scores(col_series.reshape(-1, 1), mode='col')
                time_indices = roulette_selection(scores_time, k_times)
                for t_idx in time_indices:
                    col_series = add_noise_neighborhood(col_series, t_idx, s)
                current_augmented_features[:, col_idx] = col_series

        final_augmented_sample = current_augmented_features
        augmented_samples_full_data.append(final_augmented_sample)

    internal_col_names = [f"InternalCol{j}" for j in range(N_cols)]
    combined_df_list = []
    for i_sample, data_sample_np in enumerate(augmented_samples_full_data):
        df = pd.DataFrame(data_sample_np, columns=internal_col_names)
        df["Sample"] = i_sample
        combined_df_list.append(df)

    return pd.concat(combined_df_list, ignore_index=True) if combined_df_list else pd.DataFrame()
'''# --- 4. 主函数 ---
def main():
    # --- 用户配置部分 ---
    input_filepath = r"falling_data/standardized_data.csv" # <--- 确保路径正确
    
    input_file_basename = os.path.basename(input_filepath)
    input_file_name_part, _ = os.path.splitext(input_file_basename)
    output_parent_dir = f"{input_file_name_part}_new" # 例如 "ETTh1_new"

    strength_levels_noise_std = np.linspace(0.01, 0.02, 50).tolist() 
    num_augmentations_per_strength_level = 1
    s_param = 5
    seed_param = 42
    # --- 配置结束 ---

    os.makedirs(output_parent_dir, exist_ok=True)

    if not os.path.isfile(input_filepath):
        print(f"错误：输入文件 '{input_filepath}' 未找到。请检查路径。")
        return

    print(f"正在为文件 '{input_filepath}' 生成10个不同强度等级的增强数据...")
    print(f"输出将保存在文件夹: '{output_parent_dir}'")
    print(f"强度等级 (noise_std): {[float(f'{val:.4f}') for val in strength_levels_noise_std]}") # 打印时格式化
        
    try:
        input_df_original_state = pd.read_csv(input_filepath)
        processing_df_template = input_df_original_state.copy()

        original_column_names = input_df_original_state.columns.tolist()
        if not input_df_original_state.empty:
            original_first_column_values = input_df_original_state.iloc[:, 0].copy()
        else:
            print(f"错误: 输入文件 {input_filepath} 为空。程序终止。")
            return
        
        print(f"  成功读取文件: {input_file_basename} (形状: {processing_df_template.shape})")

        if not processing_df_template.empty:
            processing_df_template.iloc[:, 0] = range(1, len(processing_df_template) + 1)
        else:
            print(f"  错误: processing_df_template 为空 {input_filepath}。程序终止。")
            return

        for col_idx in range(1, processing_df_template.shape[1]): 
            col_name = original_column_names[col_idx] 
            if str(processing_df_template[col_name].dtype) == 'object':
                processing_df_template[col_name] = pd.to_numeric(processing_df_template[col_name], errors='coerce')
                if processing_df_template[col_name].isnull().any():
                     print(f"  警告：文件 '{input_file_basename}' 的列 '{col_name}' 中存在无法转换为数字的值，已被转换为 NaN。")

        if processing_df_template.iloc[:, 1:].isnull().any().any(): 
            print(f"  警告：文件 '{input_file_basename}' 的数据列中发现 NaN。正在用 0 填充 NaN。")
            for col_idx_to_fill in range(1, processing_df_template.shape[1]):
                col_name_to_fill = original_column_names[col_idx_to_fill]
                if processing_df_template[col_name_to_fill].isnull().any():
                    processing_df_template[col_name_to_fill].fillna(0, inplace=True)
        
        base_input_data_for_augmentation = processing_df_template.to_numpy().astype(np.float32)
        print(f"  预处理完成的基础数据形状: {base_input_data_for_augmentation.shape}, 数据类型: {base_input_data_for_augmentation.dtype}")

        for strength_idx, current_noise_std in enumerate(strength_levels_noise_std):
            strength_level_id = strength_idx + 1 
            print(f"\n  --- 正在生成强度等级 {strength_level_id}/{len(strength_levels_noise_std)} (noise_std: {current_noise_std:.4f}) ---")
            
            df_all_samples_for_strength = generate_augmented_time_series_with_transformer(
                base_input_data_for_augmentation.copy(), 
                s=s_param, 
                n=num_augmentations_per_strength_level, 
                noise_std=current_noise_std, 
                seed=seed_param + strength_idx # 为每个强度等级使用不同的种子，以产生不同结果
            )

            if df_all_samples_for_strength.empty or "Sample" not in df_all_samples_for_strength.columns:
                print(f"    错误: 强度等级 {strength_level_id} 的数据增强失败或返回格式不正确。已跳过。")
                continue

            # vvvvvvvvvvvvvvvvvvvv 修改点 vvvvvvvvvvvvvvvvvvvv
            print(f"    为强度 {strength_level_id} 生成了包含原始数据和 {num_augmentations_per_strength_level} 个增强样本的 DataFrame。正在格式化...")
            
            final_dfs_to_combine_for_strength_level = []
            # 循环处理 df_all_samples_for_strength 中的每个样本 (Sample 0 是原始, 1-5 是增强)
            for sample_id_val in sorted(df_all_samples_for_strength["Sample"].unique()):
                one_sample_internal_df = df_all_samples_for_strength[df_all_samples_for_strength["Sample"] == sample_id_val].copy()
                
                # 分离数据部分 (不含 "Sample" 列)，以便重命名和替换值
                data_part_df = one_sample_internal_df.drop(columns=["Sample"], errors='ignore')
                
                # 使用原始列名创建新的DataFrame
                formatted_sample_df = pd.DataFrame(data_part_df.values, columns=original_column_names)
                
                # 恢复原始的第一列（时间列）的值
                if len(formatted_sample_df) == len(original_first_column_values):
                    formatted_sample_df.iloc[:, 0] = original_first_column_values.reset_index(drop=True).values
                else:
                    print(f"      警告: 文件 '{input_file_basename}' (强度 {strength_level_id}, 内部样本ID {sample_id_val}) 的长度 "
                          f"({len(formatted_sample_df)}) 与原始时间列长度 ({len(original_first_column_values)}) "
                          f"不匹配，无法完全恢复原始时间列值。")
                
                # 将 "Sample" ID 列添加回格式化后的DataFrame，以便区分
                formatted_sample_df["Sample"] = sample_id_val 
                
                final_dfs_to_combine_for_strength_level.append(formatted_sample_df)

            if final_dfs_to_combine_for_strength_level:
                df_to_save_for_strength = pd.concat(final_dfs_to_combine_for_strength_level, ignore_index=True)
                df_to_save_for_strength.drop(columns=['Sample'], inplace=True)
                # 构造输出文件名
                output_filename = f"{input_file_name_part}_strength_{strength_level_id}.csv"
                output_file_path = os.path.join(output_parent_dir, output_filename)
                try:
                    # 保存包含原始数据和当前强度下所有5个增强数据的组合文件
                    # "Sample" 列将被保留
                    
                    df_to_save_for_strength.to_csv(output_file_path, index=False)
                    print(f"    成功将强度 {strength_level_id} 的数据 (原始 + {num_augmentations_per_strength_level} 增强) 保存到: {output_file_path} (形状: {df_to_save_for_strength.shape})")
                except Exception as e:
                    print(f"      错误: 保存文件 '{output_file_path}' 时出错: {e}")
            else:
                print(f"    强度等级 {strength_level_id} 没有可供格式化和保存的数据。")
            # ^^^^^^^^^^^^^^^^^^^^ 修改点 ^^^^^^^^^^^^^^^^^^^^
    except FileNotFoundError:
        print(f"错误：主要输入文件未找到: {input_filepath}。")
    except Exception as e:
        print(f"处理文件 {input_filepath} 时发生意外错误: {e}")
        traceback.print_exc()

    print("\n----------------------------------------------------------------")
    print("所有强度等级处理完毕。")

# --- 5. 脚本执行入口 ---
if __name__ == "__main__":
    main()