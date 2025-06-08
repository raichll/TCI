import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import os # <--- 确保导入了 os 模块
import traceback # 用于打印详细错误信息

# ==============================================================================
# 确保 SimpleTransformerEncoder, compute_attention_scores, 
# generate_augmented_time_series_with_transformer 等函数已在此处或通过导入定义
# (此处省略这些函数的定义，假设它们已在脚本中正确定义)
# --- 你需要将下面的 main 函数替换掉你脚本中原来的 main 函数 ---
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # 用户配置部分：请修改下面的路径和参数
    # --------------------------------------------------------------------------
    # 1. 指定包含源 CSV 文件的输入文件夹路径
    #    例如: "dataset_to_augment" 或 "Time-Series-Library/dataset"
    #    你需要将 "dataset_refine" 替换为实际的文件夹名
    source_csv_directory = "Time-Series-Library/dataset_refine"  # <--- 修改这里：你的原始CSV文件所在的文件夹

    # 2. 指定保存所有新生成的增强CSV文件的主输出文件夹名称
    #output_main_dir = "dataset_augmented" # <--- 修改这里：新的输出文件夹
    output_main_dir = "dataset_new" #分开放
    #output_main_dir = "dataset_new_"  #合在一起
    # 3. 每个源文件要生成的增强样本数量
    num_augmentations_per_file = 5

    # 4. 数据增强参数 (如果需要，可以调整)
    s_param = 3
    noise_std_param = 0.01
    seed_param = 42 # 对每个文件使用相同的种子生成其对应的5个增强版本，以保证可复现性
    # --------------------------------------------------------------------------

    # 创建主输出文件夹（如果它还不存在）
    os.makedirs(output_main_dir, exist_ok=True)

    # 检查源文件夹是否存在
    if not os.path.isdir(source_csv_directory):
        print(f"错误：源文件夹 '{source_csv_directory}' 未找到。请创建该文件夹或检查路径。")
        return

    # 列出源文件夹中的所有文件
    try:
        all_files_in_source_dir = os.listdir(source_csv_directory)
    except Exception as e:
        print(f"错误：无法读取源文件夹 '{source_csv_directory}' 中的文件列表: {e}")
        return
    
    # 筛选出 CSV 文件
    csv_files_to_process = [f for f in all_files_in_source_dir if f.lower().endswith('.csv')]

    if not csv_files_to_process:
        print(f"在文件夹 '{source_csv_directory}' 中未找到 CSV 文件。")
        return

    print(f"将在文件夹 '{source_csv_directory}' 中处理以下 CSV 文件: {csv_files_to_process}")
    print(f"每个文件将生成 {num_augmentations_per_file} 个增强版本，并保存到 '{output_main_dir}' 文件夹中。")

    # 循环处理每个找到的 CSV 文件
    for input_filename_short in csv_files_to_process:
        filepath = os.path.join(source_csv_directory, input_filename_short)
        print(f"\n----------------------------------------------------------------")
        print(f"正在处理文件: {filepath} ...")
        
        try:
            # 1. 读取输入 CSV 文件
            input_df_original_state = pd.read_csv(filepath)
            
            # 复制一份用于内部处理，保留原始数据的副本
            processing_df = input_df_original_state.copy()

            # 存储原始列名和原始第一列（通常是时间列）的数据
            original_column_names = input_df_original_state.columns.tolist()
            if not input_df_original_state.empty:
                original_first_column_values = input_df_original_state.iloc[:, 0].copy()
            else:
                print(f"警告: 输入文件 {filepath} 为空。已跳过。")
                continue
            
            print(f"  成功读取文件: {input_filename_short} (形状: {processing_df.shape})")

            # 2. 预处理：将第一列（时间列）转换为序列 1...L 以供内部模型使用
            if not processing_df.empty:
                processing_df.iloc[:, 0] = range(1, len(processing_df) + 1)
            else:
                print(f"  错误: processing_df 为空 {filepath}。已跳过。") # 理论上不会到这里
                continue

            # 3. 清理数据列并转换为数字类型 (对 processing_df 操作)
            for col_idx in range(1, processing_df.shape[1]): # 从第二列开始
                col_name = original_column_names[col_idx] 
                if processing_df[col_name].dtype == 'object':
                    processing_df[col_name] = pd.to_numeric(processing_df[col_name], errors='coerce')
                    if processing_df[col_name].isnull().any():
                         print(f"  警告：文件 '{input_filename_short}' 的列 '{col_name}' 中存在无法转换为数字的值，已被转换为 NaN。")

            if processing_df.iloc[:, 1:].isnull().any().any(): 
                print(f"  警告：文件 '{input_filename_short}' 的数据列中发现 NaN。正在用 0 填充 NaN。")
                for col_idx_to_fill in range(1, processing_df.shape[1]):
                    col_name_to_fill = original_column_names[col_idx_to_fill]
                    if processing_df[col_name_to_fill].isnull().any():
                        processing_df[col_name_to_fill].fillna(0, inplace=True)
            
            input_data = processing_df.to_numpy().astype(np.float32) 
            print(f"  内部处理数据形状: {input_data.shape}, 数据类型: {input_data.dtype}")

            # 4. 生成指定数量的增强样本 (例如 n=5)
            print(f"  正在为 {input_filename_short} 生成 {num_augmentations_per_file} 个增强数据样本...")
            df_all_samples = generate_augmented_time_series_with_transformer(
                input_data, 
                s=s_param, 
                n=num_augmentations_per_file, 
                noise_std=noise_std_param, 
                seed=seed_param 
            )

            if df_all_samples.empty or "Sample" not in df_all_samples.columns:
                print(f"  错误: 文件 '{input_filename_short}' 的数据增强失败或返回格式不正确。已跳过。")
                continue

            # 5. 内部循环：提取、格式化并保存每个增强样本
            saved_count_for_this_file = 0
            for i_aug in range(1, num_augmentations_per_file + 1): # i_aug 将是 1, 2, ..., num_augmentations_per_file
                if i_aug not in df_all_samples["Sample"].unique():
                    print(f"  警告: 文件 '{input_filename_short}' 未找到期望的增强样本 ID {i_aug}。")
                    continue

                df_one_augmented_sample = df_all_samples[df_all_samples["Sample"] == i_aug].copy()

                if not df_one_augmented_sample.empty:
                    # 5a. 去掉 "Sample" 列
                    if "Sample" in df_one_augmented_sample.columns:
                        df_one_augmented_sample = df_one_augmented_sample.drop(columns=["Sample"])
                    
                    df_one_augmented_sample.reset_index(drop=True, inplace=True)
                    
                    # 准备原始第一列数据用于恢复 (确保索引对齐或使用 .values)
                    current_original_first_col_values = original_first_column_values.copy()
                    current_original_first_col_values.reset_index(drop=True, inplace=True)


                    # 5b. 恢复原始的第一列（时间列）的值
                    if len(df_one_augmented_sample) == len(current_original_first_col_values):
                        df_one_augmented_sample.iloc[:, 0] = current_original_first_col_values.values
                    else:
                        print(f"    警告: '{input_filename_short}' (增强样本 {i_aug}) 的长度 ({len(df_one_augmented_sample)}) "
                              f"与原始时间列长度 ({len(current_original_first_col_values)}) 不匹配，无法完全恢复原始时间列值。")

                    # 5c. 设置为原始的列名
                    if len(df_one_augmented_sample.columns) == len(original_column_names):
                        df_one_augmented_sample.columns = original_column_names
                    else:
                        print(f"    警告: '{input_filename_short}' (增强样本 {i_aug}) 的列数 ({len(df_one_augmented_sample.columns)}) "
                              f"与原始列数 ({len(original_column_names)}) 不匹配，无法完全恢复原始列名。")
                    
                    # 构造此增强样本的输出文件名
                    name_part, ext_part = os.path.splitext(input_filename_short)
                    new_aug_filename = f"{name_part}_new_{i_aug}{ext_part}" # 例如 ETTh1_aug_1.csv
                    output_file_path = os.path.join(output_main_dir, new_aug_filename)

                    try:
                        df_one_augmented_sample.to_csv(output_file_path, index=False)
                        saved_count_for_this_file += 1
                    except Exception as e:
                        print(f"    错误: 保存文件 '{output_file_path}' 时出错: {e}")
                else:
                    print(f"  错误: 文件 '{input_filename_short}' 提取的增强样本 (Sample ID {i_aug}) 为空。")
            
            print(f"  文件 '{input_filename_short}' 处理完成，共保存了 {saved_count_for_this_file} 个增强文件。")

        except FileNotFoundError: 
            print(f"错误：输入文件未找到: {filepath}。已跳过。")
        except Exception as e:
            print(f"处理文件 {filepath} 时发生意外错误: {e}")
            traceback.print_exc() # 打印详细的错误堆栈信息

    print("\n----------------------------------------------------------------")
    print("所有源文件处理完毕。")

if __name__ == "__main__":
    # ==============================================================================
    # 确保你的脚本中在此之前已经定义了所有必要的函数和类，例如：
    # - class SimpleTransformerEncoder(nn.Module): ... (包括 _generate_positional_encoding)
    # - def compute_attention_scores(data_np, mode='row'): ...
    # - def generate_augmented_time_series_with_transformer(...): ...
    #   (其中可能包含 roulette_selection 和 add_noise_neighborhood 的定义或调用)
    # 你需要从之前的代码中复制这些函数的定义到这个脚本的全局作用域，
    # 或者确保它们是通过 import 导入的。为了方便，我这里省略了这些函数的具体实现，
    # 但你必须使用你项目中完整的、正确的函数定义。
    # ==============================================================================
    
    # --- 示例：假设这些函数已定义（你需要用你实际的函数定义） ---
    class SimpleTransformerEncoder(nn.Module): 
        def __init__(self, d_model, nhead): 
            super().__init__() 
            if d_model <= 0: d_model = 1 
            if nhead <=0: nhead = 1 
            if d_model % nhead != 0: 
                nhead = 1
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=max(d_model*4, 1)) # Ensure dim_feedforward > 0
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        def _generate_positional_encoding(self, length, d_model):
            if d_model == 0: d_model = 1
            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1 and pe[:, 1::2].size(1) > 0:
                 if div_term.size(0) > pe[:, 1::2].size(1) : # Ensure div_term is not shorter than slice
                     pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
                 elif div_term.size(0) == pe[:,1::2].size(1):
                      pe[:, 1::2] = torch.cos(position * div_term)
                 # else: div_term is shorter, might be an issue if d_model is odd.
                 # The arange step of 2 for div_term means its length is ceil(d_model / 2.0).
                 # pe[:, 1::2] slice length is floor(d_model / 2.0). So div_term should usually be long enough or same length.
            return pe
        def forward(self, x): 
            if x.shape[2] == 0: return x 
            seq_len, d_model_input = x.shape[1], x.shape[2]
            pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
            x = x + pe.unsqueeze(0)
            return self.encoder(x) 

    def compute_attention_scores(data_np, mode='row'): 
        device = torch.device("cpu")
        d_model_transformer = 1
        if data_np.size == 0: return np.array([])
        if mode == 'row': _data_np_reshaped = data_np.reshape(-1, 1)
        elif mode == 'col': _data_np_reshaped = data_np 
        else: raise ValueError("mode must be 'row' or 'col'")
        data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0)
        if data_tensor.shape[2] == 0: return np.array([])
        transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)
        with torch.no_grad(): output = transformer(data_tensor)
        return output.squeeze(0).squeeze(-1).cpu().numpy()
    #生成dataset——augment，包括目标变量也改变

    def generate_augmented_time_series_with_transformer(original_data, s, n, noise_std, seed):
        if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
        if original_data.ndim < 2 or original_data.shape[0] == 0 or original_data.shape[1] == 0: return pd.DataFrame()
        L, N_cols = original_data.shape
        
        augmented_list = []
        augmented_list.append(original_data.copy()) 
        for i_aug_internal in range(n): # Generate n augmented samples
            new_data = original_data.copy() # Start from original for each augmentation
            # --- 执行实际的增强逻辑 (此处为简化占位符) ---
            # 示例：对除第一列外的所有列添加噪声
            noise = np.random.normal(0, noise_std, size=(L, N_cols -1 )).astype(np.float32)
            if N_cols > 1:
                new_data[:, 1:] += noise
            # --- 实际增强逻辑结束 ---
            augmented_list.append(new_data)

        internal_col_names = [f"InternalCol{j}" for j in range(N_cols)] # 内部 DataFrame 列名
        combined_df = pd.DataFrame()
        for i_sample, data_sample_np in enumerate(augmented_list): # i_sample: 0 for original, 1..n for augmented
            df = pd.DataFrame(data_sample_np, columns=internal_col_names)
            df["Sample"] = i_sample 
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df
    '''# --- 占位符结束 ---
    #生成dataset——new，不改变目标变量
    def generate_augmented_time_series_with_transformer(original_data_with_target, s=2, n=10, noise_std=0.01, seed=42):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if not isinstance(original_data_with_target, np.ndarray):
            print("错误：original_data_with_target 必须是一个 NumPy 数组。")
            return pd.DataFrame()

        if original_data_with_target.ndim < 2 or original_data_with_target.shape[0] <= 1:
            print(f"警告：输入数据行数 ({original_data_with_target.shape[0]}) 不足（至少需要1行特征和1行目标）。将返回包含原始数据的DataFrame。")
            # 如果行数不足以分离特征和目标，则按原样返回，并标记为样本0
            _L_full, _N_cols_full = original_data_with_target.shape if original_data_with_target.ndim >=2 else (0,0)
            if _L_full == 0 or _N_cols_full == 0: return pd.DataFrame() # 空数据返回空DataFrame
            
            _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
            _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
            _df["Sample"] = 0
            return _df

        # 分离特征和目标行
        original_features = original_data_with_target[:-1, :]
        target_row = original_data_with_target[-1:, :] # 保持为2D数组以便后续 vstack

        if original_features.shape[0] == 0:
            print("警告：没有特征行可供增强（只有目标行）。将返回包含原始数据的DataFrame。")
            _L_full, _N_cols_full = original_data_with_target.shape
            _internal_col_names = [f"InternalCol{j}" for j in range(_N_cols_full)]
            _df = pd.DataFrame(original_data_with_target, columns=_internal_col_names)
            _df["Sample"] = 0
            return _df

        L_features, N_cols = original_features.shape # L_features 是需要增强的时间序列长度

        # --- 内部辅助函数定义 (roulette_selection, add_noise_neighborhood) ---
        # --- 请确保这些函数已在此处定义，或者在全局作用域且此函数可以访问 ---
        # --- (为简洁起见，此处省略它们的具体实现，假设它们与你之前的版本相同) ---
        def roulette_selection(scores, k):
            # (你的 roulette_selection 实现)
            if scores.size == 0: return np.array([], dtype=int) 
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
            try:
                return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)
            except ValueError as e: # Catches issues like "probabilities do not sum to 1"
                # print(f"  Roulette selection probability error: {e}. Using uniform distribution.")
                probs = np.ones(len(scores)) / len(scores) if len(scores) > 0 else np.array([])
                if probs.size == 0: return np.array([], dtype=int)
                return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)


        def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
            # (你的 add_noise_neighborhood 实现)
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
        # --- 内部辅助函数定义结束 ---

        augmented_samples_full_data = []
        
        # 将原始数据（特征+目标）作为第一个样本 (Sample 0)
        augmented_samples_full_data.append(original_data_with_target.copy())

        # 生成 n 个新的增强样本
        for _ in range(n): 
            # 总是从原始特征开始进行当次增强
            current_augmented_features = original_features.copy()

            # --- 步骤 1: 变量扰动 (在 current_augmented_features 上操作) ---
            # 假设第一列是时间索引（在特征内部），不参与变量选择
            num_variables_to_select_from = N_cols - 1 
            if num_variables_to_select_from > 0:
                k_vars = max(1, math.ceil(num_variables_to_select_from / 10.0))
                for row_idx in range(L_features):
                    # row_variables_data 应为当前行中可供选择的变量值 (即排除第一列的时间索引)
                    row_variables_data_slice = current_augmented_features[row_idx, 1:] 
                    if row_variables_data_slice.size == 0: continue

                    scores_vars = compute_attention_scores(row_variables_data_slice, mode='row')
                    if scores_vars.size == 0: continue
                    
                    var_indices_in_slice = roulette_selection(scores_vars, k_vars)
                    
                    for var_idx_local in var_indices_in_slice:
                        # 将基于切片的索引转换回 current_augmented_features 的实际列索引
                        actual_col_idx_in_features = var_idx_local + 1 
                        current_augmented_features[row_idx, actual_col_idx_in_features] += np.random.normal(0, noise_std)
            
            # --- 步骤 2: 时间点扰动 (在 current_augmented_features 上操作) ---
            if L_features > 0: # 必须有时间点才能进行时间点扰动
                k_times = max(1, math.ceil(L_features / 10.0))
                # 对每个变量列（同样，假设特征的第一列是时间索引，不扰动它本身，而是扰动其他变量列）
                for col_idx_in_features in range(1, N_cols): 
                    variable_timeseries_features = current_augmented_features[:, col_idx_in_features].copy()
                    if variable_timeseries_features.size == 0: continue

                    col_data_for_attention = variable_timeseries_features.reshape(-1, 1)
                    scores_times = compute_attention_scores(col_data_for_attention, mode='col')
                    if scores_times.size == 0: continue

                    time_indices_in_col = roulette_selection(scores_times, k_times) # 这些是 L_features 内的索引
                    
                    for t_idx in time_indices_in_col:
                        variable_timeseries_features = add_noise_neighborhood(
                            variable_timeseries_features, t_idx, s, noise_std
                        )
                    current_augmented_features[:, col_idx_in_features] = variable_timeseries_features
            
            # 将增强后的特征与原始目标行合并
            final_augmented_sample_with_target = np.vstack((current_augmented_features, target_row))
            augmented_samples_full_data.append(final_augmented_sample_with_target)

        # --- DataFrame 组装 ---
        # 使用原始完整数据（包含目标行）的列数来确定内部列名
        _L_full_ignored, N_cols_full = original_data_with_target.shape 
        internal_col_names = [f"InternalCol{j}" for j in range(N_cols_full)]
        
        combined_df = pd.DataFrame()
        for i_sample, data_sample_np in enumerate(augmented_samples_full_data):
            # i_sample: 0 代表原始数据, 1..n 代表增强数据
            df = pd.DataFrame(data_sample_np, columns=internal_col_names)
            df["Sample"] = i_sample 
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        return combined_df
    '''
    main()