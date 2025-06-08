import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

# Transformer 编码器定义
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        seq_len, d_model_input = x.shape[1], x.shape[2]
        pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
        x = x + pe.unsqueeze(0)
        return self.encoder(x)

    def _generate_positional_encoding(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0 or d_model > 1: # Ensure index for cosine part is valid
            # Only assign to cos if there's a place for it (d_model > 1 for 1::2 to be valid, or if d_model is even, arange(0,d_model,2) might not cover last element if d_model is odd for div_term)
            # More robust: check if the slice pe[:, 1::2] is non-empty
            if pe[:, 1::2].size(1) > 0: # Check if the second dimension of the slice is greater than 0
                 pe[:, 1::2] = torch.cos(position * div_term)
        return pe

# 计算注意力分数
def compute_attention_scores(data_np, mode='row'):
    device = torch.device("cpu")
    
    d_model_transformer = 1 # Feature dimension is 1 (the value itself)

    if mode == 'row':
        # data_np is row_variables_data (e.g., shape (N-1,) where N-1 is num_variables)
        # We need to transform it to [1, num_variables, 1] for the Transformer.
        # 1. Reshape data_np to [num_variables, 1]
        _data_np_reshaped = data_np.reshape(-1, 1)
        # 2. Convert to tensor and add batch dimension
        data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0)

    elif mode == 'col':
        # data_np is col_data_for_attention (e.g., shape (L, 1) where L is num_time_points)
        # We need to transform it to [1, num_time_points, 1] for the Transformer.
        data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("mode must be 'row' or 'col'")

    transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)

    with torch.no_grad():
        output = transformer(data_tensor) # Output shape: [1, seq_len, d_model_transformer]
        attention_scores = output.squeeze(0).squeeze(-1).cpu().numpy()
    return attention_scores


# 主增强函数
def generate_augmented_time_series_with_transformer(original_data, s=2, n=10, noise_std=0.01, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    L, N_cols = original_data.shape # L = time points, N_cols = total columns

    def roulette_selection(scores, k):
        # Ensure scores are valid numbers before exponentiating
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf) # Replace NaN with -inf
        
        stable_scores = scores - np.max(scores) # Subtract max for numerical stability
        probs = np.exp(stable_scores)
        
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
            probs = np.ones_like(scores) / len(scores) # Uniform probability if sum is invalid
        else:
            probs /= probs_sum
        
        # Ensure probabilities are normalized and non-negative
        probs = np.maximum(0, probs) # clamp negative probabilities that might arise from extreme floats
        probs_sum_final = probs.sum()
        if probs_sum_final == 0 : # if all are zero after clamping
             probs = np.ones_like(scores) / len(scores)
        else:
            probs /= probs_sum_final # re-normalize

        k_actual = min(k, len(scores)) # Ensure k is not greater than the number of available items
        if k_actual == 0 :
            return np.array([], dtype=int)

        return np.random.choice(len(scores), size=k_actual, replace=False, p=probs)

    def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
        length = len(data_array)
        noisy_array = data_array.copy()
        
        start_idx = max(0, center_idx - (window_size - 1) // 2)
        end_idx = min(length, start_idx + window_size)
        if end_idx == length: # Adjust start_idx if window is clipped at the end
            start_idx = max(0, end_idx - window_size)

        actual_len_to_noise = end_idx - start_idx
        if actual_len_to_noise > 0:
            noise = np.random.normal(0, noise_std_val, actual_len_to_noise)
            noisy_array[start_idx:end_idx] += noise
        return noisy_array

    augmented_list = []

    for i in range(n + 1):
        if i == 0:
            augmented_list.append(original_data.copy())
        else:
            new_data = augmented_list[0].copy()

            # Step 1: 每行选择 ceil((N_cols-1)/10) 个变量
            # N_cols-1 because the first column is 'Time'
            num_actual_variables = N_cols - 1
            if num_actual_variables <= 0: # Should not happen if N_cols > 1
                k_vars = 0
            else:
                k_vars = max(1, math.ceil(num_actual_variables / 10.0))

            if k_vars > 0:
                for row_idx in range(L):
                    row_variables_data = new_data[row_idx, 1:] # Shape (N_cols-1,)
                    if row_variables_data.size == 0: continue # Skip if no variables

                    scores_vars = compute_attention_scores(row_variables_data, mode='row')
                    var_indices_in_row_data = roulette_selection(scores_vars, k_vars)
                    
                    for var_idx_local in var_indices_in_row_data:
                        actual_col_idx = var_idx_local + 1
                        new_data[row_idx, actual_col_idx] += np.random.normal(0, noise_std)

            # Step 2: 每列选择 ceil(L/10) 个时间点
            k_times = max(1, math.ceil(L / 10.0))
            if k_times > 0 and L > 0: # Ensure there are time points to select
                for col_idx in range(1, N_cols):
                    current_variable_timeseries = new_data[:, col_idx].copy()
                    col_data_for_attention = current_variable_timeseries.reshape(-1, 1)
                    
                    scores_times = compute_attention_scores(col_data_for_attention, mode='col')
                    time_indices_in_col = roulette_selection(scores_times, k_times)
                    
                    for t_idx in time_indices_in_col:
                        current_variable_timeseries = add_noise_neighborhood(current_variable_timeseries, t_idx, s, noise_std)
                    
                    new_data[:, col_idx] = current_variable_timeseries
            
            augmented_list.append(new_data)

    combined_df = pd.DataFrame()
    col_names = ["Time"] + [f"Var{i}" for i in range(1, N_cols)]
    for i_sample, data_sample in enumerate(augmented_list):
        df = pd.DataFrame(data_sample, columns=col_names)
        df["Sample"] = i_sample
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

# 主函数
def main():
    filepath = r"data\standardized_data.xlsx" # Use raw string for Windows paths
    filepath=r"Time-Series-Library\dataset\\ETT\\ETTh1.csv" # Fill NaN values with 0

    try:
        input_df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please check the path.")
        print("Creating a dummy dataframe for demonstration purposes.")
        L_dummy, N_dummy_vars = 50, 4 # 50 time points, 4 variables
        N_dummy_cols = N_dummy_vars + 1 # Plus Time column
        data_dummy = np.random.rand(L_dummy, N_dummy_cols)
        data_dummy[:,0] = np.arange(1, L_dummy + 1) # Time column
        input_df = pd.DataFrame(data_dummy, columns=["Time"] + [f"Var{i}" for i in range(1,N_dummy_vars+1)])
    
    # Regarding the FutureWarning:
    # If the first column from Excel is datetime and you want to replace it with sequence numbers.
    # This line does it, but pandas warns about dtype change.
    # To be more explicit and potentially avoid the warning in future pandas versions,
    # you could cast to object first, or ensure it's not datetime if it's not meant to be.
    # For now, the functionality should be okay for creating input_data.
    # Example: input_df.iloc[:, 0] = pd.Series(range(1, len(input_df) + 1), index=input_df.index, dtype=float) # or int
    try:
        # Check if the first column is already numeric, if not, then assign range
        if not pd.api.types.is_numeric_dtype(input_df.iloc[:, 0]):
            print(f"Warning: First column dtype is {input_df.iloc[:, 0].dtype}, converting to integer sequence.")
            input_df.iloc[:, 0] = range(1, len(input_df) + 1)
        elif input_df.iloc[:,0].isnull().any(): # Or if it's numeric but contains NaNs that would break range logic
             print(f"Warning: First column is numeric but contains NaNs, converting to integer sequence.")
             input_df.iloc[:, 0] = range(1, len(input_df) + 1)
        else: # If it's already numeric and fine, ensure it's 1-based for consistency if that's assumed
            # This part depends on whether you expect it to be 1...L or if original numbering is fine
            # For safety, let's ensure it's the 1 to L sequence if that's the core assumption
            # If your original "Time" column is already 1,2,3...L then this assignment is redundant but harmless
            input_df.iloc[:, 0] = range(1, len(input_df) + 1)

    except Exception as e:
        print(f"Error processing the first column of the DataFrame: {e}")
        print("Falling back to simple assignment for the first column.")
        input_df.iloc[:, 0] = range(1, len(input_df) + 1)


    input_data = input_df.to_numpy()
    
    print("Original data shape:", input_data.shape)

    df_final = generate_augmented_time_series_with_transformer(input_data, s=3, n=5, noise_std=0.01, seed=42)
    
    output_filename = "augmented_time_series_transformer_full.xlsx"
    df_final.to_excel(output_filename, index=False)
    print(f"Augmented data saved to {output_filename}")
    print("Shape of augmented data_frame:", df_final.shape)
    print("Number of unique samples:", df_final["Sample"].nunique())

if __name__ == "__main__":
    main()