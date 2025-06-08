import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

# Transformer 编码器定义
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # Ensure d_model is divisible by nhead for MultiheadAttention
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4) # Added dim_feedforward
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len, d_model_input = x.shape[1], x.shape[2]
        # The d_model for positional encoding must match the d_model of the input tensor x
        pe = self._generate_positional_encoding(seq_len, d_model_input).to(x.device)
        x = x + pe.unsqueeze(0) # Add positional encoding (broadcasts if batch_size > 1)
        return self.encoder(x)

    def _generate_positional_encoding(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1: # Avoid error if d_model is 1 and 1::2 is out of bounds
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

# 计算注意力分数
def compute_attention_scores(data_np, mode='row'):
    device = torch.device("cpu") # Or "cuda" if available and desired
    
    # The self-attention mechanism looks for relationships *within a sequence*.
    # d_model is the feature dimension of each element in the sequence.
    # nhead must be such that d_model is divisible by nhead. Here we use nhead=1.
    # d_model should consistently be the feature dimension of an item in the sequence.
    # We assume each item (variable or time-point) has a single feature: its value. So, d_model=1.

    if mode == 'row':
        # data_np is expected to be [num_variables] (e.g., from row_data = new_data[row, 1:])
        # We treat variables as the sequence. Shape for transformer: [batch, seq_len, features]
        # seq_len = num_variables, features (d_model) = 1
        # Input data_np from caller: row_data.reshape(1, -1), e.g. (1, N-1)
        # We need to transform it to [1, N-1, 1]
        data_tensor = torch.tensor(data_np.T, dtype=torch.float32).unsqueeze(0) # (1, N-1) -> (N-1, 1) -> (1, N-1, 1)
        if data_tensor.shape[2] != 1: # Should be (num_variables, 1) before unsqueeze
             # If original data_np was (num_variables,) -> .reshape(-1,1) first
             _data_np_reshaped = data_np.reshape(-1,1) # (N-1,) -> (N-1,1)
             data_tensor = torch.tensor(_data_np_reshaped, dtype=torch.float32).unsqueeze(0) # (N-1,1) -> (1, N-1, 1)

    elif mode == 'col':
        # data_np is expected to be [num_time_points, 1] (e.g., from col_data.reshape(-1, 1))
        # We treat time points as the sequence. Shape for transformer: [batch, seq_len, features]
        # seq_len = num_time_points, features (d_model) = 1
        data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0) # (L, 1) -> (1, L, 1)
    else:
        raise ValueError("mode must be 'row' or 'col'")

    d_model_transformer = 1 # Feature dimension is 1 (the value itself)
    transformer = SimpleTransformerEncoder(d_model=d_model_transformer, nhead=1).to(device)

    with torch.no_grad():
        output = transformer(data_tensor) # Output shape: [1, seq_len, d_model_transformer]
        # We want one score per item in the sequence.
        # Squeeze out batch dim (0) and feature dim (2, or -1)
        attention_scores = output.squeeze(0).squeeze(-1).cpu().numpy()
    return attention_scores


# 主增强函数
def generate_augmented_time_series_with_transformer(original_data, s=2, n=10, noise_std=0.01, seed=42):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    L, N = original_data.shape # L = time points, N = variables (including Time column)

    def roulette_selection(scores, k):
        # Ensure scores are not all -inf (e.g. if all inputs to exp are very small)
        if np.all(np.isneginf(scores - np.max(scores))):
             probs = np.ones_like(scores) / len(scores) # Uniform probability
        else:
            probs = np.exp(scores - np.max(scores)) # Subtract max for numerical stability
            probs_sum = probs.sum()
            if probs_sum == 0: # Handle case where all exp scores are zero
                probs = np.ones_like(scores) / len(scores)
            else:
                probs /= probs_sum
        
        # Ensure k is not greater than the number of available items
        k = min(k, len(scores))
        return np.random.choice(len(scores), size=k, replace=False, p=probs)

    def add_noise_neighborhood(data_array, center_idx, window_size, noise_std_val):
        # data_array is a 1D numpy array
        length = len(data_array)
        noisy_array = data_array.copy() # Work on a copy
        
        # Calculate bounds for the neighborhood
        # window_size s means s//2 to the left, s//2 to the right (or s-1 total points if s is odd)
        # If s=2, neighborhood is center_idx and center_idx+1 (or center_idx-1 and center_idx)
        # Let's define s as the total number of points in the window including the center.
        # If s=1, only center_idx. If s=2, center_idx and one neighbor. Let's assume s is total width.
        half_s = window_size // 2
        left = max(0, center_idx - half_s)
        # To make the window of size 's', if s is odd, right is center_idx + half_s.
        # If s is even, it's asymmetric or needs clarification.
        # Original code: right = min(center_idx + s // 2 + 1, L)
        # This means for s=2, left=center-1, right=center+1+1=center+2. Neighborhood [center-1, center, center+1]
        # If s=2, it implies a neighborhood of 2 or 3 points.
        # Let's assume s is the desired number of points to affect.
        # If s=1 (only center), noise is added to 1 point.
        # If s=2, noise to center and one neighbor.
        # The current right = min(center_idx + s // 2 + 1, L) makes the window size s+1 if s is even, and s if s is odd, around center_idx
        # For simplicity, let's define `s` as the number of points to perturb *around* center_idx (s/2 left, s/2 right)
        
        # Using original definition:
        # left = max(center_idx - s // 2, 0)
        # right = min(center_idx + s // 2 + 1, length) # +1 because slice upper bound is exclusive

        # A clearer way: affect 's' points centered at 'center_idx' if possible
        start_idx = max(0, center_idx - (s - 1) // 2)
        end_idx = min(length, start_idx + s)
        # Adjust start_idx if end_idx hit the boundary early
        if end_idx == length:
            start_idx = max(0, end_idx - s)

        actual_len_to_noise = end_idx - start_idx
        if actual_len_to_noise > 0:
            noise = np.random.normal(0, noise_std_val, actual_len_to_noise)
            noisy_array[start_idx:end_idx] += noise
        return noisy_array

    augmented_list = []

    for i in range(n + 1): # n+1 to include the original
        if i == 0:
            augmented_list.append(original_data.copy())
        else:
            new_data = augmented_list[0].copy() # Start from a fresh copy of original for each new sample

            # Step 1: 每行选择 ceil((N-1)/10) 个变量 (Variable perturbation per time point)
            # N includes the 'Time' column, so N-1 actual variables.
            k_vars = max(1, math.ceil((N - 1) / 10.0))
            for row_idx in range(L): # For each time point
                # Data for variables at this time point (excluding 'Time' column at index 0)
                row_variables_data = new_data[row_idx, 1:] # Shape (N-1,)
                
                # Compute attention scores over these N-1 variables
                # Input to compute_attention_scores should be (num_variables, 1) for mode='row'
                # or handle the (N-1,) shape inside compute_attention_scores
                # Current compute_attention_scores expects (1, num_variables) for mode='row' then transposes
                # Let's pass it as (num_variables,)
                scores_vars = compute_attention_scores(row_variables_data, mode='row') # Pass (N-1,)
                
                var_indices_in_row_data = roulette_selection(scores_vars, k_vars) # Indices relative to row_variables_data
                
                for var_idx_local in var_indices_in_row_data:
                    # Convert local index (0 to N-2) back to column index in new_data (1 to N-1)
                    actual_col_idx = var_idx_local + 1
                    new_data[row_idx, actual_col_idx] += np.random.normal(0, noise_std)

            # Step 2: 每列选择 ceil(L/10) 个时间点 (Time-point perturbation per variable)
            k_times = max(1, math.ceil(L / 10.0))
            for col_idx in range(1, N): # For each variable column (skip 'Time' column)
                
                # Get the current time series for this variable
                # This data will be modified iteratively by add_noise_neighborhood
                current_variable_timeseries = new_data[:, col_idx].copy() # Shape (L,)
                
                # For computing attention scores, use this current time series
                # Input to compute_attention_scores needs to be [L, 1] for mode='col'
                col_data_for_attention = current_variable_timeseries.reshape(-1, 1)
                scores_times = compute_attention_scores(col_data_for_attention, mode='col')
                
                time_indices_in_col = roulette_selection(scores_times, k_times) # Indices relative to the L time points
                
                # Apply noise to 'current_variable_timeseries' based on selected time_indices
                # Modifications should accumulate within this loop for the current column
                for t_idx in time_indices_in_col:
                    current_variable_timeseries = add_noise_neighborhood(current_variable_timeseries, t_idx, s, noise_std)
                
                # Assign the fully noised column back to new_data
                new_data[:, col_idx] = current_variable_timeseries
            
            augmented_list.append(new_data)

    # 汇总为 DataFrame
    combined_df = pd.DataFrame()
    col_names = ["Time"] + [f"Var{i}" for i in range(1, N)]
    for i, data_sample in enumerate(augmented_list):
        df = pd.DataFrame(data_sample, columns=col_names)
        df["Sample"] = i # 0 for original, 1 to n for augmented
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

# 主函数
def main():
    
    # 读取数据，注意路径使用原始字符串
    try:
        input_df = pd.read_excel(r"data\standardized_data.xlsx")
        #input_df = pd.read_excel(r"Time-Series-Library\dataset\\ETT\\ETTh1.csv")  # Fill NaN values with 0
    except FileNotFoundError:
        print("Error: 'data\\standardized_data.xlsx' not found. Please check the path.")
        print("Creating a dummy dataframe for demonstration purposes.")
        L_dummy, N_dummy = 50, 5 # 50 time points, 1 Time column + 4 variables
        data_dummy = np.random.rand(L_dummy, N_dummy)
        data_dummy[:,0] = np.arange(1, L_dummy + 1)
        input_df = pd.DataFrame(data_dummy, columns=["Time"] + [f"Var{i}" for i in range(1,N_dummy)])

    # Ensure the first column is a simple time index if needed by the logic,
    # or make sure it's treated appropriately if it's actual timestamps.
    # The current code replaces it.
    input_df.iloc[:, 0] = range(1, len(input_df) + 1) 
    input_data = input_df.to_numpy()
    
    print("Original data shape:", input_data.shape)
    # print("Sample of original data:\n", input_data[:5]) # Optional: print sample

    # Example: s=3 (neighborhood of 3 points), n=5 augmented samples
    df_final = generate_augmented_time_series_with_transformer(input_data, s=3, n=5, noise_std=0.01, seed=42)
    
    output_filename = "augmented_time_series_transformer_full.xlsx"
    df_final.to_excel(output_filename, index=False)
    print(f"Augmented data saved to {output_filename}")
    print("Shape of augmented data_frame:", df_final.shape)
    print("Number of unique samples:", df_final["Sample"].nunique())

if __name__ == "__main__":
    main()