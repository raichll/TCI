import matplotlib.pyplot as plt

# Domain fluctuation values
s_values = [0.05 * (i + 1) for i in range(10)]

# Corrected performance data
data = {
    "Autoformer": {
        "MSE": [1.065, 1.0783, 1.124, 1.1157, 1.0921, 0.9671, 1.0593, 0.9154, 0.9494, 0.8579],
        "MAE": [0.7597, 0.7753, 0.7945, 0.7885, 0.7875, 0.7348, 0.789, 0.7173, 0.7246, 0.6922],
    },
    "Informer": {
        "MSE": [3.1288, 2.461, 2.4742, 2.3992, 2.4238, 2.1881, 1.9821, 1.9572, 2.0279, 1.8696],
        "MAE": [1.3012, 1.1408, 1.1428, 1.1203, 1.1388, 1.0818, 1.0343, 1.0459, 1.0636, 1.0174],
    },
    "PatchTST": {
        "MSE": [0.9106, 0.9891, 0.9703, 0.9472, 0.8828, 0.8601, 0.8343, 1.0256, 1.4824, 0.7858],
        "MAE": [0.6795, 0.7226, 0.7138, 0.7053, 0.6912, 0.6743, 0.6691, 0.7637, 0.8638, 0.6472],
    },
    "FEDformer": {
        "MSE": [1.0257, 1.0259, 1.0068, 0.9885, 0.9953, 0.9312, 0.89, 0.8855, 0.883, 0.8355],
        "MAE": [0.7507, 0.7526, 0.7499, 0.7392, 0.7455, 0.7223, 0.7086, 0.707, 0.6979, 0.6857],
    },
    "DLinear": {
        "MSE": [1.0054, 1.0001, 0.977, 0.9576, 0.9241, 0.9001, 0.8617, 0.8478, 0.8513, 0.8158],
        "MAE": [0.7362, 0.7369, 0.7311, 0.7264, 0.7184, 0.709, 0.6976, 0.6948, 0.6913, 0.6783],
    },
    "Transformer": {
        "MSE": [2.5344, 2.6035, 2.6838, 2.4133, 2.1295, 2.0029, 1.7407, 1.5243, 1.7552, 1.3715],
        "MAE": [1.1839, 1.1703, 1.213, 1.1761, 1.1201, 1.1189, 1.0105, 0.95, 1.0227, 0.8933],
    },
    "iTransformer": {
        "MSE": [0.8905, 0.874, 0.8794, 0.8651, 0.8226, 0.7991, 0.7758, 0.7717, 0.7961, 0.7408],
        "MAE": [0.6717, 0.6688, 0.6764, 0.6688, 0.6555, 0.6464, 0.6371, 0.6396, 0.6448, 0.6204],
    },
}

models = list(data.keys())

# Plotting both MSE and MAE in one figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot MSE
for model in models:
    axes[0].plot(s_values, data[model]["MSE"], marker='o', label=model)
axes[0].set_ylabel("MSE", fontsize=20)
axes[0].set_title("Model Performance under Domain Fluctuation", fontsize=14)
axes[0].grid(True)
axes[0].legend()

# Plot MAE
for model in models:
    axes[1].plot(s_values, data[model]["MAE"], marker='s', label=model)
axes[1].set_xlabel("The test results of the perturbation dataset of ETTh1 for 10 times", fontsize=18)
axes[1].set_ylabel("MAE", fontsize=20)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("MSE_MAE_combined_plot.png")
plt.show()
