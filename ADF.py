import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# 设置数据路径
data_folder = 'falling_data'  # 修改为你的数据文件夹路径

# 支持的文件扩展名
valid_extensions = ['.csv', '.tsv', '.txt', '.xlsx']

# 判断文件夹是否存在
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"目录不存在：{data_folder}")

# ADF + Ljung-Box 检验函数
def run_tests(series, column_name, file_name):
    result = {
        'file': file_name,
        'column': column_name
    }

    try:
        adf_result = adfuller(series.dropna())
        result['ADF Statistic'] = adf_result[0]
        result['ADF p-value'] = adf_result[1]
        result['Stationary (ADF)'] = adf_result[1] < 0.05
    except Exception as e:
        result['ADF Statistic'] = None
        result['ADF p-value'] = None
        result['Stationary (ADF)'] = None
        result['ADF Error'] = str(e)

    try:
        lb_result = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)
        lb_p = lb_result["lb_pvalue"].iloc[0]
        result['Ljung-Box p-value'] = lb_p
        result['White Noise (Ljung-Box)'] = lb_p > 0.05  # p > 0.05 表示接近白噪声
    except Exception as e:
        result['Ljung-Box p-value'] = None
        result['White Noise (Ljung-Box)'] = None
        result['Ljung-Box Error'] = str(e)

    return result

# 结果收集列表
results = []

# 遍历文件夹中的所有数据文件
for file_name in os.listdir(data_folder):
    if file_name.startswith('~$'):
        continue  # 跳过 Excel 临时文件
    if any(file_name.endswith(ext) for ext in valid_extensions):
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            continue  # 跳过子目录
        try:
            if file_name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_name.endswith('.tsv') or '\t' in open(file_path, encoding='utf-8', errors='ignore').read(1000):
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8', errors='ignore')
            else:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='gbk')

            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    result = run_tests(df[column], column, file_name)
                    results.append(result)
        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")

# 转换为 DataFrame 并保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('adf_ljungbox_results.csv', index=False)

print("✅ ADF + Ljung-Box 检验完成。结果已保存到 adf_ljungbox_results.csv")
