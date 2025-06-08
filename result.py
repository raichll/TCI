import pandas as pd
import re
import os

# 定义数据集名称映射
DATASET_MAPPING = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "electricity": "Electricity",
    "exchange_rate": "Exchange_Rate",
    "national_illness": "National_Illness",
    "weather": "Weather",
    "traffic": "Traffic"
}

# 定义模型名称列表
MODELS = [
    "Autoformer", "Informer", "PatchTST", "FEDformer", 
    "DLinear", "Transformer", "iTransformer"
]

def extract_results(file_path):
    """从文件中提取结果数据"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print(f"成功读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None
    
    # 更灵活的正则表达式模式匹配
    pattern = r'long_term_forecast_(\w+)_\d+_\d+_(\w+)[\w_]*?mse:([\d.]+), mae:([\d.]+)'
    
    datasets = []
    models = []
    mse_values = []
    mae_values = []
    experiment_counts = {}
    
    # 按空行分割每个实验记录
    blocks = content.strip().split('\n\n')
    print(f"找到 {len(blocks)} 个数据块")
    
    # 调试计数器
    matched_blocks = 0
    
    for i, block in enumerate(blocks, 1):
        lines = block.split('\n')
        if len(lines) < 2:
            print(f"块 {i} 行数不足: {len(lines)}")
            continue
            
        header = lines[0].strip()
        metrics = lines[1].strip()
        
        # 提取数据集和模型名称
        match = re.search(pattern, header)
        if match:
            raw_dataset = match.group(1)
            model = match.group(2)
            mse = float(match.group(3))
            
            # 提取MAE值
            mae_match = re.search(r'mae:([\d.]+)', metrics)
            if mae_match:
                mae = float(mae_match.group(1))
            else:
                print(f"块 {i} 中未找到MAE值: {metrics}")
                continue
                
            # 标准化数据集名称
            dataset = DATASET_MAPPING.get(raw_dataset, raw_dataset)
            
            # 计算实验次数
            key = (dataset, model)
            exp_num = experiment_counts.get(key, 0) + 1
            experiment_counts[key] = exp_num
            
            datasets.append(dataset)
            models.append(model)
            mse_values.append(mse)
            mae_values.append(mae)
            matched_blocks += 1
        else:
            print(f"块 {i} 未匹配: {header}")
    
    print(f"成功匹配 {matched_blocks} 个数据块")
    
    if not datasets:
        print("未提取到任何有效数据")
        return None
    
    return pd.DataFrame({
        'Dataset': datasets,
        'Model': models,
        'Experiment': [f'Exp{experiment_counts[(d,m)]}' for d, m in zip(datasets, models)],
        'MSE': mse_values,
        'MAE': mae_values
    })

def create_excel_table(df, output_file):
    """创建Excel表格并保存"""
    if df is None or df.empty:
        print("错误: 没有数据可处理")
        return None
    
    # 重塑表格结构
    pivot_df = df.pivot_table(
        index='Dataset',
        columns=['Model', 'Experiment'],
        values=['MSE', 'MAE']
    )
    
    # 确保所有模型都被包含
    for model in MODELS:
        if model not in pivot_df.columns.get_level_values(1):
            print(f"警告: 模型 {model} 在数据中缺失，添加空列")
            for exp in ['Exp1', 'Exp2']:
                for metric in ['MSE', 'MAE']:
                    pivot_df[(metric, model, exp)] = pd.NA
    
    # 重排列层级 (Model > Experiment > Metric)
    pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
    pivot_df.columns = pivot_df.columns.swaplevel(1, 2)
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1])
    
    # 保存到Excel
    try:
        pivot_df.to_excel(output_file, float_format="%.4f")
        print(f"成功保存Excel文件: {output_file}")
        return pivot_df
    except Exception as e:
        print(f"保存Excel文件时出错: {str(e)}")
        return None

# 主程序
if __name__ == "__main__":
    input_file = "Time-Series-Library/result_long_term_forecast.txt"
    output_file = "Forecast_Results.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 未找到")
        exit(1)
    
    print(f"开始处理文件: {input_file}")
    print("=" * 50)
    
    # 提取数据
    df = extract_results(input_file)
    
    if df is None or df.empty:
        print("\n错误: 未提取到任何数据")
        print("=" * 50)
        print("调试建议:")
        print("1. 检查文件格式是否与示例一致")
        print("2. 验证正则表达式是否匹配文件内容")
        print("3. 尝试手动检查前几个数据块")
        exit(1)
    
    # 验证数据集
    unique_datasets = df['Dataset'].unique()
    print(f"\n提取到 {len(unique_datasets)} 个数据集: {', '.join(sorted(unique_datasets))}")
    
    # 验证模型
    unique_models = df['Model'].unique()
    print(f"提取到 {len(unique_models)} 个模型: {', '.join(sorted(unique_models))}")
    
    # 创建Excel表格
    print("\n创建Excel表格...")
    result_table = create_excel_table(df, output_file)
    
    if result_table is not None:
        print("\n处理完成!")
        print("=" * 50)
        print(f"生成的Excel文件: {output_file}")
        print(f"数据集数量: {len(result_table.index)}")
        print(f"列数量: {len(result_table.columns)}")
    else:
        print("\n生成Excel文件失败")