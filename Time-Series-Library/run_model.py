import os
import torch

def main():
    # 获取当前脚本完整路径和目录
     
    try:
        current_file_path = os.path.abspath(__file__)
    except NameError:
        # 交互式环境时用argv[0]
        import sys
        current_file_path = os.path.abspath(sys.argv[0])

    current_folder = os.path.dirname(current_file_path)
    print(f"当前脚本完整路径: {current_file_path}")
    print(f"当前脚本目录: {current_folder}")
    
   
    # 8个模型列表
    models = [
        'Autoformer',
        'Informer',
        'TimesNet',
        'PatchTST',
        'FEDformer',
        'DLinear',
        'Transformer',
        'iTransformer'
    ]
    models = [
        'Autoformer',
        'Informer',        
        'PatchTST',        
        'DLinear',
        'Transformer',
        'iTransformer'
    ]
    models = [
       
        'iTransformer'
    ]
    
    

    # 公共参数
    common_args = {
        '--enc_in' :'7',
        '--dec_in':'7',
        '--c_out':'7',
        '--train_epochs': '2',
        '--use_gpu': 'True',
        '--task_name': 'long_term_forecast',
        '--is_training': '1',
        '--data': 'ETTh1',
        '--root_path': r'E:\project\TCI\Time-Series-Library\dataset_refine',
        '--data_path': 'ETTh1.csv',
        '--features': 'M',
        '--seq_len': '96',
        '--label_len': '48',
        '--pred_len': '96',
        '--e_layers': '2',
        '--d_layers': '1',
        '--factor': '3',
        '--enc_in': '7',
        '--dec_in': '7',
        '--c_out': '7',
        '--des': 'Exp',
        '--itr': '1',
        '--batch_size': '32',
        '--learning_rate': '0.001'
    }
    
    for model in models:
        print(f"\n运行模型: {model}")
        # 这里打印当前脚本目录，方便调试
        print(f"Current script directory: {current_folder}")
        
        # 拼接命令
        cmd = ['python', 'e:\\project\\TCI\\Time-Series-Library\\run.py']
        cmd += ['--model_id', 'ETTh1_96_96']
        cmd += ['--model', model]
        for k, v in common_args.items():
            cmd += [k, v]

        # 把命令列表转换为字符串（注意路径和参数中不要有空格导致问题）
        cmd_str = ' '.join(cmd)
        print(f"执行命令: {cmd_str}")

        # 执行系统命令
        os.system(cmd_str)
       

if __name__ == "__main__":
    main()

