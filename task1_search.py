import os
import copy
import json
from datetime import datetime
from itertools import product
# 从训练脚本中导入修改后的核心训练函数
from task1_train import train_model

def run_hyperparameter_search():
    # 1. 基础配置初始化（作为搜索空间的默认基准底稿）
    base_config = {
        'epochs': 30,
        'batch_size': 128,
        'lr': 1e-3,
        'val_split': 0.1,
        'base_channels': 64,
        'num_blocks': [2, 2, 2],
        'dropout_rate': 0.3,
        'activation_type': 'relu',
        'optimizer_type': 'adamw',
        'weight_decay': 1e-4,
        'loss_type': 'ce',
        'label_smoothing': 0.1
    }

    # 2. 定义核心超参数搜索空间 (Grid Search Space)
    search_space = {
        'base_channels': [32, 64],
        'num_blocks': [[2, 2, 2], [3, 3, 3]],
        'optimizer_type': ['adamw', 'sgd'],
        'lr': [1e-2, 1e-3, 5e-4],
        'weight_decay': [1e-4, 1e-5],
        'dropout_rate': [0.1, 0.3]
    }

    # 3. 动态创建本次实验搜索的总文件夹 (格式：runs/task1/search_YYYYMMDD_HHMMSS)
    search_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parent_search_dir = os.path.join('runs/task1', f"search_{search_timestamp}")
    os.makedirs(parent_search_dir, exist_ok=True)
    
    # 4. 利用笛卡尔积生成实验网格
    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    total_runs = len(experiments)
    
    print("=" * 60)
    print(f" HYPERPARAMETER SEARCH INITIALIZED")
    print(f" Master Directory: {parent_search_dir}")
    print(f" Total Planned Runs: {total_runs}")
    print("=" * 60)

    # 初始化大报告字典，用于最终保存为 JSON 文件
    master_report = {
        'search_timestamp': search_timestamp,
        'search_space': search_space,
        'total_runs_planned': total_runs,
        'runs_performance_summary': []
    }

    # 5. 循环遍历所有组合执行序列化训练并拦截指标
    for idx, custom_params in enumerate(experiments, 1):
        print(f"\n[Run {idx}/{total_runs}] Preparing grid configuration...")
        
        # 执行深拷贝，确保每轮实验的配置字典完全独立
        current_config = copy.deepcopy(base_config)
        current_config.update(custom_params)
        
        # 联动修正逻辑：若损失函数选择标准 'ce'，则强制将标签平滑置为 0.0
        if current_config['loss_type'] == 'ce':
            current_config['label_smoothing'] = 0.0
            
        # 规范化子文件夹命名：增加前缀序号如 "run_01_", "run_02_" 以便文件系统自然排序
        blocks_str = "".join(map(str, current_config.get('num_blocks', [2, 2, 2])))
        sub_run_name = f"run_{idx:02d}_{current_config['optimizer_type']}_ch{current_config['base_channels']}_b{blocks_str}"

        print(f"Active Parameters: {custom_params}")

        try:
            # 将实验归拢到 parent_search_dir 目录下，并捕获传回的性能字典
            run_summary = train_model(
                config=current_config,
                save_dir=parent_search_dir,
                project_name="cifar10_hyperparameter_search",
                run_name=sub_run_name
            )
            
            # 补充运行状态与当前具体的超参数映射，方便后续分析
            run_summary['hyperparameters'] = custom_params
            run_summary['execution_status'] = 'SUCCESS'
            
            # 将该轮次结果追加至全局报告
            master_report['runs_performance_summary'].append(run_summary)
            print(f"[Run {idx}/{total_runs}] Completed. Best Val Acc: {run_summary['best_val_accuracy']*100:.2f}%")

        except Exception as e:
            print(f"[Run {idx}/{total_runs}] Execution failed with Error: {e}")
            # 容错处理：即使单次实验失败，也保留其日志记录，确保长周期运行不因意外中断
            failed_summary = {
                'run_name': sub_run_name,
                'execution_status': f'FAILED: {str(e)}',
                'hyperparameters': custom_params
            }
            master_report['runs_performance_summary'].append(failed_summary)
            continue
            
    # 6. 按验证集准确率从高到低对整个实验网格的结果进行排序
    master_report['runs_performance_summary'].sort(
        key=lambda x: x.get('best_val_accuracy', 0.0), 
        reverse=True
    )

    # 7. 将完整的实验总结报告写入总目录下的 JSON 文件
    report_save_path = os.path.join(parent_search_dir, 'search_report.json')
    with open(report_save_path, 'w', encoding='utf-8') as f:
        json.dump(master_report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("          ALL HYPERPARAMETER SWEEPS COMPLETED SUCCESSFULLY          ")
    print(f" Master Search Report Saved to: {report_save_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_hyperparameter_search()
