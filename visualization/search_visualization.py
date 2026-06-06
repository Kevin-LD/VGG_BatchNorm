import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator  


sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def load_search_results(json_path):
    """
    读取 search_report.json 并将其展平为 DataFrame。
    【优化项 1】自动将列表型超参数（如 num_blocks: [2, 2, 2]）智能转化为数值型标量。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到指定的搜索报告文件: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    runs_summary = data.get('runs_performance_summary', [])
    
    for item in runs_summary:
        if item.get('execution_status') != 'SUCCESS':
            continue
            
        record = {
            'run_name': item['run_name'],
            'val_acc': item.get('best_val_accuracy', 0.0),
            'val_loss': item.get('corresponding_val_loss', float('inf')),
            'train_acc': item.get('corresponding_train_accuracy', 0.0),
            'train_loss': item.get('corresponding_train_loss', float('inf')),
            'total_parameters': item.get('total_parameters', 0)
        }
        
        params = item.get('hyperparameters', {})
        for param_key, param_val in params.items():
            if isinstance(param_val, list):
                # 如果是数值型列表（如 [2, 2, 2]），尝试进行标量转化
                if len(param_val) > 0 and all(isinstance(x, (int, float)) for x in param_val):
                    if all(x == param_val[0] for x in param_val):
                        record[param_key] = param_val[0]  # [2, 2, 2] -> 2
                    else:
                        record[param_key] = sum(param_val) / len(param_val)  # 异构列表取均值
                else:
                    record[param_key] = str(param_val)
            else:
                record[param_key] = param_val
                
        records.append(record)
        
    if not records:
        raise ValueError(f"报告文件 {json_path} 中未包含任何成功的实验记录，无法可视化。")
        
    return pd.DataFrame(records)


def plot_marginal_effects(df, param_cols, save_dir):
    """
    绘制各个超参数针对验证集准确率（Validation Accuracy）的边缘主效应图。
    """
    num_params = len(param_cols)
    if num_params == 0:
        return

    if num_params <= 3:
        cols = num_params
    elif num_params == 4:
        cols = 2
    else:
        cols = 3  
        
    rows = int(np.ceil(num_params / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()
    
    idx = 0
    for idx, col in enumerate(param_cols):
        ax = axes[idx]
        
        is_discrete = not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 6
        
        if is_discrete:
            plot_df = df.copy()
            if pd.api.types.is_numeric_dtype(plot_df[col]):
                plot_df[col] = plot_df[col].apply(lambda x: f'{x:.1e}' if x < 1e-2 else f'{x:.2f}')
            else:
                plot_df[col] = plot_df[col].astype(str)
            
            sns.boxplot(x=col, y='val_acc', data=plot_df, ax=ax, color='#f1f3f5', showfliers=False)
            sns.stripplot(x=col, y='val_acc', data=plot_df, ax=ax, alpha=0.7, jitter=0.2, palette="deep")
            ax.set_title(f'Main Effect: {col}', weight='bold')
        else:
            use_log_scale = (df[col].max() / (df[col].min() + 1e-9)) > 100
            sns.regplot(x=col, y='val_acc', data=df, ax=ax, 
                        scatter_kws={'alpha': 0.6, 'color': '#1c7ed6'}, 
                        line_kws={'color': '#e03131', 'linewidth': 1.5}, 
                        logx=use_log_scale)
            
            if use_log_scale:
                ax.set_xscale('log')
                ax.set_title(f'Main Effect: {col} (Log Scale)', weight='bold')
            else:
                ax.set_title(f'Main Effect: {col}', weight='bold')
                
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Validation Accuracy', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'marginal_effects.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 边缘主效应图已成功保存至: {save_path}")
    plt.close()


def plot_correlation_heatmap(df, param_cols, save_dir):
    """
    绘制超参数数值表征与 val_acc 之间的 Pearson 相关性热力图。
    此时转换为数值的 num_blocks 会自动加入分析。
    """
    numeric_params = [c for c in param_cols if pd.api.types.is_numeric_dtype(df[c])]
    heatmap_cols = numeric_params + ['val_acc']
    
    if len(numeric_params) == 0:
        print("[Warning] 搜索空间内未检测到数值型超参数，跳过热力图绘制。")
        return
        
    corr_df = df[heatmap_cols].corr()
    
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title("Numerical Hyperparameters & Val Acc Correlation", weight='bold', pad=15)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 相关性热力图已成功保存至: {save_path}")
    plt.close()


def plot_top_models(df, param_cols, save_dir, top_n=5):
    """
    【优化项 2】绘制性能排名前 5 的超参数全空间配置散点图。
    - 削减至 Top 5，精简全局布局。
    - 移除 global mean 干扰线。
    - 使用 MaxNLocator 限制横坐标轴刻度密度，彻底消除百分比重叠和挤压现象。
    """
    top_df = df.sort_values(by='val_acc', ascending=False).head(top_n).copy()
    
    def generate_full_config_label(row):
        label_parts = []
        for col in param_cols:
            val = row[col]
            if isinstance(val, float):
                label_parts.append(f"{col}:{val:.1e}" if val < 1e-2 else f"{col}:{val:.2f}")
            else:
                label_parts.append(f"{col}:{val}")
        return " | ".join(label_parts)
        
    top_df['full_configuration'] = top_df.apply(generate_full_config_label, axis=1)
    top_df = top_df.iloc[::-1]
    
    plt.figure(figsize=(11, 3.5))  # 适度微调高度使 Top 5 的间距更松弛
    
    plt.plot(top_df['val_acc'], top_df['full_configuration'], 'o--', 
             color='#1c7ed6', markersize=8, linewidth=1.5, label='Best Val Accuracy')
    
    # 动态调配更宽松的观察视界边缘留白
    min_val = top_df['val_acc'].min()
    max_val = top_df['val_acc'].max()
    span = max_val - min_val
    padding = max(span * 0.6, 0.005)  # 至少给予 0.5% 的安全留白空间
    plt.xlim(min_val - padding, max_val + padding)
    
    ax = plt.gca()
    # 【核心修复】强制限制最大刻度数量为 4 或 5 个，从根本上防止刻度文本过密堆叠
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.2f}%'))
    
    plt.title(f"Top {top_n} Hyperparameter Configurations Space Comparison", weight='bold', pad=15)
    plt.xlabel("Validation Accuracy")
    plt.ylabel("Hyperparameter Space Specifications")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'top_models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] 前 {top_n} 强全参空间对比图已成功保存至: {save_path}")
    plt.close()


def get_latest_search_dir(base_dir='runs'):
    """自动扫描 runs 目录下时间戳最新的 search_* 父文件夹"""
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith('search_')]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def execute_visualization_flow(json_path=None):
    """
    独立的可视化执行控制流函数。
    """
    if json_path is None or json_path == "":
        latest_dir = get_latest_search_dir('runs')
        if latest_dir is None:
            raise FileNotFoundError("[Error] 未在 runs/ 线下检索到任何符合 'search_*' 命名的父文件夹。")
        json_path = os.path.join(latest_dir, 'search_report.json')
        target_experiment_dir = latest_dir
        print(f"[*] 已自动捕获最新实验报告: {json_path}")
    else:
        target_experiment_dir = os.path.dirname(json_path)

    save_figures_dir = os.path.join(target_experiment_dir, 'figures')
    os.makedirs(save_figures_dir, exist_ok=True)
    
    print("-" * 60)
    print(f"  VISUALIZATION ENGINE LAUNCHED")
    print(f"  Target Report: {json_path}")
    print(f"  Output Folder: {save_figures_dir}")
    print("-" * 60)
    
    try:
        experiment_df = load_search_results(json_path)
        print(f"[*] 数据成功载入，检测到已完成的有效实验样本共计: {len(experiment_df)} 组。")
        
        exclude_cols = ['run_name', 'val_acc', 'val_loss', 'train_acc', 'train_loss', 'total_parameters']
        param_cols = [c for c in experiment_df.columns if c not in exclude_cols]
        
        plot_marginal_effects(experiment_df, param_cols, save_figures_dir)
        plot_correlation_heatmap(experiment_df, param_cols, save_figures_dir)
        plot_top_models(experiment_df, param_cols, save_figures_dir, top_n=5)
        
        print("\n" + "=" * 60)
        print("          ALL VISUALIZATION CHARTS GENERATED SUCCESSFULLY         ")
        print(f" All outputs are saved safely under: {save_figures_dir}")
        print("=" * 60 + "\n")
        
    except Exception as error:
        print(f"[Fatal Error] 可视化分析流非正常中断，错误原因: {error}")
        raise error


if __name__ == "__main__":
    target_json = "runs/search_20260603_223602/search_report.json"
    execute_visualization_flow(json_path=target_json)