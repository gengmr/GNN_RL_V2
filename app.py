# FILE: app.py

import pandas as pd
from flask import Flask, render_template, jsonify
import numpy as np

# 初始化 Flask 应用
app = Flask(__name__)

# 定义指标文件的路径
# 确保这个路径与您的训练脚本保存文件的位置一致
METRICS_FILE_PATH = "./checkpoints/training_metrics.csv"
EVAL_DETAILS_FILE_PATH = "./checkpoints/evaluation_results_details.csv"


@app.route('/')
def index():
    """渲染主仪表板页面。"""
    return render_template('index.html')


@app.route('/data')
def get_data():
    """
    提供训练指标数据的 API 端点。
    它会读取 CSV 文件并将其转换为 JSON 格式。
    """
    try:
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(METRICS_FILE_PATH)

        # [关键修复 - 更稳健的版本]
        # 直接将 numpy.nan 明确替换为 Python 的 None。
        # None 会被 jsonify 正确地序列化为 JSON 标准的 'null'。
        df.replace({np.nan: None}, inplace=True)

        # 将 DataFrame 转换为字典，格式为 {'列名': [值1, 值2, ...]}
        data = df.to_dict(orient='list')

        return jsonify(data)
    except FileNotFoundError:
        # 如果文件不存在，返回一个空的数据结构，前端会处理这种情况
        print(f"Warning: Metrics file not found at '{METRICS_FILE_PATH}'")
        return jsonify({})
    except Exception as e:
        # 捕获其他可能的读取错误
        print(f"An error occurred while reading the metrics file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/eval_details_data')
def get_eval_details_data():
    """
    提供详细评估结果数据的 API 端点。
    它会读取 evaluation_results_details.csv 文件并处理成适合绘制箱线图的格式。
    """
    try:
        # [优化] 如果文件不存在或为空，提前返回，避免后续错误
        if not os.path.exists(EVAL_DETAILS_FILE_PATH) or os.path.getsize(EVAL_DETAILS_FILE_PATH) == 0:
            print(f"Warning: Evaluation details file is missing or empty at '{EVAL_DETAILS_FILE_PATH}'")
            return jsonify({})

        df = pd.read_csv(EVAL_DETAILS_FILE_PATH)
        df.replace({np.nan: None}, inplace=True)

        # 按迭代次数分组，并将每个组的 makespan 转换为列表
        # [优化] 只选择最近的 N 个迭代进行显示，防止图表过于拥挤
        unique_iterations = sorted(df['iteration'].unique())
        latest_iterations = unique_iterations[-10:] # 最多显示最近10次评估
        df_filtered = df[df['iteration'].isin(latest_iterations)]

        grouped = df_filtered.groupby('iteration')

        # 准备用于JSON输出的数据结构
        iterations = []
        candidate_makespans = []
        best_model_makespans = []
        heft_makespans = []

        # 确保按迭代顺序处理
        for name in sorted(grouped.groups.keys()):
            group = grouped.get_group(name)
            iterations.append(name)
            candidate_makespans.append(group['candidate_makespan'].tolist())
            best_model_makespans.append(group['best_model_makespan'].tolist())
            heft_makespans.append(group['heft_makespan'].tolist())

        data = {
            'iterations': iterations,
            'candidate_makespans': candidate_makespans,
            'best_model_makespans': best_model_makespans,
            'heft_makespans': heft_makespans
        }

        return jsonify(data)
    except FileNotFoundError:
        print(f"Warning: Evaluation details file not found at '{EVAL_DETAILS_FILE_PATH}'")
        return jsonify({})
    except Exception as e:
        print(f"An error occurred while reading the evaluation details file: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 运行 Flask 应用
    # debug=True 允许多次修改后自动重载，但在生产环境中应设为 False
    app.run(host='0.0.0.0', port=5001, debug=True)