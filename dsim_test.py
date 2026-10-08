"""
dsim_test.py
使用DSIM的SimilarityEvaluator对6个数据集各5个仿真日志进行评估，
计算dl、mae和day_hour_emd三个指标。

测试日志: input_files/event_logs/exp_data/test_partitions/tst_{dataset}.csv
仿真日志: data/dsim_test/{dataset}/gen_{dataset}_{1-5}.csv

指标说明:
  - dl          : Damerau-Levenshtein 距离归一化后的相似度 (0~1, 越高越好)
  - mae         : 周期时间(cycle time)的平均绝对误差 (秒, 越低越好)
  - day_hour_emd: 按星期+小时分组的Wasserstein/EMD距离 (越低越好)
"""

import os
import random
import traceback
import numpy as np
import pandas as pd

# ============================================================
# 随机种子（保证可复现性）
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# pandas >= 2.0 兼容性补丁
# DSIM 的 log_emd_metric 使用了已在 pandas 2.0 中移除的 DataFrame.append
# ============================================================
if not hasattr(pd.DataFrame, 'append'):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: \
        pd.concat([self, other], ignore_index=ignore_index, **kw)

from analyzers.sim_evaluator import SimilarityEvaluator

# ============================================================
# 路径与参数配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_LOG_DIR = os.path.join(BASE_DIR, 'input_files', 'event_logs',
                            'exp_data', 'test_partitions')
SIM_LOG_DIR = os.path.join(BASE_DIR, 'data', 'dsim_test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'results')

NUM_RUNS = 5
METRICS = ['dl', 'mae', 'day_hour_emd']
MAX_CASES = 500
SETTINGS = {'read_options': {'one_timestamp': False}}

DATASETS = [
    'BPI_Challenge_2012_W_Two_TS',
    'BPI_Challenge_2017_W_Two_TS',
    'ConsultaDataMining201618',
    'Production',
    'PurchasingExample',
    'cvs_pharmacy',
]

# 指标显示名（用于输出）
METRIC_NAMES = {
    'dl': 'Damerau-Levenshtein (similarity)',
    'mae': 'MAE Cycle Time (seconds)',
    'day_hour_emd': 'Day-Hour EMD (distance)',
}


def load_log(file_path):
    """
    加载CSV日志文件，解析时间戳，保留SimilarityEvaluator所需的列。
    所需列: caseid, task, start_timestamp, end_timestamp
    """
    df = pd.read_csv(file_path)

    # 解析时间戳
    for col in ['start_timestamp', 'end_timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # 移除时区（如果有），DSIM不支持带时区的时间
            if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

    # 只保留 SimilarityEvaluator 所需的列
    keep = ['caseid', 'task', 'start_timestamp', 'end_timestamp']
    available = [c for c in keep if c in df.columns]
    df = df[available]

    # 移除无效行
    df = df.dropna(subset=['caseid', 'task', 'start_timestamp',
                           'end_timestamp'])
    # caseid 统一为字符串
    df['caseid'] = df['caseid'].astype(str)
    return df


def evaluate_run(test_log, sim_log, metrics):
    """
    对一组 (test_log, sim_log) 创建 SimilarityEvaluator 并计算所有指标。
    预处理只在构造函数中执行一次，所有指标共享同一预处理结果。
    """
    results = {}
    evaluator = SimilarityEvaluator(
        log_data=test_log.copy(),
        simulation_data=sim_log.copy(),
        settings=SETTINGS,
        max_cases=MAX_CASES,
        dtype='log'
    )
    for metric in metrics:
        evaluator.measure_distance(metric)
        results[metric] = evaluator.similarity['sim_val']
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for dataset in DATASETS:
        test_path = os.path.join(TEST_LOG_DIR, 'tst_{}.csv'.format(dataset))
        if not os.path.exists(test_path):
            print('[跳过] 测试日志不存在: {}'.format(test_path))
            continue

        print('\n' + '=' * 60)
        print('数据集: {}'.format(dataset))
        print('=' * 60)

        test_log = load_log(test_path)
        n_test_cases = test_log['caseid'].nunique()
        print('  测试日志: {}'.format(test_path))
        print('    事件数={}, 案例数={}'.format(len(test_log), n_test_cases))

        for run in range(1, NUM_RUNS + 1):
            sim_path = os.path.join(SIM_LOG_DIR, dataset,
                                    'gen_{}_{}.csv'.format(dataset, run))
            if not os.path.exists(sim_path):
                print('  [跳过] 仿真日志不存在: {}'.format(sim_path))
                continue

            print('\n  --- Run {}/{}: {} ---'.format(
                run, NUM_RUNS, os.path.basename(sim_path)))
            sim_log = load_log(sim_path)
            n_sim_cases = sim_log['caseid'].nunique()
            print('    事件数={}, 案例数={}'.format(
                len(sim_log), n_sim_cases))

            try:
                results = evaluate_run(test_log, sim_log, METRICS)
                for metric in METRICS:
                    val = results[metric]
                    print('    {:15s}: {:.10f}'.format(metric, val))
                    all_results.append({
                        'dataset': dataset,
                        'run': run,
                        'metric': metric,
                        'metric_name': METRIC_NAMES[metric],
                        'value': val,
                    })
            except Exception as e:
                print('    [错误] {}'.format(e))
                traceback.print_exc()
                for metric in METRICS:
                    all_results.append({
                        'dataset': dataset,
                        'run': run,
                        'metric': metric,
                        'metric_name': METRIC_NAMES[metric],
                        'value': np.nan,
                    })

    # ============================================================
    # 保存详细结果
    # ============================================================
    df = pd.DataFrame(all_results)
    out_path = os.path.join(OUTPUT_DIR, 'dsim_test_results.csv')
    df.to_csv(out_path, index=False, float_format='%.10g')
    print('\n' + '=' * 60)
    print('详细结果已保存至: {}'.format(out_path))

    # ============================================================
    # 汇总表（每个数据集每个指标的 5 次运行 mean ± std）
    # ============================================================
    print('\n' + '=' * 60)
    print('汇总 ({}次运行 mean +/- std)'.format(NUM_RUNS))
    print('=' * 60)
    summary = df.groupby(['dataset', 'metric'])['value'].agg(
        ['mean', 'std']).reset_index()
    print(summary.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, 'dsim_test_summary.csv')
    summary.to_csv(summary_path, index=False, float_format='%.10g')
    print('\n汇总已保存至: {}'.format(summary_path))


if __name__ == '__main__':
    main()
