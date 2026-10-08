import pm4py
import evaluator as sim
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import readers.log_reader as lr
import copy
import support_modules.common as cm
import numpy as np


class LogEvaluationPipeline:
    def __init__(self, seed: int = 42):
        self.sim_values = list()
        self.seed = seed
        self.parms = {'gl': {}}  # 核心修复：补全缺失的属性初始化

        # 强制设置 Pandas 显示精度
        pd.set_option('display.precision', 12)

        # 指标定义
        self.metrics = {
            'dl': 'Damerau-Levenshtein Distance',
            'tsd': 'Timed String Distance (Alpha-aware)',
            'ngd_2': '2-Gram Distance',
            'ngd_3': '3-Gram Distance',
            'aed': 'Absolute Event Distribution',
            'red': 'Relative Event Distribution',
            'ced': 'Circadian Event Distribution',
            'day_hour_emd': 'Day-Hour EMD',
            'car': 'Case Arrival Rate',
            'ctd': 'Cycle Time Distribution',
            'mae': 'Mean Absolute Error (Cycle Time)',
        }

        self.default_settings = {
            'read_options': {
                'one_timestamp': False,
                'column_names': {
                    'CaseID': 'caseid', 'Activity': 'task',
                    'Resource': 'user', 'StartTime': 'start_timestamp',
                    'EndTime': 'end_timestamp'
                },
                'filter_d_attrib': True
            }
        }

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载并预处理CSV日志"""
        print(f"加载CSV文件: {file_path}")
        try:
            df = pd.read_csv(file_path)

            # [DEBUG-1] 原始数据检查
            print(f"  [调试] 原始CSV列名: {df.columns.tolist()}")

            df = df.rename(columns=self.default_settings['read_options']['column_names'])
            df = self._prepare_data(df)
            return df
        except Exception as e:
            print(f"  [错误] 加载CSV失败: {e}")
            raise e

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        增强版数据预处理：
        1. 统一列名
        2. 智能解析时间（不再硬编码格式）
        3. 移除时区（DeepSimulator/Evaluator 不支持带时区的时间）
        """
        df = df.copy()

        # 1. 列名统一映射
        rename_map = {
            'resource': 'user',
        }
        df.rename(columns=rename_map, inplace=True)

        # 2. 处理缺失的 end_timestamp
        if 'end_timestamp' not in df.columns and 'start_timestamp' in df.columns:
            df['end_timestamp'] = df['start_timestamp']

        # 3. 智能时间转换
        time_cols = ['start_timestamp', 'end_timestamp']
        for col in time_cols:
            if col in df.columns:
                # 优化：只对非datetime类型进行转换
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
                    df[col] = pd.to_datetime(df[col], errors='coerce')

                # 4. 强制移除时区 (tz-naive)
                if hasattr(df[col], 'dt'):
                    if df[col].dt.tz is not None:
                        df[col] = df[col].dt.tz_localize(None)

        # [DEBUG-2] 预处理后检查
        print("  [调试] _prepare_data 完成后的状态:")
        for col in time_cols:
            if col in df.columns:
                nat_count = df[col].isna().sum()
                print(f"    - 列 '{col}': 类型={df[col].dtype}, NaT数量={nat_count}/{len(df)}")
                if len(df) > 0:
                    print(f"      示例值: {df[col].iloc[0]}")

        # 5. 严重错误阻断
        if 'start_timestamp' in df.columns and df['start_timestamp'].isna().all():
            print("  [严重错误] start_timestamp 全是 NaT！解析完全失败。")

        return df

    def evaluate_logs(self, real_log, simulated_log, metrics_to_evaluate: List[str] = None,
                      rep_num: int = 0, sim_file_name: str = "") -> List[Dict]:
        """评估日志相似度：记录成功与失败的每一次运行"""
        try:
            log = copy.deepcopy(real_log)
            if 'task' in log.columns:
                log = log[~log.task.isin(['Start', 'End'])].copy()
            log['source'] = 'log'
            if 'caseid' in log.columns:
                log['caseid'] = log['caseid'].astype(str)

            # 初始化评估器
            evaluator = sim.Evaluator(
                log_data=log, simulation_data=simulated_log,
                settings=self.default_settings, max_cases=1000,
                dtype='log', seed=self.seed
            )

            if metrics_to_evaluate is None:
                metrics_to_evaluate = list(self.metrics.keys())

            results = []
            for i, metric in enumerate(metrics_to_evaluate, 1):
                metric_name = self.metrics.get(metric, metric)
                # 关键修改：默认值为 NaN，确保失败也能记录
                sim_val = np.nan

                try:
                    print(f"[{i}/{len(metrics_to_evaluate)}] 评估 {metric_name}...", end=' ')
                    evaluator.measure_distance(metric)
                    if hasattr(evaluator, 'similarity') and evaluator.similarity:
                        sim_val = evaluator.similarity['sim_val']
                        print(f"✓ {sim_val:.10f}")
                    else:
                        print(f"✗ 失败: 未生成相似度结果")
                except Exception as e:
                    print(f"✗ 错误: {str(e)}")

                # 无论成功还是失败（NaN），都存入结果列表
                results.append({
                    'run_num': rep_num,
                    'sim_file': sim_file_name,
                    'metric': metric,
                    'metric_name': metric_name,
                    'sim_val': sim_val
                })

            self.sim_values.extend(results)
            return results
        except Exception:
            traceback.print_exc()
            return []
    def run_batch_evaluation(self,
                             real_log_path: str = None,
                             simulated_logs_dir: str = None,
                             metrics_to_evaluate: List[str] = None,
                             split_ratio: float = 0.8):
        """批量评估多个仿真日志文件"""
        print("\n" + "=" * 80)
        print(f"  批量日志高精度评估流水线启动 (Seed: {self.seed})")
        print("=" * 80)

        real_log_path_obj = Path(real_log_path or "data/input/xes/Production.xes")
        sim_logs_dir_obj = Path(simulated_logs_dir or "data/input/mp")

        if not real_log_path_obj.exists():
            print(f"[错误] 真实日志文件不存在: {real_log_path_obj}")
            return

        sim_files = sorted(sim_logs_dir_obj.glob("gen_Production_*.csv"))
        if not sim_files:
            print(f"[错误] 未找到仿真日志文件，路径: {sim_logs_dir_obj}")
            return

        # 加载真实日志并切分
        column_names = {
            'Case ID': 'caseid', 'Activity': 'task',
            'lifecycle:transition': 'event_type', 'Resource': 'user'
        }
        self.parms['gl'] = {'read_options': {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f', 'column_names': column_names,
            'one_timestamp': False, 'filter_d_attrib': True
        }}

        print(f"正在读取真实日志: {real_log_path_obj} ...")
        self.log = lr.LogReader(str(real_log_path_obj), self.parms['gl']['read_options'])
        train, test = cm.split_log(self.log, False, split_ratio)
        log_test_df = pd.DataFrame(test)

        # =========================================================
        # 【核心新增】清洗真实日志的时区 (与仿真日志对齐)
        # =========================================================
        print("正在清洗真实日志时区...")
        time_cols = ['start_timestamp', 'end_timestamp']
        for col in time_cols:
            if col in log_test_df.columns:
                # 确保是 datetime 类型
                log_test_df[col] = pd.to_datetime(log_test_df[col], errors='coerce')
                # 强制移除时区
                if hasattr(log_test_df[col], 'dt') and log_test_df[col].dt.tz is not None:
                    log_test_df[col] = log_test_df[col].dt.tz_localize(None)
        # =========================================================

        # [DEBUG] 检查真实日志时间
        print(f"真实日志(测试集)加载完成，行数: {len(log_test_df)}")
        if 'start_timestamp' in log_test_df.columns:
            print(f"真实日志时间示例 (已清洗): {log_test_df['start_timestamp'].iloc[0]}")

        # 循环评估每个仿真日志
        for rep_num, sim_file_path in enumerate(sim_files, 1):
            print(f"\n处理仿真日志 {rep_num}/{len(sim_files)}: {sim_file_path.name}")
            sim_log = self._load_csv(str(sim_file_path))
            self.evaluate_logs(
                real_log=log_test_df,
                simulated_log=sim_log,
                metrics_to_evaluate=metrics_to_evaluate,
                rep_num=rep_num,
                sim_file_name=sim_file_path.name
            )

        self._export_results()

    def _export_results(self):
        """
        核心导出逻辑：
        1. 仅保留 evaluation_results.csv。
        2. 按指标分组，每组后添加 MEAN 均值行。
        3. 即使存在 NaN，均值也会基于有效值计算。
        """
        print(f"\n{'=' * 60}\n  导出评估报告: evaluation_results.csv\n{'=' * 60}")
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.sim_values:
            print("没有产生结果数据。")
            return

        # 转换为 DataFrame
        df_raw = pd.DataFrame(self.sim_values)

        # 排序：先按指标代码排序，再按运行序号排序
        df_raw = df_raw.sort_values(by=['metric', 'run_num'])

        final_list = []

        # 遍历每个指标的分组
        for metric_code, group in df_raw.groupby('metric', sort=False):
            # 1. 添加原始运行数据（包含 NaN）
            final_list.append(group)

            # 2. 计算当前指标的均值
            # skipna=True 是 Pandas 默认属性，它会自动忽略 NaN 计算剩余数字的平均值
            mean_val = group['sim_val'].mean(skipna=True)

            # 3. 构造均值行 (标记为 AVG)
            mean_row = pd.DataFrame([{
                'run_num': 'AVG',
                'sim_file': f'--- {metric_code} MEAN ---',
                'metric': metric_code,
                'metric_name': group['metric_name'].iloc[0],
                'sim_val': mean_val
            }])
            final_list.append(mean_row)

        # 合并所有行
        final_df = pd.concat(final_list, ignore_index=True)

        # 导出唯一的 CSV 文件
        file_path = output_dir / "evaluation_results.csv"
        final_df.to_csv(file_path, index=False, float_format='%.16g')

        print(f"✓ 评估结果（含均值）已保存至: {file_path}")

        # 控制台实时预览
        print("\n指标均值汇总 (已忽略 NaN):")
        summary_view = final_df[final_df['run_num'] == 'AVG'][['metric', 'sim_val']]
        print(summary_view.to_string(index=False))
if __name__ == "__main__":
    # 使用固定种子启动实验
    pipeline = LogEvaluationPipeline(seed=42)
    pipeline.run_batch_evaluation(
        real_log_path="data/input/xes/Production.xes",
        simulated_logs_dir="data/input/mp",
        metrics_to_evaluate=None,  # 运行所有指标
        split_ratio=0.8
    )