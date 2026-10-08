import copy
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import evaluator as sim
import readers.log_reader as lr
import support_modules.common as cm

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REAL_LOG_DIR = BASE_DIR / "data" / "input" / "xes"
DEFAULT_SIM_ROOT_DIR = BASE_DIR / "data" / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "results"


class LogEvaluationPipeline:
    def __init__(self, seed: int = 42):
        self.sim_values = []
        self.seed = seed
        self.parms = {'gl': {}}

        pd.set_option('display.precision', 12)
        self.sim_dir_map = {
            "Production": "mp",
            "PurchasingExample": "P2P",
            "cvs_pharmacy": "cvs",
            "ConsultaDataMining201618": "ACR",
            "BPI_Challenge_2017_W_Two_TS": "BPI17W",
            "BPI_Challenge_2012_W_Two_TS": "BPI12W",
        }
        self.metrics = {
            'cfld': 'Control-flow Log Distance (CFLD)',
            'ngd_2': '2-Gram Distance',
            'ngd_3': '3-Gram Distance',
            'aed': 'Absolute Event Distribution',
            'red': 'Relative Event Distribution',
            'ced': 'Circadian Event Distribution',
            'car': 'Case Arrival Rate',
            'ctd': 'Cycle Time Distribution',
        }

        self.default_settings = {
            'read_options': {
                'one_timestamp': False,
                'column_names': {
                    'CaseID': 'caseid',
                    'Case ID': 'caseid',
                    'Activity': 'task',
                    'Resource': 'user',
                    'resource': 'user',
                    'StartTime': 'start_timestamp',
                    'EndTime': 'end_timestamp'
                },
                'filter_d_attrib': True
            }
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {
            **self.default_settings['read_options']['column_names'],
            'start_time': 'start_timestamp',
            'end_time': 'end_timestamp',
            'resource': 'user',
        }
        return df.rename(columns=rename_map)

    @staticmethod
    def _strip_timezone(series: pd.Series) -> pd.Series:
        series = pd.to_datetime(series, errors='coerce')
        if hasattr(series, 'dt') and getattr(series.dt, 'tz', None) is not None:
            series = series.dt.tz_localize(None)
        return series

    def _prepare_data(self, df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
        df = self._standardize_columns(df)

        required_cols = ['caseid', 'task', 'start_timestamp', 'end_timestamp']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{source_name} 缺少严格评估所需列: {missing}。必须至少包含 {required_cols}")


        df = df.copy()
        df['caseid'] = df['caseid'].astype(str)
        df['task'] = df['task'].astype(str)

        df['start_timestamp'] = self._strip_timezone(df['start_timestamp'])
        df['end_timestamp'] = self._strip_timezone(df['end_timestamp'])

        before_len = len(df)
        df = df.dropna(subset=['caseid', 'task', 'start_timestamp', 'end_timestamp']).copy()
        dropped_na = before_len - len(df)

        before_len_2 = len(df)
        df = df[df['end_timestamp'] >= df['start_timestamp']].copy()
        dropped_invalid = before_len_2 - len(df)

        df = df.sort_values(
            by=['caseid', 'end_timestamp', 'start_timestamp', 'task']
        ).reset_index(drop=True)

        print(f"[数据准备] {source_name}: 原始行数={before_len}, "
              f"删除缺失时间行={dropped_na}, 删除非法时间行={dropped_invalid}, "
              f"保留行数={len(df)}")
        return df
    def _get_sim_logs_dir_for_log(self, log_stem: str, sim_root_dir: Path) -> Path:
        """
        根据真实日志 stem 自动定位对应的仿真日志目录
        """
        if log_stem not in self.sim_dir_map:
            raise ValueError(
                f"未为真实日志 {log_stem} 配置仿真目录映射，请检查 self.sim_dir_map"
            )

        sim_subdir = self.sim_dir_map[log_stem]
        sim_dir = sim_root_dir / sim_subdir

        if not sim_dir.exists():
            raise FileNotFoundError(
                f"真实日志 {log_stem} 对应的仿真目录不存在: {sim_dir}"
            )

        return sim_dir
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        print(f"加载CSV文件: {file_path}")
        df = pd.read_csv(file_path)
        print(f"  [调试] 原始CSV列名: {df.columns.tolist()}")
        return self._prepare_data(df, source_name=f"仿真日志 {Path(file_path).name}")

    def _prepare_real_test_log(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_data(df, source_name="真实日志测试集")

    @staticmethod
    def _drop_boundary_tasks(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
        if not enabled or 'task' not in df.columns:
            return df
        return df[~df['task'].isin(['Start', 'End'])].copy()

    @staticmethod
    def _summarize_log_window(df: pd.DataFrame, title: str):
        if df.empty:
            print(f"[窗口检查] {title}: 日志为空")
            return

        case_cnt = df['caseid'].nunique() if 'caseid' in df.columns else 0
        min_start = df['start_timestamp'].min() if 'start_timestamp' in df.columns else None
        max_end = df['end_timestamp'].max() if 'end_timestamp' in df.columns else None

        print(f"[窗口检查] {title}: "
              f"events={len(df)}, cases={case_cnt}, "
              f"min_start={min_start}, max_end={max_end}")

    @staticmethod
    def _check_alignment(real_df: pd.DataFrame, sim_df: pd.DataFrame, sim_name: str):
        if real_df.empty or sim_df.empty:
            return

        real_cases = real_df['caseid'].nunique()
        sim_cases = sim_df['caseid'].nunique()

        real_min = real_df['start_timestamp'].min()
        sim_min = sim_df['start_timestamp'].min()

        print(f"[对齐检查] {sim_name}: "
              f"real_cases={real_cases}, sim_cases={sim_cases}, "
              f"real_min_start={real_min}, sim_min_start={sim_min}")

        if real_cases != sim_cases:
            print(f"[警告] {sim_name}: 仿真日志 case 数与测试集不一致。"
                  f"严格论文版 CFLD 要求两边 case 数一致，否则 CFLD 会报错或不可比。")

    def evaluate_logs(
        self,
        real_log: pd.DataFrame,
        simulated_log: pd.DataFrame,
        metrics_to_evaluate: Optional[List[str]] = None,
        rep_num: int = 0,
        sim_file_name: str = "",
        log_name: str = "",
        drop_boundary_tasks: bool = True
    ) -> List[Dict]:
        try:
            log = copy.deepcopy(real_log)
            sim_log_copy = copy.deepcopy(simulated_log)

            log = self._drop_boundary_tasks(log, enabled=drop_boundary_tasks)
            sim_log_copy = self._drop_boundary_tasks(sim_log_copy, enabled=drop_boundary_tasks)

            log['source'] = 'log'
            sim_log_copy['source'] = 'simulation'

            log['caseid'] = log['caseid'].astype(str)
            sim_log_copy['caseid'] = sim_log_copy['caseid'].astype(str)
            log['task'] = log['task'].astype(str)
            sim_log_copy['task'] = sim_log_copy['task'].astype(str)

            self._check_alignment(log, sim_log_copy, sim_file_name)

            evaluator = sim.Evaluator(
                log_data=log,
                simulation_data=sim_log_copy,
                settings=self.default_settings,
                max_cases=1000,
                dtype='log',
                seed=self.seed
            )

            if metrics_to_evaluate is None:
                metrics_to_evaluate = list(self.metrics.keys())

            results = []
            for i, metric in enumerate(metrics_to_evaluate, 1):
                metric_name = self.metrics.get(metric, metric)
                sim_val = np.nan

                try:
                    print(f"[{i}/{len(metrics_to_evaluate)}] 评估 {metric_name}...", end=' ')
                    evaluator.measure_distance(metric)

                    if hasattr(evaluator, 'similarity') and evaluator.similarity:
                        sim_val = evaluator.similarity['sim_val']
                        print(f"✓ {sim_val:.10f}")
                    else:
                        print("✗ 未生成相似度结果")
                except Exception as e:
                    print(f"✗ 错误: {e}")

                results.append({
                    'log_name': log_name,
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

    def _build_reader_params(self):
        column_names = {
            'Case ID': 'caseid',
            'Activity': 'task',
            'lifecycle:transition': 'event_type',
            'Resource': 'user'
        }
        self.parms['gl'] = {
            'read_options': {
                'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                'column_names': column_names,
                'one_timestamp': False,
                'filter_d_attrib': True
            }
        }

    def run_single_log_evaluation(
            self,
            real_log_path: str,
            simulated_logs_dir: str = None,
            metrics_to_evaluate: Optional[List[str]] = None,
            split_ratio: float = 0.8,
            drop_boundary_tasks: bool = True
    ):
        """
        针对单个真实日志执行评估，并输出单独结果文件
        simulated_logs_dir 现在表示仿真日志根目录，例如 data/input
        """
        self.sim_values = []

        real_log_path_obj = Path(real_log_path)
        sim_root_dir_obj = Path(simulated_logs_dir) if simulated_logs_dir else DEFAULT_SIM_ROOT_DIR

        if not real_log_path_obj.exists():
            print(f"[错误] 真实日志文件不存在: {real_log_path_obj}")
            return

        if not sim_root_dir_obj.exists():
            print(f"[错误] 仿真日志根目录不存在: {sim_root_dir_obj}")
            return

        if not (0 < split_ratio < 1):
            print(f"[错误] split_ratio 必须在 (0,1) 内，当前为 {split_ratio}")
            return

        log_stem = real_log_path_obj.stem

        # 自动定位当前真实日志对应的仿真目录
        try:
            sim_logs_dir_obj = self._get_sim_logs_dir_for_log(log_stem, sim_root_dir_obj)
        except Exception as e:
            print(f"[错误] 无法定位 {log_stem} 对应的仿真目录: {e}")
            return

        sim_pattern = f"gen_{log_stem}_*.csv"
        sim_files = sorted(sim_logs_dir_obj.glob(sim_pattern))

        print("\n" + "=" * 80)
        print(f"  开始评估日志: {real_log_path_obj.name}")
        print("=" * 80)
        print(f"[仿真目录] {sim_logs_dir_obj}")
        print(f"[仿真日志匹配模式] {sim_pattern}")

        if not sim_files:
            print(f"[警告] 未找到与 {real_log_path_obj.name} 对应的仿真日志，目录: {sim_logs_dir_obj}")
            return

        print(f"[切分策略] 完全复现 DeepSimulator：cm.split_log(self.log, False, {split_ratio})")
        print(f"[切分说明] split_ratio 原样传入 cm.split_log，不做 0.8->0.2 转换")

        self._build_reader_params()

        print(f"正在读取真实日志: {real_log_path_obj} ...")
        self.log = lr.LogReader(str(real_log_path_obj), self.parms['gl']['read_options'])

        train_df, test_df = cm.split_log(self.log, False, split_ratio)

        train_df = pd.DataFrame(train_df)
        test_df = pd.DataFrame(test_df)

        print(f"[切分完成] 使用与 DeepSimulator 完全一致的 split_ratio={split_ratio}")
        print(f"[切分完成] train_events={len(train_df)}, test_events={len(test_df)}")

        if not train_df.empty and 'caseid' in train_df.columns:
            print(f"[切分完成] train_cases={train_df['caseid'].nunique()}")
        if not test_df.empty and 'caseid' in test_df.columns:
            print(f"[切分完成] test_cases={test_df['caseid'].nunique()}")

        log_test_df = self._prepare_real_test_log(test_df)

        test_for_eval = self._drop_boundary_tasks(log_test_df, enabled=drop_boundary_tasks).copy()
        self._summarize_log_window(test_for_eval, f"真实测试集(评估用, {real_log_path_obj.name})")

        if not test_for_eval.empty:
            eval_anchor = test_for_eval['start_timestamp'].min()
            print(f"[评估锚点] 测试集最早开始时间: {eval_anchor}")

        for rep_num, sim_file_path in enumerate(sim_files, 1):
            print(f"\n处理仿真日志 {rep_num}/{len(sim_files)}: {sim_file_path.name}")

            try:
                sim_log = self._load_csv(str(sim_file_path))
                sim_for_eval = self._drop_boundary_tasks(sim_log, enabled=drop_boundary_tasks).copy()
                self._summarize_log_window(sim_for_eval, f"仿真日志 {sim_file_path.name}")

                self.evaluate_logs(
                    real_log=log_test_df,
                    simulated_log=sim_log,
                    metrics_to_evaluate=metrics_to_evaluate,
                    rep_num=rep_num,
                    sim_file_name=sim_file_path.name,
                    log_name=log_stem,
                    drop_boundary_tasks=drop_boundary_tasks
                )
            except Exception as e:
                print(f"[错误] 处理仿真日志失败: {sim_file_path.name}, 原因: {e}")

                failed_metrics = metrics_to_evaluate if metrics_to_evaluate is not None else list(self.metrics.keys())

                fail_rows = []
                for metric in failed_metrics:
                    fail_rows.append({
                        'log_name': log_stem,
                        'run_num': rep_num,
                        'sim_file': sim_file_path.name,
                        'metric': metric,
                        'metric_name': self.metrics.get(metric, metric),
                        'sim_val': np.nan
                    })
                self.sim_values.extend(fail_rows)

        self._export_results(log_stem)

    def run_all_logs_evaluation(
            self,
            real_logs_dir: str = None,
            simulated_logs_dir: str = None,
            metrics_to_evaluate: Optional[List[str]] = None,
            split_ratio: float = 0.8,
            drop_boundary_tasks: bool = True,
            log_file_names: Optional[List[str]] = None
    ):
        """
        批量评估多个 xes 日志。
        - 真实日志统一在 xes 目录
        - 仿真日志根目录统一是 data/input
        - 每个真实日志自动映射到不同子目录
        - 每个日志单独输出一个结果文件
        """
        real_logs_dir_obj = Path(real_logs_dir) if real_logs_dir else DEFAULT_REAL_LOG_DIR
        sim_root_dir_obj = Path(simulated_logs_dir) if simulated_logs_dir else DEFAULT_SIM_ROOT_DIR

        if not real_logs_dir_obj.exists():
            print(f"[错误] 真实日志目录不存在: {real_logs_dir_obj}")
            return

        if not sim_root_dir_obj.exists():
            print(f"[错误] 仿真日志根目录不存在: {sim_root_dir_obj}")
            return

        if log_file_names:
            real_log_files = [real_logs_dir_obj / name for name in log_file_names]
        else:
            real_log_files = sorted(real_logs_dir_obj.glob("*.xes"))

        if not real_log_files:
            print(f"[错误] 未找到任何 .xes 日志文件，目录: {real_logs_dir_obj}")
            return

        print("\n" + "=" * 80)
        print(f"  开始批量评估 {len(real_log_files)} 个日志")
        print("=" * 80)

        for log_path in real_log_files:
            if not log_path.exists():
                print(f"[警告] 跳过不存在的日志文件: {log_path}")
                continue

            self.run_single_log_evaluation(
                real_log_path=str(log_path),
                simulated_logs_dir=str(sim_root_dir_obj),
                metrics_to_evaluate=metrics_to_evaluate,
                split_ratio=split_ratio,
                drop_boundary_tasks=drop_boundary_tasks
            )
    def _export_results(self, log_stem: str):
        print(f"\n{'=' * 60}\n  导出评估报告: {log_stem}\n{'=' * 60}")

        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.sim_values:
            print("没有产生结果数据。")
            return

        df_raw = pd.DataFrame(self.sim_values)
        df_raw = df_raw.sort_values(by=['metric', 'run_num'])

        final_list = []

        for metric_code, group in df_raw.groupby('metric', sort=False):
            final_list.append(group)

            mean_val = group['sim_val'].mean(skipna=True)

            mean_row = pd.DataFrame([{
                'log_name': log_stem,
                'run_num': 'AVG',
                'sim_file': f'--- {metric_code} MEAN ---',
                'metric': metric_code,
                'metric_name': group['metric_name'].iloc[0],
                'sim_val': mean_val
            }])
            final_list.append(mean_row)

        final_df = pd.concat(final_list, ignore_index=True)

        file_path = output_dir / f"evaluation_results_{log_stem}.csv"
        final_df.to_csv(file_path, index=False, float_format='%.16g')

        print(f"✓ 评估结果（含均值）已保存至: {file_path}")

        print("\n指标均值汇总 (已忽略 NaN):")
        summary_view = final_df[final_df['run_num'] == 'AVG'][['metric', 'sim_val']]
        print(summary_view.to_string(index=False))


if __name__ == "__main__":
    pipeline = LogEvaluationPipeline(seed=42)

    # pipeline.run_all_logs_evaluation(
    #     real_logs_dir=str(DEFAULT_REAL_LOG_DIR),
    #     simulated_logs_dir=str(DEFAULT_SIM_ROOT_DIR),
    #     metrics_to_evaluate=None,
    #     split_ratio=0.8,
    #     drop_boundary_tasks=True
    # )
    pipeline.run_single_log_evaluation(
        # Production
        real_log_path=str(DEFAULT_REAL_LOG_DIR / "Production.xes"),
        # # PurchasingExample
        # real_log_path = str(DEFAULT_REAL_LOG_DIR / "PurchasingExample.xes"),
        # cvs_pharmacy
        # real_log_path = str(DEFAULT_REAL_LOG_DIR / "cvs_pharmacy.xes"),
        # # ConsultaDataMining201618
        # real_log_path = str(DEFAULT_REAL_LOG_DIR / "ConsultaDataMining201618.xes"),
        # # BPI 2017
        # real_log_path = str(DEFAULT_REAL_LOG_DIR / "BPI_Challenge_2017_W_Two_TS.xes"),
        # # BPI 2012
        # real_log_path = str(DEFAULT_REAL_LOG_DIR / "BPI_Challenge_2012_W_Two_TS.xes"),
        simulated_logs_dir=str(DEFAULT_SIM_ROOT_DIR),
        metrics_to_evaluate=None,
        split_ratio=0.8,
        drop_boundary_tasks=True
    )

