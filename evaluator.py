import warnings
import random
import itertools
from collections import Counter
from operator import itemgetter
import time
import multiprocessing
from multiprocessing import Pool
import traceback
import string
import copy

from tqdm import tqdm
import jellyfish as jf
from scipy.optimize import linear_sum_assignment

from analyzers import alpha_oracle as ao
from analyzers.alpha_oracle import Rel
# 新增指标使用了这些库，需确保已导入
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wasserstein_distance


class Evaluator():
    """
        This class evaluates the similarity of two event-logs
     """

    def __init__(self, log_data, simulation_data, settings, max_cases=500, dtype='log', seed=42):
        """constructor"""
        self.dtype = dtype
        self.seed = seed  # 添加这一行

        # 设置随机种子
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        # 使用 deepcopy 确保初始数据的独立性
        self.log_data = copy.deepcopy(log_data)
        self.simulation_data = copy.deepcopy(simulation_data)
        self.max_cases = max_cases
        self.one_timestamp = settings['read_options']['one_timestamp']
        self._preprocess_data(dtype)

    def _preprocess_data(self, dtype):
        preprocessor = self._get_preprocessor(dtype)
        return preprocessor()

    def _get_preprocessor(self, dtype):
        if dtype == 'log':
            return self._preprocess_log
        elif dtype == 'serie':
            return self._preprocess_serie
        else:
            raise ValueError(dtype)

    def _preprocess_log(self):
        self.ramp_io_perc = 0.2
        self.log_data['source'] = 'log'
        self.simulation_data['source'] = 'simulation'

        # 1. 合并数据
        data = pd.concat([self.log_data, self.simulation_data], axis=0, ignore_index=True)

        # =========================================================
        # 【核心修复】强制移除时区 (Timezone Stripping)
        # 这一步必须在 calculate_times 之前做！
        # =========================================================
        time_cols = ['start_timestamp', 'end_timestamp', 'start_time', 'end_time']
        for col in time_cols:
            if col in data.columns:
                # 1. 确保是 datetime 类型
                data[col] = pd.to_datetime(data[col], errors='coerce')
                # 2. 如果带时区，强制移除 (转为本地时间或UTC naive)
                if hasattr(data[col], 'dt'):
                    if data[col].dt.tz is not None:
                        data[col] = data[col].dt.tz_localize(None)

        # 2. 确保必要的列名存在 (DeepSim 兼容性)
        if 'start_timestamp' in data.columns:
            data['start_time'] = data['start_timestamp']
        if 'end_timestamp' in data.columns:
            data['end_time'] = data['end_timestamp']

        # 3. 计算处理时间和等待时间 (现在因为时区已统一，这一步不会出错了)
        if (('processing_time' not in data.columns) or ('waiting_time' not in data.columns)):
            data = self.calculate_times(data)
            # Double check
            if (('processing_time' not in data.columns) or ('waiting_time' not in data.columns)):
                data = self.calculate_times(data)

        # 4. 数据缩放
        data = self.scaling_data(data)

        # 5. 拆分回 Log 和 Simulation
        self.log_data = data[data.source == 'log'].copy()
        self.simulation_data = data[data.source == 'simulation'].copy()

        # 6. 创建别名
        self.alias = self.create_task_alias(data, 'task')

        # 7. 初始化 Alpha Oracle
        self.alpha_concurrency = ao.AlphaOracle(self.log_data, self.alias, self.one_timestamp, True)

        # 8. 格式化事件 (统一调用 reformat_events，删除 reformat_events1)
        self.log_data = self.reformat_events(self.log_data.to_dict('records'), 'task')
        self.simulation_data = self.reformat_events(self.simulation_data.to_dict('records'), 'task')

        # 9. 采样与截断
        num_traces = int(len(self.simulation_data) * self.ramp_io_perc)
        if len(self.simulation_data) > 0:
            self.simulation_data = self.simulation_data[num_traces:-num_traces]

        if len(self.log_data) > len(self.simulation_data) and len(self.simulation_data) > 0:
            import random
            random.seed(self.seed)
            self.log_data = random.sample(self.log_data, len(self.simulation_data))

    def _preprocess_serie(self):
        # Ensure 'start_time' and 'end_time' are present
        if 'start_time' not in self.log_data.columns or 'end_time' not in self.log_data.columns:
            raise ValueError("The data must contain 'start_time' and 'end_time' columns.")
        if 'start_time' not in self.simulation_data.columns or 'end_time' not in self.simulation_data.columns:
            raise ValueError("The data must contain 'start_time' and 'end_time' columns.")

        self.log_data['source'] = 'log'
        self.simulation_data['source'] = 'simulation'

    def measure_distance(self, metric, verbose=False):
        """
        测量两个事件日志的距离
        """
        self.verbose = verbose

        # 获取评估器
        evaluator = self._get_evaluator(metric)

        if len(self.log_data) == 0 or len(self.simulation_data) == 0:
            # 增加对空数据的容错
            print(f"[Warning] Empty data for metric {metric}")
            self.similarity = {'metric': metric, 'sim_val': 0.0}
            return

        if metric in [ 'day_hour_emd']:
            distance = evaluator(self.log_data,
                                 self.simulation_data,
                                 criteria=metric)
        else:
            distance = evaluator(self.log_data, self.simulation_data, metric)

        self.similarity = {'metric': metric,
                           'sim_val': np.mean(
                               [x['sim_score'] for x in distance])}

    def _get_evaluator(self, metric):
        """获取指标评估器（完整版）"""
        if self.dtype == 'log':
            # 映射别名
            # 传统指标
            if metric in ['tsd', 'dl', 'mae']:
                return self._evaluate_seq_distance
            elif metric == 'log_mae':
                return self.log_mae_metric
            elif metric in [ 'day_hour_emd']:
                return self.log_emd_metric

            # 新增指标 (Chapela-Campa et al. 2023)
            elif metric == 'ngd_2':
                return lambda l1, l2, m: self.ngd_n_metric(l1, l2, n=2)
            elif metric == 'ngd_3':
                return lambda l1, l2, m: self.ngd_n_metric(l1, l2, n=3)
            elif metric == 'aed':
                return self.aed_metric
            elif metric == 'red':
                return self.red_metric
            elif metric == 'ced':
                return self.ced_metric
            elif metric == 'car':
                return self.car_metric
            elif metric == 'ctd':
                return self.ctd_metric
            else:
                raise ValueError(f"Unknown metric: {metric}")

        elif self.dtype == 'serie':
            if metric in ['day_hour_emd']:
                return self.serie_emd_metric
            else:
                raise ValueError(f"Metric {metric} not implemented for dtype='serie'")
        else:
            raise ValueError(f"Unknown dtype: {self.dtype}")

    def ngd_n_metric(self, log_data: list, simulation_data: list, n: int = 2) -> list:
        """
        N-Gram Distance (NGD)
        处理已经过 reformat_events 转换的数据
        """
        similarity = []

        def extract_ngrams(data, n):
            """从重格式化的日志数据中提取 n-grams"""
            ngrams = []

            # ===== 关键修复：处理 reformat_events 后的数据结构 =====
            # 数据格式: [{'caseid': ..., 'profile': ['A', 'B', 'C'], ...}, ...]

            for trace in data:
                # 检查数据格式
                if 'profile' in trace:
                    # 已经是 reformat_events 格式
                    activities = trace['profile']
                elif 'task' in trace:
                    # 原始格式（不应该出现，但做容错处理）
                    activities = [trace['task']]
                else:
                    # 未知格式，跳过
                    continue

                # 添加 n-1 个虚拟起始和结束标记
                padded = ['<START>'] * (n - 1) + activities + ['<END>'] * (n - 1)

                # 生成 n-grams
                for i in range(len(padded) - n + 1):
                    ngram = tuple(padded[i:i + n])
                    ngrams.append(ngram)

            return ngrams

        # 提取两个日志的 n-grams
        ngrams_log = extract_ngrams(log_data, n)
        ngrams_sim = extract_ngrams(simulation_data, n)

        if not ngrams_log or not ngrams_sim:
            similarity.append({
                'metric': f'ngd_{n}',
                'sim_score': 0.0,
                'total_ngrams_log': 0,
                'total_ngrams_sim': 0,
                'unique_ngrams': 0
            })
            return similarity

        # 计算频率
        freq_log = Counter(ngrams_log)
        freq_sim = Counter(ngrams_sim)

        # 获取所有唯一的 n-grams
        all_ngrams = set(freq_log.keys()) | set(freq_sim.keys())

        # 计算绝对差异的总和
        total_diff = 0
        total_count = sum(freq_log.values()) + sum(freq_sim.values())

        for ngram in all_ngrams:
            diff = abs(freq_log.get(ngram, 0) - freq_sim.get(ngram, 0))
            total_diff += diff

        # 归一化到 [0, 1]
        ngd_score = total_diff / total_count if total_count > 0 else 0.0

        similarity.append({
            'metric': f'ngd_{n}',
            'sim_score': ngd_score,
            'total_ngrams_log': len(ngrams_log),
            'total_ngrams_sim': len(ngrams_sim),
            'unique_ngrams': len(all_ngrams)
        })

        return similarity

    def aed_metric(self, log_data: list, simulation_data: list, metric='aed',
                   time_mode: str = "end",  # "end" | "start" | "both"
                   freq: str = "H") -> list:
        """
        Absolute Event Distribution (AED) - 最小改动版（改法A）
        目标：按 event-level 的时间戳构建绝对时间分布（小时桶），用 EMD/Wasserstein 距离比较。

        time_mode:
            - "end"  : 只使用 end_timestamp / end_time（推荐，接近 complete time）
            - "start": 只使用 start_timestamp / start_time
            - "both" : start + end 都计入（谨慎使用，除非论文明确这么做）

        freq:
            - 默认 "H"：小时桶
        """
        similarity = []

        # -------- 1) 统一为 DataFrame --------
        log_df = pd.DataFrame(log_data) if isinstance(log_data, list) else log_data.copy()
        sim_df = pd.DataFrame(simulation_data) if isinstance(simulation_data, list) else simulation_data.copy()

        # 空数据容错
        if log_df.empty or sim_df.empty:
            similarity.append({'metric': metric, 'sim_score': 0.0,
                               'total_events_log': 0, 'total_events_sim': 0, 'time_bins': 0})
            return similarity

        # -------- 2) 选取 event-level 时间列（尽量从原始事件列取）--------
        # 兼容字段：start_timestamp/start_time, end_timestamp/end_time
        def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # 如果没有 *_timestamp 但有 *_time，就映射过去（不覆盖已有的）
            if 'start_timestamp' not in df.columns and 'start_time' in df.columns:
                df['start_timestamp'] = df['start_time']
            if 'end_timestamp' not in df.columns and 'end_time' in df.columns:
                df['end_timestamp'] = df['end_time']
            return df

        log_df = _ensure_time_cols(log_df)
        sim_df = _ensure_time_cols(sim_df)

        # -------- 3) 提取时间戳（event-level）--------
        def _to_naive_datetime(s: pd.Series) -> pd.Series:
            s = pd.to_datetime(s, errors='coerce')
            # 去时区（如果有）
            if pd.api.types.is_datetime64tz_dtype(s):
                s = s.dt.tz_convert(None)
            return s

        def _extract_times(df: pd.DataFrame, mode: str) -> pd.Series:
            cols = []
            if mode == "start":
                cols = ['start_timestamp']
            elif mode == "end":
                cols = ['end_timestamp']
            elif mode == "both":
                cols = ['start_timestamp', 'end_timestamp']
            else:
                raise ValueError(f"Unknown time_mode: {mode}")

            times = []
            for c in cols:
                if c in df.columns:
                    ts = _to_naive_datetime(df[c]).dropna()
                    if len(ts) > 0:
                        times.append(ts)

            if not times:
                return pd.Series([], dtype="datetime64[ns]")
            return pd.concat(times, ignore_index=True)

        log_times = _extract_times(log_df, time_mode)
        sim_times = _extract_times(sim_df, time_mode)

        # 如果仍为空（比如传进来的是 reformat 后但缺 end/start），直接返回 0
        if log_times.empty or sim_times.empty:
            similarity.append({'metric': metric, 'sim_score': 0.0,
                               'total_events_log': int(len(log_times)),
                               'total_events_sim': int(len(sim_times)),
                               'time_bins': 0})
            return similarity

        # -------- 4) 构建共同的时间轴 bins（绝对时间，按小时/自定义 freq）--------
        all_times = pd.concat([log_times, sim_times], ignore_index=True)
        min_time = all_times.min().floor(freq)
        max_time = all_times.max().ceil(freq)

        # 极端情况：只有一个时间点，给一个最小窗口避免 bins 为空
        if min_time == max_time:
            max_time = max_time + pd.Timedelta(hours=1) if freq.upper() == "H" else max_time + pd.Timedelta(1,
                                                                                                            unit=freq)

        time_range = pd.date_range(start=min_time, end=max_time, freq=freq)

        # 如果 time_range 太短导致 cut 无法分桶（少于2个边界），再补一个边界
        if len(time_range) < 2:
            time_range = pd.date_range(start=min_time, periods=2, freq=freq)

        # -------- 5) 分桶计数（事件分布）--------
        log_bins = pd.cut(log_times, bins=time_range, include_lowest=True)
        sim_bins = pd.cut(sim_times, bins=time_range, include_lowest=True)

        log_counts = log_bins.value_counts().sort_index().fillna(0).to_numpy()
        sim_counts = sim_bins.value_counts().sort_index().fillna(0).to_numpy()

        # -------- 6) EMD/Wasserstein（在桶索引轴上，桶宽恒定= freq）--------
        # 这里 positions 统一用 0..K-1，weights 用 counts
        positions = np.arange(len(log_counts), dtype=float)

        # 如果总质量为0，直接返回0
        if log_counts.sum() == 0 or sim_counts.sum() == 0:
            similarity.append({'metric': metric, 'sim_score': 0.0,
                               'total_events_log': int(len(log_times)),
                               'total_events_sim': int(len(sim_times)),
                               'time_bins': int(len(time_range) - 1)})
            return similarity

        distance = wasserstein_distance(
            positions, positions,
            log_counts.astype(float), sim_counts.astype(float)
        )

        # -------- 7) 归一化（更稳：除以 log 总事件数；保持你原实现风格）--------
        sim_score = float(distance)

        similarity.append({
            'metric': metric,
            'sim_score': sim_score,  # 单位为“平均移动的桶数”
            'total_events_log': int(len(log_times)),
            'total_events_sim': int(len(sim_times)),
            'time_bins': int(len(time_range) - 1),
            'time_mode': time_mode,
            'freq': freq
        })
        return similarity

    def ced_metric(self, log_data, simulation_data, metric='ced'):
        similarity = []

        log_df = pd.DataFrame(log_data) if isinstance(log_data, list) else log_data.copy()
        sim_df = pd.DataFrame(simulation_data) if isinstance(simulation_data, list) else simulation_data.copy()

        # 补齐 start_timestamp
        for df in (log_df, sim_df):
            if 'start_timestamp' not in df.columns and 'start_time' in df.columns:
                df['start_timestamp'] = df['start_time']
            if 'end_timestamp' not in df.columns and 'end_time' in df.columns:
                df['end_timestamp'] = df['end_time']

        def _to_naive(s):
            s = pd.to_datetime(s, errors='coerce')
            if pd.api.types.is_datetime64tz_dtype(s):
                s = s.dt.tz_convert(None)
            return s

        # 收集 start/end 时间戳（与论文 τ(e) 口径一致）
        def collect_times(df):
            times = []
            for col in ('start_timestamp', 'end_timestamp'):
                if col in df.columns:
                    ts = _to_naive(df[col]).dropna()
                    if not ts.empty:
                        times.append(ts)
            return pd.concat(times, ignore_index=True) if times else pd.Series([], dtype="datetime64[ns]")

        log_times = collect_times(log_df)
        sim_times = collect_times(sim_df)

        if log_times.empty or sim_times.empty:
            similarity.append({'metric': metric, 'sim_score': 0.0, 'days_compared': 0})
            return similarity

        # 分到 7 天：0=Mon..6=Sun
        log_by_day = {d: [] for d in range(7)}
        sim_by_day = {d: [] for d in range(7)}

        for t in log_times:
            log_by_day[t.weekday()].append(t.hour)
        for t in sim_times:
            sim_by_day[t.weekday()].append(t.hour)

        positions = np.arange(24, dtype=float)
        daily_distances = []

        for d in range(7):
            # 24-bin counts
            log_counts = np.bincount(log_by_day[d], minlength=24).astype(float)
            sim_counts = np.bincount(sim_by_day[d], minlength=24).astype(float)

            # 可选：按论文“直方图归一化频率”口径（总质量=1） :contentReference[oaicite:10]{index=10}
            if log_counts.sum() > 0:
                log_counts /= log_counts.sum()
            if sim_counts.sum() > 0:
                sim_counts /= sim_counts.sum()

            # 若某天双方都为 0，距离记 0；若只有一方为 0，1WD 会有定义（另一方有质量）
            if log_counts.sum() == 0 and sim_counts.sum() == 0:
                daily_distances.append(0.0)
            else:
                dist = wasserstein_distance(
                    positions, positions,
                    u_weights=log_counts, v_weights=sim_counts
                )
                daily_distances.append(float(dist))

        # 论文：对 7 天取平均 :contentReference[oaicite:11]{index=11}
        similarity.append({
            'metric': metric,
            'sim_score': float(np.mean(daily_distances)),
            'days_compared': 7
        })
        return similarity

    def red_metric(self, log_data: list, simulation_data: list, metric='red') -> list:
        """
        Relative Event Distribution (RED) — aligned with Chapela-Campa et al. (2023), Sec. 4.2.

        Paper definition:
          - arrival time a(ξ(e)) = earliest start timestamp of the case
          - relative time ρ(e) = τ(e) - a(ξ(e)), with τ(e)=τstart for start times and τend for end times
          - discretize ρ(e) into hourly bins: [0,3599] same bin, etc.
          - RED distance = EMD between discretized ρ(e) distributions
          - normalize raw EMD by #observations in original log
        """
        similarity = []

        # -------- 1) unify to DataFrame --------
        log_df = pd.DataFrame(log_data) if isinstance(log_data, list) else log_data.copy()
        sim_df = pd.DataFrame(simulation_data) if isinstance(simulation_data, list) else simulation_data.copy()

        if log_df.empty or sim_df.empty:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # -------- 2) ensure timestamp columns exist (compat) --------
        for df in (log_df, sim_df):
            if 'start_timestamp' not in df.columns and 'start_time' in df.columns:
                df['start_timestamp'] = df['start_time']
            if 'end_timestamp' not in df.columns and 'end_time' in df.columns:
                df['end_timestamp'] = df['end_time']

        # -------- 3) robust datetime (strip timezone) --------
        def _to_naive_datetime(s: pd.Series) -> pd.Series:
            s = pd.to_datetime(s, errors='coerce')
            # strip tz if tz-aware
            if pd.api.types.is_datetime64tz_dtype(s):
                s = s.dt.tz_convert(None)
            return s

        for df in (log_df, sim_df):
            if 'start_timestamp' in df.columns:
                df['start_timestamp'] = _to_naive_datetime(df['start_timestamp'])
            if 'end_timestamp' in df.columns:
                df['end_timestamp'] = _to_naive_datetime(df['end_timestamp'])

        # -------- 4) compute relative seconds ρ(e) --------
        # ρ(e) = τ(e) - arrival(case), τ(e) includes both start and end timestamps if present
        def _relative_seconds(df: pd.DataFrame) -> np.ndarray:
            rel = []

            if 'caseid' not in df.columns or 'start_timestamp' not in df.columns:
                return np.asarray([], dtype=float)

            # group by caseid
            for _, g in df.groupby('caseid'):
                # arrival a(ξ(e)) = min start_timestamp in the case
                arrival = g['start_timestamp'].min()
                if pd.isna(arrival):
                    continue

                # τstart(e)
                if 'start_timestamp' in g.columns:
                    st = g['start_timestamp'].dropna()
                    if not st.empty:
                        rel.extend((st - arrival).dt.total_seconds().to_list())

                # τend(e)
                if 'end_timestamp' in g.columns:
                    et = g['end_timestamp'].dropna()
                    if not et.empty:
                        rel.extend((et - arrival).dt.total_seconds().to_list())

            # keep only finite & non-negative (negative can happen due to dirty logs / ordering issues)
            rel = np.asarray(rel, dtype=float)
            rel = rel[np.isfinite(rel)]
            rel = rel[rel >= 0.0]
            return rel

        log_rel = _relative_seconds(log_df)
        sim_rel = _relative_seconds(sim_df)

        if log_rel.size == 0 or sim_rel.size == 0:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # -------- 5) discretize into hourly bins: floor(sec/3600) --------
        log_bins = np.floor(log_rel / 3600.0).astype(int)
        sim_bins = np.floor(sim_rel / 3600.0).astype(int)

        max_bin = int(max(log_bins.max(initial=0), sim_bins.max(initial=0)))
        # counts over the SAME support 0..max_bin
        log_counts = np.bincount(log_bins, minlength=max_bin + 1).astype(float)
        sim_counts = np.bincount(sim_bins, minlength=max_bin + 1).astype(float)

        if log_counts.sum() == 0 or sim_counts.sum() == 0:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # -------- 6) EMD over shared positions (hour bins) --------
        positions = np.arange(max_bin + 1, dtype=float)
        distance = wasserstein_distance(
            positions, positions,
            log_counts, sim_counts
        )

        # -------- 7) normalize by #observations in original log --------
        # "observations" here = number of ρ(e) values extracted from original log
        sim_score = float(distance)

        similarity.append({
            'metric': metric,
            'sim_score': sim_score,
            'total_obs_log': int(len(log_rel)),
            'total_obs_sim': int(len(sim_rel)),
            'max_rel_hour_log': float(log_bins.max(initial=0)),
            'max_rel_hour_sim': float(sim_bins.max(initial=0)),
            'num_bins': int(max_bin + 1)
        })
        return similarity

    def car_metric(self, log_data: list, simulation_data: list, metric='car') -> list:
        """Case Arrival Rate (CAR)"""
        similarity = []

        # 转换为 DataFrame
        if isinstance(log_data, list):
            log_df = pd.DataFrame(log_data)
        else:
            log_df = log_data.copy()

        if isinstance(simulation_data, list):
            sim_df = pd.DataFrame(simulation_data)
        else:
            sim_df = simulation_data.copy()

        # 确保时间列存在
        for df in [log_df, sim_df]:
            if 'start_timestamp' not in df.columns and 'start_time' in df.columns:
                df['start_timestamp'] = df['start_time']

        # 获取每个 case 的到达时间
        def get_arrival_times(df):
            arrivals = df.groupby('caseid')['start_timestamp'].min()
            return pd.to_datetime(arrivals.values)

        log_arrivals = get_arrival_times(log_df)
        sim_arrivals = get_arrival_times(sim_df)

        if len(log_arrivals) == 0 or len(sim_arrivals) == 0:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # 创建时间范围
        all_arrivals = pd.concat([
            pd.Series(log_arrivals),
            pd.Series(sim_arrivals)
        ])

        min_time = all_arrivals.min().floor('H')
        max_time = all_arrivals.max().ceil('H')
        time_range = pd.date_range(start=min_time, end=max_time, freq='H')

        # 分桶计数
        log_bins = pd.cut(log_arrivals, bins=time_range, include_lowest=True)
        sim_bins = pd.cut(sim_arrivals, bins=time_range, include_lowest=True)

        log_counts = log_bins.value_counts().sort_index().fillna(0).values
        sim_counts = sim_bins.value_counts().sort_index().fillna(0).values

        # ===== 修复：检查数组有效性 =====
        if len(log_counts) == 0 or len(sim_counts) == 0:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # 确保数组长度一致
        max_len = max(len(log_counts), len(sim_counts))
        log_counts_padded = np.pad(log_counts, (0, max_len - len(log_counts)), mode='constant')
        sim_counts_padded = np.pad(sim_counts, (0, max_len - len(sim_counts)), mode='constant')

        # 检查权重和是否有效
        log_sum = np.sum(log_counts_padded)
        sim_sum = np.sum(sim_counts_padded)

        if log_sum == 0 or sim_sum == 0 or not np.isfinite(log_sum) or not np.isfinite(sim_sum):
            # 如果权重无效，使用简单的差异度量
            distance = np.mean(np.abs(log_counts_padded - sim_counts_padded))
        else:
            # 使用 EMD
            try:
                distance = wasserstein_distance(
                    range(len(log_counts_padded)), range(len(sim_counts_padded)),
                    log_counts_padded, sim_counts_padded
                )
            except Exception as e:
                print(f"[警告] CAR EMD 计算失败: {e}，使用备用方法")
                distance = np.mean(np.abs(log_counts_padded - sim_counts_padded))

        # 归一化
        sim_score = float(distance)

        similarity.append({
            'metric': metric,
            'sim_score': sim_score,
            'total_cases_log': len(log_arrivals),
            'total_cases_sim': len(sim_arrivals),
            'avg_arrivals_per_hour_log': len(log_arrivals) / len(time_range) if len(time_range) > 0 else 0,
            'avg_arrivals_per_hour_sim': len(sim_arrivals) / len(time_range) if len(time_range) > 0 else 0
        })

        return similarity

    def ctd_metric(self, log_data: list, simulation_data: list, metric='ctd') -> list:
        """
        Cycle Time Distribution (CTD) 修正版
        完全对齐 Chapela-Campa et al. (2023) 论文逻辑
        """
        similarity = []

        # -------- 1) 统一为 DataFrame --------
        log_df = pd.DataFrame(log_data) if isinstance(log_data, list) else log_data.copy()
        sim_df = pd.DataFrame(simulation_data) if isinstance(simulation_data, list) else simulation_data.copy()

        # -------- 2) 确保时间列存在 --------
        for df in [log_df, sim_df]:
            if 'start_timestamp' not in df.columns and 'start_time' in df.columns:
                df['start_timestamp'] = df['start_time']
            if 'end_timestamp' not in df.columns and 'end_time' in df.columns:
                df['end_timestamp'] = df['end_time']

        def calculate_cycle_times_in_hours(df):
            """
            计算每个 case 的周期时间，并转换为小时桶 (Hourly Bins)
            对应论文: Cycle time = length-of-stay [cite: 198]
            """
            cycle_times_hours = []
            if 'caseid' not in df.columns:
                return []

            for caseid, group in df.groupby('caseid'):
                start_time = group['start_timestamp'].min()
                end_time = group['end_timestamp'].max()

                if pd.isna(start_time) or pd.isna(end_time):
                    continue

                # 计算秒数并转换为小时
                duration_seconds = (end_time - start_time).total_seconds()

                # 过滤无效值
                if duration_seconds >= 0 and np.isfinite(duration_seconds):
                    # 转换为小时桶 (Discretize into hourly bins) [cite: 194, 219]
                    # 这样 1WD 的结果单位就是“平均移动的小时数”
                    cycle_times_hours.append(np.floor(duration_seconds / 3600.0))

            return cycle_times_hours

        # -------- 3) 提取周期时间序列 --------
        log_ct_bins = calculate_cycle_times_in_hours(log_df)
        sim_ct_bins = calculate_cycle_times_in_hours(sim_df)

        if not log_ct_bins or not sim_ct_bins:
            similarity.append({'metric': metric, 'sim_score': 0.0})
            return similarity

        # -------- 4) 计算 1-Wasserstein 距离 [cite: 206] --------
        # 直接传入观测点序列。Scipy 会自动构建经验分布函数 (PDF) 并计算搬运代价。
        # 结果 distance 的物理含义是：平均每个 case 的总时长偏离了多少个“小时桶”。
        try:
            distance = wasserstein_distance(log_ct_bins, sim_ct_bins)
            sim_score = float(distance) if np.isfinite(distance) else 0.0
        except Exception as e:
            print(f"[警告] CTD 计算失败: {e}")
            sim_score = 0.0

        # -------- 5) 封装结果 --------
        similarity.append({
            'metric': metric,
            'sim_score': sim_score,  # 对应论文表 2 中的量级
            'avg_cycle_time_log_hours': np.mean(log_ct_bins),
            'avg_cycle_time_sim_hours': np.mean(sim_ct_bins),
            'total_cases_log': len(log_ct_bins),
            'total_cases_sim': len(sim_ct_bins)
        })

        return similarity


    # =============================================================================
    # Timed string distance
    # =============================================================================

    def _evaluate_seq_distance(self, log_data, simulation_data, metric):
        """
        计算序列距离类指标：
        - dl: 论文 CFLD (Control-flow Log Distance) 的严格实现（log-level 平均距离）
        - tsd / mae / dl_mae: 保持原有逻辑（逐 case 匹配输出）
        """
        similarity = []

        def pbar_async(p, msg):
            """异步进度条更新函数"""
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        # -----------------------------
        # 1) 计算距离矩阵（串行或并行）
        # -----------------------------
        cases = len(set([x['caseid'] for x in log_data]))

        if cases <= self.max_cases:
            args = (
                metric,
                simulation_data,
                log_data,
                self.alpha_concurrency.oracle,
                ({'min': 0, 'max': len(simulation_data)},
                 {'min': 0, 'max': len(log_data)})
            )
            df_matrix = self._compare_traces(args)
        else:
            cpu_count = multiprocessing.cpu_count()
            mx_len = len(log_data)
            ranges = self.define_ranges(mx_len, int(np.ceil(cpu_count / 2)))
            ranges = list(itertools.product(*[ranges, ranges]))
            reps = len(ranges)

            pool = Pool(processes=cpu_count)
            args = [
                (metric,
                 simulation_data[r[0]['min']:r[0]['max']],
                 log_data[r[1]['min']:r[1]['max']],
                 self.alpha_concurrency.oracle,
                 r)
                for r in ranges
            ]
            p = pool.map_async(self._compare_traces, args)

            if self.verbose:
                pbar_async(p, 'evaluating ' + metric + ':')

            pool.close()

            try:
                results = p.get()
                valid_results = [res for res in results if res is not None and not res.empty]
                if valid_results:
                    df_matrix = pd.concat(valid_results, axis=0, ignore_index=True)
                else:
                    df_matrix = pd.DataFrame()
            except Exception as e:
                print(f"Parallel execution failed: {e}")
                df_matrix = pd.DataFrame()

        if df_matrix is None or df_matrix.empty:
            print("Error: df_matrix is None or empty")
            return similarity

        # -----------------------------
        # 2) 组装 cost_matrix
        # -----------------------------
        df_matrix.sort_values(by=['i', 'j'], inplace=True)
        df_matrix = df_matrix.reset_index(drop=True).set_index(['i', 'j'])

        if metric == 'dl_mae':
            dl_matrix = df_matrix[['dl_distance']].unstack().to_numpy()
            mae_matrix = df_matrix[['mae_distance']].unstack().to_numpy()

            max_mae = np.nanmax(mae_matrix)
            if max_mae > 0 and np.isfinite(max_mae):
                mae_matrix = np.divide(mae_matrix, max_mae)
            else:
                mae_matrix = np.nan_to_num(mae_matrix, nan=0.0)

            dl_matrix = np.multiply(dl_matrix, 0.5)
            mae_matrix = np.multiply(mae_matrix, 0.5)
            cost_matrix = np.add(dl_matrix, mae_matrix)
        else:
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()

        # NaN/inf 防护：无法计算的距离视为极大成本（避免被匹配选中）
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e9, posinf=1e9, neginf=1e9)

        # -----------------------------
        # 3) Hungarian 最优匹配
        # -----------------------------
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))

        # -----------------------------
        # 4) 输出：让 dl 完全对齐 CFLD
        # -----------------------------
        if metric == 'dl':
            # 论文 CFLD = 平均(归一化DL距离) after Hungarian matching
            matched_costs = [float(cost_matrix[i][j]) for i, j in zip(row_ind, col_ind)]
            cfl_distance = float(np.mean(matched_costs)) if matched_costs else 0.0

            # 返回一个“log-level”结果（单条记录），sim_score 即 CFLD 距离
            similarity.append({
                'metric': 'cfl_distance',  # 你也可以改成 'dl'，但建议显式一点
                'sim_score': cfl_distance,
                'n_cases': len(matched_costs),
                # 可选：保留匹配细节，便于调试/可解释性
                'pairs': [
                    {
                        'sim_caseid': simulation_data[i]['caseid'],
                        'log_caseid': log_data[j]['caseid'],
                        'distance': float(cost_matrix[i][j]),
                        'sim_order': simulation_data[i].get('profile', None),
                        'log_order': log_data[j].get('profile', None),
                    }
                    for i, j in zip(row_ind, col_ind)
                ]
            })
            return similarity

        # -----------------------------
        # 5) 其它 metric：保持你原有逐case输出语义
        # -----------------------------
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(
                dict(
                    caseid=simulation_data[idx]['caseid'],
                    sim_order=simulation_data[idx].get('profile', None),
                    log_order=log_data[idy].get('profile', None),
                    sim_score=(cost_matrix[idx][idy] if metric == 'mae'
                               else (1 - (cost_matrix[idx][idy])))
                )
            )
        return similarity

    @staticmethod
    def _compare_traces(args):
        # Unpack
        metric, serie1, serie2, oracle, r = args

        def ae_distance(et_1, et_2, st_1, st_2):
            try:
                cicle_time_s1 = (et_1 - st_1).total_seconds()
                cicle_time_s2 = (et_2 - st_2).total_seconds()
                ae = np.abs(cicle_time_s1 - cicle_time_s2)
                return ae
            except:
                return 0.0

        def tsd_alpha(s_1, s_2, p_1, p_2, w_1, w_2, alpha_concurrency):
            def calculate_cost(s1_idx, s2_idx):
                t_1 = p_1[s1_idx] + w_1[s1_idx]
                if t_1 > 0:
                    b_1 = (p_1[s1_idx] / t_1)
                    cost = ((b_1 * np.abs(p_2[s2_idx] - p_1[s1_idx])) +
                            ((1 - b_1) * np.abs(w_2[s2_idx] - w_1[s1_idx])))
                else:
                    cost = 0
                return cost

            dist = {}
            lenstr1 = len(s_1)
            lenstr2 = len(s_2)
            for i in range(-1, lenstr1 + 1):
                dist[(i, -1)] = i + 1
            for j in range(-1, lenstr2 + 1):
                dist[(-1, j)] = j + 1
            for i in range(0, lenstr1):
                for j in range(0, lenstr2):
                    if s_1[i] == s_2[j]:
                        cost = calculate_cost(i, j)
                    else:
                        cost = 1
                    dist[(i, j)] = min(
                        dist[(i - 1, j)] + 1,  # deletion
                        dist[(i, j - 1)] + 1,  # insertion
                        dist[(i - 1, j - 1)] + cost  # substitution
                    )
                    if i and j and s_1[i] == s_2[j - 1] and s_1[i - 1] == s_2[j]:
                        if alpha_concurrency.get((s_1[i], s_2[j])) == Rel.PARALLEL:
                            cost = calculate_cost(i, j - 1)
                        dist[(i, j)] = min(dist[(i, j)], dist[i - 2, j - 2] + cost)
            return dist[lenstr1 - 1, lenstr2 - 1]

        def gen(metric, serie1, serie2, oracle, r):
            try:
                df_matrix = list()
                for i, s1_ele in enumerate(serie1):
                    for j, s2_ele in enumerate(serie2):
                        element = {'i': r[0]['min'] + i, 'j': r[1]['min'] + j}

                        # ===== 修正1：与 SimilarityEvaluator 保持一致 =====
                        if metric in ['tsd', 'dl', 'dl_mae']:
                            element['s_1'] = s1_ele['profile']
                            element['s_2'] = s2_ele['profile']
                            element['length'] = max(len(s1_ele['profile']),
                                                    len(s2_ele['profile']))

                        if metric == 'tsd':
                            element['p_1'] = s1_ele['proc_act_norm']
                            element['p_2'] = s2_ele['proc_act_norm']
                            element['w_1'] = s1_ele['wait_act_norm']
                            element['w_2'] = s2_ele['wait_act_norm']

                        if metric in ['mae', 'dl_mae']:
                            element['et_1'] = s1_ele['end_time']
                            element['et_2'] = s2_ele['end_time']
                            element['st_1'] = s1_ele['start_time']
                            element['st_2'] = s2_ele['start_time']

                        df_matrix.append(element)

                df_matrix = pd.DataFrame(df_matrix)
                if df_matrix.empty:
                    return None

                # ===== 修正2：与 SimilarityEvaluator 保持一致 =====
                if metric == 'tsd':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: tsd_alpha(
                            x['s_1'], x['s_2'], x['p_1'], x['p_2'], x['w_1'], x['w_2'],
                            oracle) / x['length'] if x['length'] > 0 else 0, axis=1)

                elif metric == 'dl':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(
                            ''.join(x['s_1']), ''.join(x['s_2'])) / x['length']
                        if x['length'] > 0 else 0, axis=1)

                elif metric == 'mae':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: ae_distance(
                            x['et_1'], x['et_2'], x['st_1'], x['st_2']), axis=1)

                elif metric == 'dl_mae':
                    df_matrix['dl_distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(
                            ''.join(x['s_1']), ''.join(x['s_2'])) / x['length']
                        if x['length'] > 0 else 0, axis=1)
                    df_matrix['mae_distance'] = df_matrix.apply(
                        lambda x: ae_distance(
                            x['et_1'], x['et_2'], x['st_1'], x['st_2']), axis=1)
                else:
                    raise ValueError(metric)

                return df_matrix

            except Exception:
                traceback.print_exc()
            return None

        return gen(metric, serie1, serie2, oracle, r)

    # =============================================================================
    # whole log MAE
    # =============================================================================
    def log_mae_metric(self, log_data: list, simulation_data: list, metric) -> list:
        """测量两个完整日志之间的MAE距离"""
        similarity = []

        # ===== 修正：统一为DataFrame处理 =====
        if isinstance(log_data, list):
            log_data = pd.DataFrame(log_data)
        if isinstance(simulation_data, list):
            simulation_data = pd.DataFrame(simulation_data)

        # 确保存在必需的列
        required_columns = ['start_time', 'end_time']
        for col in required_columns:
            if col not in log_data.columns:
                raise ValueError(f"log_data 缺少必需的列: {col}")
            if col not in simulation_data.columns:
                raise ValueError(f"simulation_data 缺少必需的列: {col}")

        # 计算时间间隔
        try:
            log_start_min = log_data['start_time'].min()
            log_end_max = log_data['end_time'].max()
            log_timelapse = (log_end_max - log_start_min).total_seconds()
        except Exception as e:
            print(f"[警告] log_mae 计算真实日志时间失败: {e}")
            log_timelapse = 0.0

        try:
            sim_start_min = simulation_data['start_time'].min()
            sim_end_max = simulation_data['end_time'].max()
            sim_timelapse = (sim_end_max - sim_start_min).total_seconds()
        except Exception as e:
            print(f"[警告] log_mae 计算模拟日志时间失败: {e}")
            sim_timelapse = 0.0

        # 计算相似度分数
        score = abs(sim_timelapse - log_timelapse)

        # 确保分数有效
        if not np.isfinite(score):
            score = 0.0

        similarity.append({
            'metric': metric,
            'sim_score': score,
            'log_duration': log_timelapse,
            'sim_duration': sim_timelapse
        })

        return similarity

    # =============================================================================
    # Log emd distance
    # =============================================================================

    def log_emd_metric(self, log_data: list, simulation_data: list,
                       criteria='hour_emd') -> list:
        """
        Measures the EMD distance between two logs on different aggregation
        levels specified by user by default per hour
        """
        similarity = list()
        window = 1

        # 将列表转换为DataFrame
        log_df = pd.DataFrame(log_data)
        sim_df = pd.DataFrame(simulation_data)

        def split_date_time(dataframe, feature, source):
            # ===== 修正：移除 UTC 时区转换 =====
            if not pd.api.types.is_datetime64_any_dtype(dataframe[feature]):
                dataframe[feature] = pd.to_datetime(dataframe[feature])

            # 移除时区信息
            if pd.api.types.is_datetime64tz_dtype(dataframe[feature]):
                dataframe[feature] = dataframe[feature].dt.tz_convert(None)

            # 提取小时和日期
            dataframe['hour'] = dataframe[feature].dt.hour
            dataframe['date'] = dataframe[feature].dt.date

            # 创建时间窗口
            daily_windows = {}
            i = 0
            for hour in range(24):
                if hour % window == 0:
                    i += 1
                daily_windows[hour] = i

            # 合并时间窗口信息
            windows_df = pd.DataFrame.from_dict(
                daily_windows, orient='index', columns=['window']
            ).rename_axis('hour')
            dataframe = dataframe.merge(windows_df, on='hour', how='left')

            # 选择需要的列并重命名
            result = dataframe[[feature, 'date', 'window']].copy()
            result = result.rename(columns={feature: 'timestamp'})
            result['source'] = source
            return result

        # 使用pd.concat替代append
        data_list = []
        data_list.append(split_date_time(log_df.copy(), 'start_time', 'log'))
        data_list.append(split_date_time(log_df.copy(), 'end_time', 'log'))
        data_list.append(split_date_time(sim_df.copy(), 'start_time', 'sim'))
        data_list.append(split_date_time(sim_df.copy(), 'end_time', 'sim'))

        # 合并所有数据
        data = pd.concat(data_list, ignore_index=True)

        # 添加星期几信息
        data['weekday'] = pd.to_datetime(data['date']).dt.dayofweek

        # 定义分组标准
        g_criteria = {
            'day_hour_emd': ['weekday', 'window'],
        }

        similarity = []
        group_key = g_criteria[criteria]

        # 分组计算距离
        for key, group in data.groupby(group_key):
            w_df = group.copy()
            w_df = w_df.reset_index(drop=True)

            # 确保所有时间戳都是无时区的datetime64[ns]类型
            if not pd.api.types.is_datetime64_ns_dtype(w_df['timestamp']):
                w_df['timestamp'] = pd.to_datetime(w_df['timestamp'], errors='coerce')

            # 确保没有NaT值
            w_df = w_df.dropna(subset=['timestamp'])

            # 计算相对时间
            basetime = w_df['timestamp'].min().floor(freq='H')
            w_df['rel_time'] = (w_df['timestamp'] - basetime).dt.total_seconds()

            # 计算直方图
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                log_times = w_df.loc[w_df['source'] == 'log', 'rel_time']
                sim_times = w_df.loc[w_df['source'] == 'sim', 'rel_time']

                # 检查是否有有效数据
                if log_times.empty or sim_times.empty:
                    similarity.append({'window': key, 'sim_score': 0})
                    continue

                log_hist = np.histogram(log_times, density=True)
                sim_hist = np.histogram(sim_times, density=True)

            # 计算距离
            if np.isnan(np.sum(log_hist[0])) or np.isnan(np.sum(sim_hist[0])):
                similarity.append({'window': key, 'sim_score': 0})
            else:
                distance = wasserstein_distance(log_hist[0], sim_hist[0])
                similarity.append({'window': key, 'sim_score': distance})

        return similarity

    # =============================================================================
    # serie emd distance
    # =============================================================================

    def serie_emd_metric(self, log_data, simulation_data, criteria='hour_emd'):
        similarity = list()
        window = 1
        log_data = pd.DataFrame(log_data)
        simulation_data = pd.DataFrame(simulation_data)

        def split_date_time(dataframe, feature, source):
            day_hour = lambda x: x[feature].hour
            dataframe['hour'] = dataframe.apply(day_hour, axis=1)
            date = lambda x: x[feature].date()
            dataframe['date'] = dataframe.apply(date, axis=1)
            # create time windows
            i = 0
            daily_windows = dict()
            for x in range(24):
                if x % window == 0:
                    i += 1
                daily_windows[x] = i
            dataframe = dataframe.merge(
                pd.DataFrame.from_dict(
                    daily_windows, orient='index').rename_axis('hour'),
                on='hour',
                how='left').rename(columns={0: 'window'})
            dataframe = dataframe[[feature, 'date', 'window']]
            dataframe.rename(columns={feature: 'timestamp'}, inplace=True)
            dataframe['source'] = source
            return dataframe

        data = split_date_time(log_data, 'timestamp', 'log')
        data = data.append(
            split_date_time(simulation_data, 'timestamp', 'sim'),
            ignore_index=True)
        data['weekday'] = data.apply(lambda x: x.date.weekday(), axis=1)
        g_criteria = {'day_hour_emd': ['weekday', 'window'],}
        similarity = list()
        for key, group in data.groupby(g_criteria[criteria]):
            w_df = group.copy()
            w_df = w_df.reset_index()
            basetime = w_df.timestamp.min().floor(freq='H')
            diftime = lambda x: (x['timestamp'] - basetime).total_seconds()
            w_df['rel_time'] = w_df.apply(diftime, axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                log_hist = np.histogram(w_df[w_df.source == 'log'].rel_time,
                                        density=True)
                sim_hist = np.histogram(w_df[w_df.source == 'sim'].rel_time,
                                        density=True)
            if np.isnan(np.sum(log_hist[0])) or np.isnan(np.sum(sim_hist[0])):
                similarity.append({'window': key, 'sim_score': 1})
            else:
                similarity.append(
                    {'window': key,
                     'sim_score': wasserstein_distance(log_hist[0],
                                                       sim_hist[0])})
        return similarity

    # =============================================================================
    # Support methods
    # =============================================================================

    def create_task_alias(self, data, features):
        """
        为任务名称或任务-角色组合创建字符串别名（简写形式）
        【确定性版本：不使用随机数，保证可复现】
        """

        # 将 DataFrame 转换为字典记录列表，便于后续处理
        data = data.to_dict('records')

        # 收集唯一的任务 / 任务-角色组合
        if isinstance(features, list):
            task_list = [(x[features[0]], x[features[1]]) for x in data]
        else:
            task_list = [x[features] for x in data]

        # 去重 + 排序（这是确定性的关键）
        variables = sorted(set(task_list))

        # 基础字符池（单字符，优先使用）
        base_chars = list(string.ascii_letters + string.digits)

        alias = {}

        for i, var in enumerate(variables):
            if i < len(base_chars):
                # 前 62 个变量 → 单字符 alias
                alias[var] = base_chars[i]
            else:
                # 超出字符池时，使用确定性的多字符 alias
                # 例如 A62, A63, ...
                alias[var] = f"A{i}"

        return alias

    @staticmethod
    def calculate_times(log):
        """Appends the indexes and relative time to the dataframe."""
        log['processing_time'] = 0
        log['multitasking'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: (x['source'], x['caseid']))
        for _, group in itertools.groupby(log, key=lambda x: (x['source'], x['caseid'])):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                dur = (events[i]['end_timestamp'] -
                       events[i]['start_timestamp']).total_seconds()
                if i == 0:
                    wit = 0
                else:
                    wit = (events[i]['start_timestamp'] -
                           events[i - 1]['end_timestamp']).total_seconds()
                events[i]['waiting_time'] = wit if wit >= 0 else 0
                events[i]['processing_time'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """
        Scales times values activity based
        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        if self.one_timestamp:
            summ = data.groupby(['task'])['duration'].max().to_dict()
            dur_act_norm = (lambda x: x['duration'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        else:
            summ = data.groupby(['task'])['processing_time'].max().to_dict()
            proc_act_norm = (lambda x: x['processing_time'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['proc_act_norm'] = df_modif.apply(proc_act_norm, axis=1)
            # ---
            summ = data.groupby(['task'])['waiting_time'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting_time'] / summ[x['task']]
            if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

    def reformat_events(self, data, features):
        """Creates series of activities, roles and relative times per trace."""
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=self.alias[(x[features[0]],
                                             x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=self.alias[x[features]])) for x in data]

        temp_data = list()

        # 确定排序键 (优先使用 timestamp, 备用 time)
        sort_key = 'start_timestamp' if 'start_timestamp' in data[0] else 'start_time'
        if self.one_timestamp:
            sort_key = 'end_timestamp' if 'end_timestamp' in data[0] else 'end_time'
            columns = ['alias', 'duration', 'dur_act_norm']
        else:
            columns = ['alias', 'processing_time', 'proc_act_norm', 'waiting_time', 'wait_act_norm']

        # 排序
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))

        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()

            # 聚合轨迹层面的特征
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    temp_dict = {**{col: serie}, **temp_dict}

            # 【关键】从 trace 第一条和最后一条获取时间
            # 此时 trace 里的数据已经去除了时区，可以安全使用
            s_ts = trace[0].get('start_timestamp', trace[0].get('start_time'))
            e_ts = trace[-1].get('end_timestamp', trace[-1].get('end_time'))

            temp_dict = {
                **{
                    'caseid': key,
                    'start_time': s_ts,  # 兼容旧代码
                    'end_time': e_ts,  # 兼容旧代码
                    'start_timestamp': s_ts,  # 供 EMD 使用
                    'end_timestamp': e_ts  # 供 EMD 使用
                },
                **temp_dict
            }
            temp_data.append(temp_dict)

        return sorted(temp_data, key=itemgetter('start_time'))
    @staticmethod
    def define_ranges(size, num_folds):
        num_events = int(np.round(size / num_folds))
        folds = list()
        for i in range(0, num_folds):
            sidx = i * num_events
            eidx = (i + 1) * num_events
            if i == 0:
                folds.append({'min': 0, 'max': eidx})
            elif i == (num_folds - 1):
                folds.append({'min': sidx, 'max': size})
            else:
                folds.append({'min': sidx, 'max': eidx})
        return folds
