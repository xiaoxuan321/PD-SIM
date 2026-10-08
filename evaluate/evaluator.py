import copy
import itertools
import string
from collections import Counter
from operator import itemgetter

import jellyfish as jf
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance


class Evaluator:
    """
    严格对齐 Chapela-Campa et al. (2023, 2025) 的 7 个指标实现：
    - CFLD
    - NGD
    - AED
    - CED
    - RED
    - CAR
    - CTD

    关键修正：
    1) 同时保留事件级数据(event-level)与案例级数据(case-level)
    2) 控制流指标走 case-level；时间/拥塞指标走 event-level
    3) AED/CED/RED/CAR 使用 EMD 语义，并按论文缩放为“每个原始观测平均移动多少个 bin”
    4) CTD 使用 1-WD 比较 cycle-time histogram
    """

    CONTROL_FLOW_METRICS = {"cfld", "dl", "ngd_2", "ngd_3"}
    TIME_AND_CONGESTION_METRICS = {"aed", "ced", "red", "car", "ctd"}

    def __init__(self, log_data, simulation_data, settings, max_cases=500, dtype='log', seed=42):
        self.dtype = dtype
        self.seed = seed
        self.max_cases = max_cases
        self.one_timestamp = settings['read_options']['one_timestamp']

        np.random.seed(seed)

        # 原始输入（深拷贝）
        self.raw_log_data = copy.deepcopy(log_data)
        self.raw_simulation_data = copy.deepcopy(simulation_data)

        # 预处理后数据
        self.log_event_data = None
        self.sim_event_data = None
        self.log_case_data = None
        self.sim_case_data = None

        self.alias = {}
        self.similarity = {}

        self._preprocess_data(dtype)

    # ---------------------------------------------------------------------
    # 预处理
    # ---------------------------------------------------------------------
    def _preprocess_data(self, dtype):
        if dtype != 'log':
            raise ValueError(f"Unsupported dtype: {dtype}")
        self._preprocess_log()

    def _preprocess_log(self):
        # 1) 事件级数据：严格要求 start/end 同时存在
        self.log_event_data = self._clean_event_log(self.raw_log_data, source="log")
        self.sim_event_data = self._clean_event_log(self.raw_simulation_data, source="simulation")

        # 2) 为控制流指标准备 case-level trace
        merged = pd.concat([self.log_event_data, self.sim_event_data], ignore_index=True)
        self.alias = self.create_task_alias(merged, 'task')

        self.log_case_data = self.reformat_traces(self.log_event_data, 'task')
        self.sim_case_data = self.reformat_traces(self.sim_event_data, 'task')

    def _clean_event_log(self, data, source: str) -> pd.DataFrame:
        """
        严格论文版：输入必须是 activity-instance log，
        至少包含 caseid, task, start_timestamp, end_timestamp。
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # 兼容列名
        rename_map = {
            'start_time': 'start_timestamp',
            'end_time': 'end_timestamp',
            'resource': 'user',
        }
        df = df.rename(columns=rename_map)

        required = ['caseid', 'task', 'start_timestamp', 'end_timestamp']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{source}] 严格论文版要求事件日志至少包含 {required}，当前缺失 {missing}"
            )

        # 基本类型
        df['caseid'] = df['caseid'].astype(str)
        df['task'] = df['task'].astype(str)
        df['source'] = source

        # 时间解析 + 去时区
        for col in ['start_timestamp', 'end_timestamp']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = self._strip_tz(df[col])

        # 严格模式：缺 start/end 的事件直接丢弃
        df = df.dropna(subset=['caseid', 'task', 'start_timestamp', 'end_timestamp']).copy()

        # 严格模式：结束早于开始的异常事件丢弃
        df = df[df['end_timestamp'] >= df['start_timestamp']].copy()

        # 统一附加 start_time / end_time 别名，避免旧代码依赖
        df['start_time'] = df['start_timestamp']
        df['end_time'] = df['end_timestamp']

        # 排序
        df = df.sort_values(by=['caseid', 'end_timestamp', 'start_timestamp', 'task']).reset_index(drop=True)
        return df

    @staticmethod
    def _strip_tz(series: pd.Series) -> pd.Series:
        series = pd.to_datetime(series, errors='coerce')
        if hasattr(series, 'dt') and getattr(series.dt, 'tz', None) is not None:
            series = series.dt.tz_localize(None)
        return series

    # ---------------------------------------------------------------------
    # 指标调度
    # ---------------------------------------------------------------------
    def measure_distance(self, metric, verbose=False):
        self.verbose = verbose
        evaluator = self._get_evaluator(metric)

        if metric in self.CONTROL_FLOW_METRICS:
            left = self.log_case_data
            right = self.sim_case_data
            empty = (len(left) == 0 or len(right) == 0)
        else:
            left = self.log_event_data
            right = self.sim_event_data
            empty = (left.empty or right.empty)

        if empty:
            self.similarity = {'metric': metric, 'sim_val': 0.0}
            return

        distance = evaluator(left, right, metric)
        self.similarity = {
            'metric': metric,
            'sim_val': float(np.mean([x['sim_score'] for x in distance])) if distance else 0.0
        }

    def _get_evaluator(self, metric):
        if metric in ['cfld', 'dl']:
            return self.cfld_metric
        elif metric == 'ngd_2':
            return lambda l1, l2, m: self.ngd_n_metric(l1, l2, n=2)
        elif metric == 'ngd_3':
            return lambda l1, l2, m: self.ngd_n_metric(l1, l2, n=3)
        elif metric == 'aed':
            return self.aed_metric
        elif metric == 'ced':
            return self.ced_metric
        elif metric == 'red':
            return self.red_metric
        elif metric == 'car':
            return self.car_metric
        elif metric == 'ctd':
            return self.ctd_metric
        else:
            raise ValueError(f"Unknown metric: {metric}")


    # ---------------------------------------------------------------------
    # CFLD (重写版：支持自动对齐与多轮采样)
    # ---------------------------------------------------------------------
    def cfld_metric(self, log_cases: list, sim_cases: list, metric='cfld', n_iterations=5) -> list:
        """
        CFLD (Academic Version):
        - 支持案例数量不等时的自动下采样。
        - 采用 Bootstrapping 方法重复计算 n_iterations 次并取均值。
        - 保持与 Chapela-Campa 论文一致的 Normalized Damerau-Levenshtein 逻辑。
        """
        import random

        n_log = len(log_cases)
        n_sim = len(sim_cases)
        n_min = min(n_log, n_sim)

        if n_log == 0 or n_sim == 0:
            return [{'metric': metric, 'sim_score': 0.0}]

        if n_log != n_sim:
            print(f"\n[学术对齐] 检测到案例数不匹配 (Real:{n_log}, Sim:{n_sim})")
            print(f"[策略] 自动采样至较小规模: {n_min}, 执行 {n_iterations} 轮计算取均值...")

        # 1. 构建全局字符映射 (确保所有轮次的字符含义一致)
        all_activities = set()
        for trace in log_cases + sim_cases:
            all_activities.update(trace['activities'])
        all_acts_sorted = sorted(list(all_activities))

        # 使用偏移量确保映射到有效的 Unicode 字符
        act_to_char = {act: chr(0x4E00 + i) for i, act in enumerate(all_acts_sorted)}

        def to_char_string(seq):
            return "".join(act_to_char[a] for a in seq)

        # 2. 预处理所有轨迹为字符字符串以提高计算速度
        log_strs_all = [to_char_string(t['activities']) for t in log_cases]
        sim_strs_all = [to_char_string(t['activities']) for t in sim_cases]

        iteration_scores = []

        # 3. 开始多轮采样计算
        for r in range(n_iterations):
            # 固定每轮的种子，但确保每轮种子不同 (基于初始 seed + 轮数)
            random.seed(self.seed + r)

            # 随机采样
            current_log_strs = random.sample(log_strs_all, n_min)
            current_sim_strs = random.sample(sim_strs_all, n_min)

            # 构建成本矩阵 (n_min x n_min)
            cost_matrix = np.zeros((n_min, n_min), dtype=float)
            for i, s_sim in enumerate(current_sim_strs):
                for j, s_log in enumerate(current_log_strs):
                    l1, l2 = len(s_sim), len(s_log)
                    if l1 == 0 and l2 == 0:
                        dist = 0.0
                    else:
                        # 计算归一化的 Damerau-Levenshtein 距离
                        raw_dl = jf.damerau_levenshtein_distance(s_sim, s_log)
                        dist = raw_dl / max(l1, l2)
                    cost_matrix[i, j] = dist

            # 匈牙利指派算法求解最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_costs = cost_matrix[row_ind, col_ind]
            iteration_scores.append(np.mean(matched_costs))

            if self.verbose:
                print(f"  > 第 {r + 1} 轮 CFLD: {iteration_scores[-1]:.4f}")

        # 4. 计算多轮平均值
        final_cfld_distance = float(np.mean(iteration_scores))

        return [{
            'metric': metric,
            'sim_score': final_cfld_distance,
            'n_cases_matched': n_min,
            'iterations': n_iterations,
            'std_dev': float(np.std(iteration_scores))  # 额外返回标准差，对论文分析很有用
        }]
    # ---------------------------------------------------------------------
    # NGD
    # ---------------------------------------------------------------------
    def ngd_n_metric(self, log_cases: list, sim_cases: list, n: int = 2) -> list:
        """
        NGD:
        - case-level activity sequence
        - 首尾各补 n-1 个 dummy activities
        - 统计 n-gram 频率
        - sum(abs(freq1-freq2)) / sum(freq1+freq2)
        """
        def extract_ngrams(cases, ngram_n):
            ngrams = []
            for trace in cases:
                activities = trace['activities']
                padded = ['__START__'] * (ngram_n - 1) + activities + ['__END__'] * (ngram_n - 1)
                for i in range(len(padded) - ngram_n + 1):
                    ngrams.append(tuple(padded[i:i + ngram_n]))
            return ngrams

        ngrams_log = extract_ngrams(log_cases, n)
        ngrams_sim = extract_ngrams(sim_cases, n)

        if len(ngrams_log) == 0 and len(ngrams_sim) == 0:
            return [{
                'metric': f'ngd_{n}',
                'sim_score': 0.0,
                'total_ngrams_log': 0,
                'total_ngrams_sim': 0,
                'unique_ngrams': 0
            }]

        freq_log = Counter(ngrams_log)
        freq_sim = Counter(ngrams_sim)
        all_ngrams = set(freq_log.keys()) | set(freq_sim.keys())

        total_diff = 0
        total_count = sum(freq_log.values()) + sum(freq_sim.values())

        for ng in all_ngrams:
            total_diff += abs(freq_log.get(ng, 0) - freq_sim.get(ng, 0))

        ngd_score = (total_diff / total_count) if total_count > 0 else 0.0

        return [{
            'metric': f'ngd_{n}',
            'sim_score': float(ngd_score),
            'total_ngrams_log': len(ngrams_log),
            'total_ngrams_sim': len(ngrams_sim),
            'unique_ngrams': len(all_ngrams)
        }]

    # ---------------------------------------------------------------------
    # AED
    # ---------------------------------------------------------------------
    def aed_metric(self, log_df: pd.DataFrame, sim_df: pd.DataFrame, metric='aed') -> list:
        """
        AED:
        - event-level
        - 收集所有事件 start/end timestamp
        - 按绝对小时桶计数
        - EMD（含 extra-mass penalty）/ 原始日志观测数
        """
        log_times = self._collect_event_times(log_df)
        sim_times = self._collect_event_times(sim_df)

        if log_times.empty and sim_times.empty:
            return [{'metric': metric, 'sim_score': 0.0, 'time_bins': 0}]

        log_counts, sim_counts, n_bins = self._hourly_counts_over_union_timeline(log_times, sim_times)
        distance = self._scaled_emd_time_series(log_counts, sim_counts, original_obs=int(log_counts.sum()))

        return [{
            'metric': metric,
            'sim_score': distance,
            'total_events_log': int(log_counts.sum()),
            'total_events_sim': int(sim_counts.sum()),
            'time_bins': n_bins
        }]

    # ---------------------------------------------------------------------
    # CED
    # ---------------------------------------------------------------------
    def ced_metric(self, log_df: pd.DataFrame, sim_df: pd.DataFrame, metric='ced') -> list:
        """
        CED:
        - event-level
        - 收集所有事件的 start/end timestamps
        - 先按 weekday 划分
        - 再在每个 weekday 内按 24 小时桶计数
        - 对 7 个 weekday 分别计算 raw EMD
        - 先对 7 个 raw EMD 求平均
        - 最后整体除以原始日志总观测数（而不是按每天分别缩放）

        返回:
        - sim_score: 与论文实验报告风格一致的 scaled CED
        - ced_raw: 未缩放的原始 CED
        """
        log_times = self._collect_event_times(log_df)
        sim_times = self._collect_event_times(sim_df)

        if log_times.empty and sim_times.empty:
            return [{
                'metric': metric,
                'sim_score': 0.0,
                'ced_raw': 0.0,
                'days_compared': 0,
                'total_obs_log': 0,
                'total_obs_sim': 0
            }]

        daily_raw_emd = []

        for weekday in range(7):
            log_day = log_times[log_times.dt.weekday == weekday]
            sim_day = sim_times[sim_times.dt.weekday == weekday]

            # 每个 weekday 内按 24 小时统计
            log_counts = (
                np.bincount(log_day.dt.hour.to_numpy(), minlength=24).astype(float)
                if len(log_day) > 0 else np.zeros(24, dtype=float)
            )
            sim_counts = (
                np.bincount(sim_day.dt.hour.to_numpy(), minlength=24).astype(float)
                if len(sim_day) > 0 else np.zeros(24, dtype=float)
            )

            # 按论文定义：先算每天的 raw EMD
            raw_emd = self._emd_1d_with_extra_mass_penalty(log_counts, sim_counts)
            daily_raw_emd.append(float(raw_emd))

        # 先求 7 天 raw EMD 的平均
        ced_raw = float(np.mean(daily_raw_emd))

        # 再整体按原始日志总观测数缩放
        total_original_obs = max(int(len(log_times)), 1)
        ced_scaled = ced_raw / total_original_obs

        return [{
            'metric': metric,
            'sim_score': ced_scaled,
            'ced_raw': ced_raw,
            'days_compared': 7,
            'total_obs_log': int(len(log_times)),
            'total_obs_sim': int(len(sim_times))
        }]
    # ---------------------------------------------------------------------
    # RED
    # ---------------------------------------------------------------------
    def red_metric(self, log_df: pd.DataFrame, sim_df: pd.DataFrame, metric='red') -> list:
        """
        RED:
        - event-level
        - 对每个事件，计算 start-arrival 和 end-arrival
        - arrival = 同 case 最早开始时间
        - 按完整小时 floor 到 bin
        - EMD（含 extra-mass penalty）/ 原始日志观测数
        """
        log_rel = self._relative_event_hour_bins(log_df)
        sim_rel = self._relative_event_hour_bins(sim_df)

        if len(log_rel) == 0 and len(sim_rel) == 0:
            return [{'metric': metric, 'sim_score': 0.0, 'num_bins': 0}]

        max_bin = int(max(log_rel.max(initial=0), sim_rel.max(initial=0)))
        log_counts = np.bincount(log_rel, minlength=max_bin + 1).astype(float)
        sim_counts = np.bincount(sim_rel, minlength=max_bin + 1).astype(float)

        distance = self._scaled_emd_time_series(log_counts, sim_counts, original_obs=int(log_counts.sum()))

        return [{
            'metric': metric,
            'sim_score': distance,
            'total_obs_log': int(log_counts.sum()),
            'total_obs_sim': int(sim_counts.sum()),
            'num_bins': int(max_bin + 1)
        }]

    # ---------------------------------------------------------------------
    # CAR
    # ---------------------------------------------------------------------
    def car_metric(self, log_df: pd.DataFrame, sim_df: pd.DataFrame, metric='car') -> list:
        """
        CAR:
        - event-level
        - 每个 case 只保留最早 start_timestamp
        - 按绝对小时桶计数
        - EMD（含 extra-mass penalty）/ 原始日志 case 数
        """
        log_arrivals = self._case_arrival_times(log_df)
        sim_arrivals = self._case_arrival_times(sim_df)

        if log_arrivals.empty and sim_arrivals.empty:
            return [{'metric': metric, 'sim_score': 0.0, 'time_bins': 0}]

        log_counts, sim_counts, n_bins = self._hourly_counts_over_union_timeline(log_arrivals, sim_arrivals)
        distance = self._scaled_emd_time_series(log_counts, sim_counts, original_obs=int(log_counts.sum()))

        return [{
            'metric': metric,
            'sim_score': distance,
            'total_cases_log': int(log_counts.sum()),
            'total_cases_sim': int(sim_counts.sum()),
            'time_bins': n_bins
        }]

    # ---------------------------------------------------------------------
    # CTD
    # ---------------------------------------------------------------------
    def ctd_metric(self, log_df: pd.DataFrame, sim_df: pd.DataFrame, metric='ctd') -> list:
        """
        CTD:
        - event-level
        - 每个 case 取 earliest start 与 latest end
        - floor 到完整小时
        - 构造 histogram
        - 用 1-Wasserstein Distance 比较 histogram
        """
        log_ct = self._case_cycle_time_hour_bins(log_df)
        sim_ct = self._case_cycle_time_hour_bins(sim_df)

        if len(log_ct) == 0 and len(sim_ct) == 0:
            return [{'metric': metric, 'sim_score': 0.0, 'num_bins': 0}]

        max_bin = int(max(log_ct.max(initial=0), sim_ct.max(initial=0)))
        log_hist = np.bincount(log_ct, minlength=max_bin + 1).astype(float)
        sim_hist = np.bincount(sim_ct, minlength=max_bin + 1).astype(float)

        positions = np.arange(max_bin + 1, dtype=float)

        # 论文语义：比较 empirical PDF -> 使用归一化频率
        log_weights = log_hist / log_hist.sum() if log_hist.sum() > 0 else np.zeros_like(log_hist)
        sim_weights = sim_hist / sim_hist.sum() if sim_hist.sum() > 0 else np.zeros_like(sim_hist)

        if log_weights.sum() == 0 and sim_weights.sum() == 0:
            distance = 0.0
        elif log_weights.sum() == 0 or sim_weights.sum() == 0:
            distance = float(max_bin if max_bin > 0 else 0.0)
        else:
            distance = float(
                wasserstein_distance(
                    positions,
                    positions,
                    u_weights=log_weights,
                    v_weights=sim_weights
                )
            )

        return [{
            'metric': metric,
            'sim_score': distance,
            'total_cases_log': int(log_hist.sum()),
            'total_cases_sim': int(sim_hist.sum()),
            'num_bins': int(max_bin + 1)
        }]

    # ---------------------------------------------------------------------
    # 事件级辅助函数
    # ---------------------------------------------------------------------
    def _collect_event_times(self, df: pd.DataFrame) -> pd.Series:
        times = []
        if 'start_timestamp' in df.columns:
            times.append(df['start_timestamp'].dropna())
        if 'end_timestamp' in df.columns:
            times.append(df['end_timestamp'].dropna())
        if not times:
            return pd.Series([], dtype='datetime64[ns]')
        return pd.concat(times, ignore_index=True)

    def _case_arrival_times(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('caseid')['start_timestamp'].min().dropna()

    def _case_cycle_time_hour_bins(self, df: pd.DataFrame) -> np.ndarray:
        agg = df.groupby('caseid').agg(
            case_start=('start_timestamp', 'min'),
            case_end=('end_timestamp', 'max')
        ).dropna()

        if agg.empty:
            return np.array([], dtype=int)

        durations = (agg['case_end'] - agg['case_start']).dt.total_seconds()
        durations = durations[durations >= 0]
        return np.floor(durations / 3600.0).astype(int).to_numpy()

    def _relative_event_hour_bins(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=int)

        arrivals = df.groupby('caseid')['start_timestamp'].min().rename('arrival')
        merged = df.merge(arrivals, left_on='caseid', right_index=True, how='left')

        start_rel = np.floor(
            (merged['start_timestamp'] - merged['arrival']).dt.total_seconds() / 3600.0
        ).astype('Int64')
        end_rel = np.floor(
            (merged['end_timestamp'] - merged['arrival']).dt.total_seconds() / 3600.0
        ).astype('Int64')

        rel = pd.concat([start_rel, end_rel], ignore_index=True).dropna()
        rel = rel[rel >= 0]
        return rel.astype(int).to_numpy()

    def _hourly_counts_over_union_timeline(self, s1: pd.Series, s2: pd.Series):
        s1 = pd.to_datetime(s1, errors='coerce').dropna()
        s2 = pd.to_datetime(s2, errors='coerce').dropna()

        if s1.empty and s2.empty:
            return np.array([], dtype=float), np.array([], dtype=float), 0

        all_times = pd.concat([s1, s2], ignore_index=True)
        min_hour = all_times.min().floor('h')
        max_hour = all_times.max().floor('h')

        n_bins = int(((max_hour - min_hour).total_seconds() // 3600) + 1)

        idx1 = ((s1.dt.floor('h') - min_hour).dt.total_seconds() // 3600).astype(int)
        idx2 = ((s2.dt.floor('h') - min_hour).dt.total_seconds() // 3600).astype(int)

        c1 = np.bincount(idx1.to_numpy(), minlength=n_bins).astype(float)
        c2 = np.bincount(idx2.to_numpy(), minlength=n_bins).astype(float)

        return c1, c2, n_bins

    # ---------------------------------------------------------------------
    # EMD / 1WD 辅助函数
    # ---------------------------------------------------------------------
    def _scaled_emd_time_series(self, x: np.ndarray, y: np.ndarray, original_obs: int) -> float:
        """
        论文语义：
        - 对 AED/CED/RED/CAR 使用 EMD
        - 再除以 original log 中的观测数，使其可解释为“平均每个原始观测移动多少个 bin”
        """
        raw = self._emd_1d_with_extra_mass_penalty(x, y)
        denom = max(int(original_obs), 1)
        return float(raw / denom)

    @staticmethod
    def _emd_1d_with_extra_mass_penalty(x: np.ndarray, y: np.ndarray) -> float:
        """
        一维有序 bins 的 EMD（absolute ground distance）实现：
        - 对共享质量做最优一维运输
        - 对总质量差加入 extra-mass penalty
        - penalty 取 max(1, diameter)，以保证“计数差”也会被惩罚

        说明：
        论文要求的是 EMD，并强调当总质量不同需要对冗余质量加罚。
        这里使用一维 ordered bins 的精确运输 + extra-mass penalty 实现，
        适用于 AED / CED / RED / CAR 这种按时间顺序排列的时间序列。
        """
        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        n = max(len(x), len(y))
        if n == 0:
            return 0.0

        if len(x) < n:
            x = np.pad(x, (0, n - len(x)), constant_values=0.0)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)), constant_values=0.0)

        total_x = float(x.sum())
        total_y = float(y.sum())

        if total_x == 0.0 and total_y == 0.0:
            return 0.0

        # 共享质量的一维最优运输
        i, j = 0, 0
        x_rem = x[0]
        y_rem = y[0]
        transport_cost = 0.0

        while i < n and j < n:
            while i < n and x_rem <= 0:
                i += 1
                if i < n:
                    x_rem = x[i]
            while j < n and y_rem <= 0:
                j += 1
                if j < n:
                    y_rem = y[j]

            if i >= n or j >= n:
                break

            flow = min(x_rem, y_rem)
            transport_cost += flow * abs(i - j)
            x_rem -= flow
            y_rem -= flow

        # 对多余质量加罚
        extra_mass_penalty = max(1.0, float(n - 1))
        penalty_cost = abs(total_x - total_y) * extra_mass_penalty

        return float(transport_cost + penalty_cost)

    # ---------------------------------------------------------------------
    # trace 构造
    # ---------------------------------------------------------------------
    def create_task_alias(self, data: pd.DataFrame, feature='task'):
        task_list = sorted(set(data[feature].astype(str).tolist()))
        base_chars = list(string.ascii_letters + string.digits)

        alias = {}
        for i, task in enumerate(task_list):
            if i < len(base_chars):
                alias[task] = base_chars[i]
            else:
                alias[task] = f"A{i}"
        return alias

    def reformat_traces(self, df: pd.DataFrame, feature='task') -> list:
        """
        为 CFLD / NGD 构造 case-level trace。
        论文 formalization 中 CFLD 使用 end timestamp 顺序，这里按 end_timestamp 排序。
        """
        rows = df.copy()
        rows['alias'] = rows[feature].map(self.alias)

        rows = rows.sort_values(
            by=['caseid', 'end_timestamp', 'start_timestamp', feature]
        ).reset_index(drop=True)

        traces = []
        for caseid, g in rows.groupby('caseid', sort=False):
            activities = g[feature].astype(str).tolist()
            aliases = g['alias'].astype(str).tolist()

            trace = {
                'caseid': str(caseid),
                'activities': activities,          # 原始活动名，供 CFLD / NGD 使用
                'profile': aliases,               # 保留兼容字段
                'start_time': g['start_timestamp'].min(),
                'end_time': g['end_timestamp'].max(),
                'start_timestamp': g['start_timestamp'].min(),
                'end_timestamp': g['end_timestamp'].max(),
            }
            traces.append(trace)

        return sorted(traces, key=itemgetter('start_time'))