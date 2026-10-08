
import warnings
import random
import itertools
from operator import itemgetter
import time
import multiprocessing
from multiprocessing import Pool
import traceback
import string
import copy

from tqdm import tqdm
import numpy as np
import pandas as pd
import jellyfish as jf
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

from analyzers import alpha_oracle as ao
from analyzers.alpha_oracle import Rel


class SimilarityEvaluator():
    """
        This class evaluates the similarity of two event-logs
     """

    def __init__(self, log_data, simulation_data, settings, max_cases=500, dtype='log'):
        """constructor"""
        self.dtype = dtype
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
        data = pd.concat([self.log_data, self.simulation_data], axis=0, ignore_index=True)
        if (('processing_time' not in data.columns) or ('waiting_time' not in data.columns)):
            data = self.calculate_times(data)

            # 检查时间列
            if (('processing_time' not in data.columns) or ('waiting_time' not in data.columns)):
                data = self.calculate_times(data)

            data = self.scaling_data(data)
            self.log_data = data[data.source == 'log']
            self.simulation_data = data[data.source == 'simulation']
            # Ensure that simulation_data has the correct column names
            if 'start_timestamp' in self.simulation_data.columns and 'end_timestamp' in self.simulation_data.columns:
                self.simulation_data.rename(columns={'start_timestamp': 'start_time', 'end_timestamp': 'end_time'},
                                            inplace=True)
            self.alias = self.create_task_alias(data, 'task')

            self.alpha_concurrency = ao.AlphaOracle(self.log_data,
                                                    self.alias,
                                                    self.one_timestamp, True)

            self.log_data = self.reformat_events(self.log_data.to_dict('records'), 'task')
            self.simulation_data = self.reformat_events1(self.simulation_data.to_dict('records'), 'task')

            num_traces = int(len(self.simulation_data) * self.ramp_io_perc)
            self.simulation_data = self.simulation_data[num_traces:-num_traces]
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
            raise ValueError("log_data or simulation_data is empty.")

        if metric in ['day_emd', 'day_hour_emd', 'cal_emd']:
            distance = evaluator(self.log_data,
                                 self.simulation_data,
                                 criteria=metric)
        else:
            distance = evaluator(self.log_data, self.simulation_data, metric)

        self.similarity = {'metric': metric,
                           'sim_val': np.mean(
                               [x['sim_score'] for x in distance])}


    def _get_evaluator(self, metric):
        if self.dtype == 'log':
            if metric in ['tsd', 'dl', 'mae', 'dl_mae']:
                return self._evaluate_seq_distance
            elif metric == 'log_mae':
                return self.log_mae_metric
            elif metric in ['hour_emd', 'day_emd', 'day_hour_emd', 'cal_emd']:
                return self.log_emd_metric
            else:
                raise ValueError(metric)
        elif self.dtype == 'serie':
            if metric in ['hour_emd', 'day_emd', 'day_hour_emd', 'cal_emd']:
                return self.serie_emd_metric
            else:
                raise ValueError(metric)
        else:
            raise ValueError(self.dtype)

# =============================================================================
# Timed string distance
# =============================================================================

    def _evaluate_seq_distance(self, log_data, simulation_data, metric):
        """
        计算时间序列数据之间的相似度（基于字符串距离或动态时间规整等方法）

        Parameters
        ----------
        log_data : list of dict
            真实事件日志数据，每个元素包含'caseid'和'profile'等信息
        simulation_data : list of dict
            模拟生成的事件数据，结构与log_data相同
        metric : str
            使用的相似度度量方法，如'dl_mae'（Damerau-Levenshtein + MAE）或'mae'等

        Returns
        -------
        list of dict
            包含匹配结果和相似度得分的字典列表，每个字典包含：
            - caseid: 案例ID
            - sim_order: 模拟数据中的顺序
            - log_order: 真实日志中的顺序
            - sim_score: 相似度得分（已归一化）
        """
        similarity = list()  # 存储最终相似度结果的列表

        def pbar_async(p, msg):
            """
            异步进度条更新函数（用于并行计算时显示进度）

            Parameters
            ----------
            p : multiprocessing.pool.AsyncResult
                异步任务对象
            msg : str
                进度条描述信息
            """
            pbar = tqdm(total=reps, desc=msg)  # 初始化进度条
            processed = 0
            while not p.ready():  # 当任务未完成时
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)  # 更新进度条
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))  # 完成最后的进度更新
            p.wait()
            pbar.close()

        # 根据数据量决定使用串行还是并行处理
        cases = len(set([x['caseid'] for x in log_data]))  # 计算唯一案例数量

        if cases <= self.max_cases:
            # 串行处理模式（小数据量）
            args = (metric, simulation_data, log_data,
                    self.alpha_concurrency.oracle,  # 并发关系检查器
                    ({'min': 0, 'max': len(simulation_data)},  # 模拟数据范围
                     {'min': 0, 'max': len(log_data)}))  # 真实数据范围
            df_matrix = self._compare_traces(args)  # 直接调用比较函数
        else:
            # 并行处理模式（大数据量）
            cpu_count = multiprocessing.cpu_count()  # 获取CPU核心数
            mx_len = len(log_data)
            ranges = self.define_ranges(mx_len, int(np.ceil(cpu_count / 2)))  # 定义数据分块范围
            ranges = list(itertools.product(*[ranges, ranges]))  # 生成所有分块组合
            reps = len(ranges)  # 总任务数

            pool = Pool(processes=cpu_count)  # 创建进程池
            # 准备每个任务的参数
            args = [(metric, simulation_data[r[0]['min']:r[0]['max']],
                     log_data[r[1]['min']:r[1]['max']],
                     self.alpha_concurrency.oracle,
                     r) for r in ranges]
            p = pool.map_async(self._compare_traces, args)  # 异步提交任务

            if self.verbose:  # 如果启用详细输出
                pbar_async(p, 'evaluating ' + metric + ':')  # 显示进度条

            pool.close()  # 关闭进程池（不再接受新任务）
            # 合并所有子任务的结果
            df_matrix = pd.concat(list(p.get()), axis=0, ignore_index=True)

        # 检查结果有效性
        if df_matrix is None or df_matrix.empty:
            print("Error: df_matrix is None or empty")
            return similarity  # 返回空列表

        # 数据整理：按i,j排序并建立多级索引
        df_matrix.sort_values(by=['i', 'j'], inplace=True)
        df_matrix = df_matrix.reset_index().set_index(['i', 'j'])

        # 根据不同的度量方法处理距离矩阵
        if metric == 'dl_mae':
            # 处理Damerau-Levenshtein + MAE混合度量
            dl_matrix = df_matrix[['dl_distance']].unstack().to_numpy()  # 解堆叠DL距离
            mae_matrix = df_matrix[['mae_distance']].unstack().to_numpy()  # 解堆叠MAE距离

            # MAE归一化处理
            max_mae = mae_matrix.max()
            mae_matrix = np.divide(mae_matrix, max_mae)

            # 加权合并两个距离矩阵（各占50%权重）
            dl_matrix = np.multiply(dl_matrix, 0.5)
            mae_matrix = np.multiply(mae_matrix, 0.5)
            cost_matrix = np.add(dl_matrix, mae_matrix)  # 最终成本矩阵
        else:
            # 单一度量方法直接使用
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()

        # 使用匈牙利算法进行最优匹配
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))

        # 构建返回结果
        for idx, idy in zip(row_ind, col_ind):
            similarity.append(
                dict(
                    caseid=simulation_data[idx]['caseid'],  # 案例ID
                    sim_order=simulation_data[idx]['profile'],  # 模拟数据中的顺序
                    log_order=log_data[idy]['profile'],  # 匹配的真实日志顺序
                    # 计算相似度得分（MAE直接使用，其他度量转换为1-distance）
                    sim_score=(cost_matrix[idx][idy] if metric == 'mae'
                               else (1 - (cost_matrix[idx][idy])))
                )
            )
        return similarity

    @staticmethod
    def _compare_traces(args):
        def ae_distance(et_1, et_2, st_1, st_2):
            cicle_time_s1 = (et_1 - st_1).total_seconds()
            cicle_time_s2 = (et_2 - st_2).total_seconds()
            ae = np.abs(cicle_time_s1 - cicle_time_s2)
            return ae

        def tsd_alpha(s_1, s_2, p_1, p_2, w_1, w_2, alpha_concurrency):
            """
            Compute the Damerau-Levenshtein distance between two given
            strings (s_1 and s_2)
            Parameters
            ----------
            comp_sec : dict
            alpha_concurrency : dict
            Returns
            -------
            Float
            """

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
                        if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                            cost = calculate_cost(i, j - 1)
                        dist[(i, j)] = min(dist[(i, j)], dist[i - 2, j - 2] + cost)  # transposition
            return dist[lenstr1 - 1, lenstr2 - 1]

        def gen(metric, serie1, serie2, oracle, r):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            try:
                df_matrix = list()
                for i, s1_ele in enumerate(serie1):
                    for j, s2_ele in enumerate(serie2):
                        element = {'i': r[0]['min'] + i, 'j': r[1]['min'] + j}
                        if metric in ['tsd', 'dl', 'dl_mae','els']:
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

                if metric == 'tsd':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: tsd_alpha(
                            x.s_1, x.s_2, x.p_1, x.p_2, x.w_1, x.w_2,
                            oracle) / x.length, axis=1)
                elif metric in 'dl':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(
                            ''.join(x.s_1), ''.join(x.s_2)) / x.length, axis=1)
                elif metric == 'mae':
                    df_matrix['distance'] = df_matrix.apply(
                        lambda x: ae_distance(
                            x.et_1, x.et_2, x.st_1, x.st_2), axis=1)
                elif metric == 'dl_mae':
                    df_matrix['dl_distance'] = df_matrix.apply(
                        lambda x: jf.damerau_levenshtein_distance(
                            ''.join(x.s_1), ''.join(x.s_2)) / x.length, axis=1)
                    df_matrix['mae_distance'] = df_matrix.apply(
                        lambda x: ae_distance(
                            x.et_1, x.et_2, x.st_1, x.st_2), axis=1)
                else:
                    raise ValueError(metric)
                return df_matrix
            except Exception:
                traceback.print_exc()
            return None

        return gen(*args)

# =============================================================================
# whole log MAE
# =============================================================================
    def log_mae_metric(self, log_data: list, simulation_data: list, metric) -> list:
        """
        测量两个完整日志之间的MAE距离
        """

        similarity = []

        # 转换为DataFrame（如果需要）
        if isinstance(log_data, list):
            print("[log_mae_metric] 转换log_data为DataFrame")
            log_data = pd.DataFrame(log_data)
        if isinstance(simulation_data, list):
            print("[log_mae_metric] 转换simulation_data为DataFrame")
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
            print("\n[log_mae_metric] 计算日志时间范围...")
            log_start_min = log_data['start_time'].min()
            log_end_max = log_data['end_time'].max()
            log_timelapse = (log_end_max - log_start_min).total_seconds()
        except Exception as e:
            log_timelapse = 0.0

        try:
            sim_start_min = simulation_data['start_time'].min()
            sim_end_max = simulation_data['end_time'].max()
            sim_timelapse = (sim_end_max - sim_start_min).total_seconds()
        except Exception as e:
            sim_timelapse = 0.0

        # 计算相似度分数
        score = abs(sim_timelapse - log_timelapse)

        # 添加结果
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

    def log_emd_metric(self, log_data: list,
                       simulation_data: list, criteria='hour_emd') -> list:
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
            # 确保时间列是datetime类型并统一时区
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
            windows_df = pd.DataFrame.from_dict(daily_windows, orient='index', columns=['window']).rename_axis('hour')
            dataframe = dataframe.merge(windows_df, on='hour', how='left')

            # 选择需要的列并重命名
            result = dataframe[[feature, 'date', 'window']].copy()
            result = result.rename(columns={feature: 'timestamp'})
            result['source'] = source
            return result

        # 使用pd.concat替代append
        data_list = []

        # 处理原始日志的开始时间和结束时间
        data_list.append(split_date_time(log_df.copy(), 'start_time', 'log'))
        data_list.append(split_date_time(log_df.copy(), 'end_time', 'log'))

        # 处理模拟日志的开始时间和结束时间
        data_list.append(split_date_time(sim_df.copy(), 'start_time', 'sim'))
        data_list.append(split_date_time(sim_df.copy(), 'end_time', 'sim'))

        # 合并所有数据
        data = pd.concat(data_list, ignore_index=True)

        # 添加星期几信息
        data['weekday'] = pd.to_datetime(data['date']).dt.dayofweek

        # 定义分组标准
        g_criteria = {'hour_emd': 'window',
                      'day_emd': 'weekday',
                      'day_hour_emd': ['weekday', 'window'],
                      'cal_emd': 'date'}

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
        g_criteria = {'hour_emd': 'window', 'day_emd': 'weekday',
                      'day_hour_emd': ['weekday', 'window'], 'cal_emd': 'date'}
        similarity = list()
        for key, group in data.groupby(g_criteria[criteria]):
            w_df = group.copy()
            w_df = w_df.reset_index()
            basetime = w_df.timestamp.min().floor(freq ='H')
            diftime = lambda x: (x['timestamp'] - basetime).total_seconds()
            w_df['rel_time'] = w_df.apply(diftime, axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                log_hist = np.histogram(w_df[w_df.source=='log'].rel_time,
                                        density=True)
                sim_hist = np.histogram(w_df[w_df.source=='sim'].rel_time,
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

        参数
        ----------
        data : pandas DataFrame
            包含任务信息的数据集
        features : str 或 list
            指定用于创建别名的特征列：
            - 如果是列表：表示需要组合多个特征（如任务+角色）
            - 如果是字符串：表示使用单个特征（如任务名称）

        返回
        -------
        alias : dict
            别名字典，将原始任务或任务-角色组合映射到生成的简写别名
        """

        # 将DataFrame转换为字典记录列表，便于后续处理
        data = data.to_dict('records')

        # 创建一个空集合用于存储唯一的任务/任务-角色组合
        subsec_set = set()

        # 根据features参数类型提取任务/任务-角色组合列表
        if isinstance(features, list):
            # 如果是特征列表：组合多个特征（如任务和角色）
            task_list = [(x[features[0]], x[features[1]]) for x in data]
        else:
            # 如果是单个特征：直接提取特征值
            task_list = [x[features] for x in data]

        # 将提取的任务/任务-角色组合添加到集合中（自动去重）
        [subsec_set.add(x) for x in task_list]

        # 将唯一值集合转换为排序后的列表（确保顺序一致性）
        variables = sorted(list(subsec_set))

        # 准备用于生成别名的字符池（所有字母+数字）
        characters = string.ascii_letters + string.digits

        # 从字符池中随机采样，生成与变量数量相同的唯一别名
        aliases = random.sample(characters, len(variables))

        # 创建别名映射字典
        alias = dict()
        for i, _ in enumerate(variables):
            # 将原始值映射到生成的简写别名
            alias[variables[i]] = aliases[i]

        return alias

    @staticmethod
    def calculate_times(log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
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
                           events[i-1]['end_timestamp']).total_seconds()
                events[i]['waiting_time'] = wit if wit >= 0 else 0
                events[i]['processing_time'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """
        Scales times values activity based

        Parameters
        ----------
        data : dataframe

        Returns
        -------
        data : dataframe with normalized times

        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        if self.one_timestamp:
            summ = data.groupby(['task'])['duration'].max().to_dict()
            dur_act_norm = (lambda x: x['duration']/summ[x['task']]
                            if summ[x['task']] > 0 else 0)
            df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        else:
            summ = data.groupby(['task'])['processing_time'].max().to_dict()
            proc_act_norm = (lambda x: x['processing_time']/summ[x['task']]
                             if summ[x['task']] > 0 else 0)
            df_modif['proc_act_norm'] = df_modif.apply(proc_act_norm, axis=1)
            # ---
            summ = data.groupby(['task'])['waiting_time'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting_time']/summ[x['task']]
                             if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

    def reformat_events(self, data, features):
        """Creates series of activities, roles and relative times per trace.
        parms:
            log_df: dataframe.
            ac_table (dict): index of activities.
            rl_table (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=self.alias[(x[features[0]],
                                             x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=self.alias[x[features]])) for x in data]
        temp_data = list()
        # define ordering keys and columns
        if self.one_timestamp:
            columns = ['alias', 'duration', 'dur_act_norm']
            sort_key = 'end_timestamp'
        else:
            sort_key = 'start_timestamp'
            columns = ['alias', 'processing_time',
                       'proc_act_norm', 'waiting_time', 'wait_act_norm']
        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {**{col: serie}, **temp_dict}
            temp_dict = {**{'caseid': key, 'start_time': trace[0][sort_key],
                            'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))

    def reformat_events1(self, data, features):
        """Creates series of activities, roles and relative times per trace."""
        # Update alias
        if isinstance(features, list):
            [x.update(dict(alias=self.alias[(x[features[0]], x[features[1]])])) for x in data]
        else:
            [x.update(dict(alias=self.alias[x[features]])) for x in data]

        temp_data = list()

        # 修复：使用正确的字段名
        # 旧代码：sort_key = 'start_timestamp'
        # 新代码：使用重命名后的字段名
        if self.one_timestamp:
            sort_key = 'end_time'  # 重命名后的字段
            columns = ['alias', 'duration', 'dur_act_norm']
        else:
            sort_key = 'start_time'  # 重命名后的字段
            columns = ['alias', 'processing_time',
                       'proc_act_norm', 'waiting_time', 'wait_act_norm']

        # 添加调试信息
        print(f"[reformat_events] 使用的排序键: {sort_key}")
        if data:
            print(f"[reformat_events] 第一条记录的键: {list(data[0].keys())}")

        data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))

        for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for col in columns:
                serie = [y[col] for y in trace]
                if col == 'alias':
                    temp_dict = {**{'profile': serie}, **temp_dict}
                else:
                    serie = [y[col] for y in trace]
                temp_dict = {**{col: serie}, **temp_dict}

            # 修复：使用正确的字段名获取时间戳
            temp_dict = {**{'caseid': key,
                            'start_time': trace[0][sort_key],
                            'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)

        return sorted(temp_data, key=itemgetter('start_time'))
    @staticmethod
    def define_ranges(size, num_folds):
        num_events = int(np.round(size/num_folds))
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
