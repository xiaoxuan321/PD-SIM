# -*- coding: utf-8 -*-

import copy
import os
import shutil
import traceback
import warnings
from operator import itemgetter

import external_tools.SimilarityEvaluator as sim
import pandas as pd
import readers.bpmn_reader as br
import readers.log_reader as lr
import readers.process_structure as gph
import utils.support as sup
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD, DictionaryLearning
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from datetime import timedelta
from utils.support import timeit, safe_exec

import support_modules.common as cm
from core_modules.instances_generator import instances_generator as gen
from core_modules.sequences_generator import seq_generator as sg
from core_modules.sequences_generator.seq_generator import StochasticProcessModelGenerator
from core_modules.times_allocator import times_generator as ta

warnings.filterwarnings("ignore")


# 数据加载策略：优先读取 exp_data 预分割文件（log_train.xes / log_test.xes），
# 无需时区转换和分割；若不存在则回退到读取完整 XES + 时区转换 + 分割。


class DeepSimulator:
    """
    Main class of the Simulator
    """

    def __init__(self, parms):
        """constructor"""
        self.parms = parms
        self.is_safe = True
        self.sim_values = list()

    def execute_pipeline(self) -> None:
        # -------------------- 1. 初始化 --------------------
        # 创建两个空 DataFrame，分别用来存放训练集日志和测试集日志
        self.log_test = pd.DataFrame()
        self.log_train = pd.DataFrame()

        exec_times = dict()  # 记录各阶段耗时
        self.is_safe = self._read_inputs(  # 读取输入文件（日志、参数）划分
            log_time=exec_times,
            is_safe=self.is_safe
        )

        # 如果日志读取失败或测试日志为空，直接退出
        if not self.is_safe or self.log_test.empty:
            print("[严重错误] 日志读取失败或测试日志为空，无法继续执行")
            return

        # -------------------- 2. 计算仿真起始时间 --------------------
        # 取测试日志中所有事件的最小时间作为仿真起点
        start_time = (
            self.log_test.start_timestamp.min()
                .strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        )

        print('############ 结构优化 ############')
        # 根据参数选择序列生成器（stochastic_process_model）
        seq_gen = StochasticProcessModelGenerator(
            {**self.parms['gl'], **self.parms['s_gen']},  # 全局参数 + 序列生成参数
            self.log_train  # 训练日志
        )
        self.log_train_copy = copy.deepcopy(self.log_train)
        # 5.1 生成序列
        seq_gen.discovery_model()
        self.log_train = copy.deepcopy(self.log_train_copy)
        # -------------------- 3. 生成到达间隔时间 --------------------
        print('############ 生成到达间隔时间 ############')
        self.is_safe = self._read_bpmn(log_time=exec_times, is_safe=self.is_safe)
        generator = gen.InstancesGenerator(
            self.process_graph,
            self.log_train,
            self.parms['i_gen']['gen_method'],  # 到达间隔时间生成方法
            {**self.parms['gl'], **self.parms['i_gen']}  # 全局参数 + 间隔生成参数
        )

        # -------------------- 4. 生成实例时间 --------------------
        print('########### 生成实例时间 ###########')
        times_allocator = ta.TimesGenerator(
            self.process_graph,
            self.log_train,
            {**self.parms['gl'], **self.parms['t_gen']}  # 全局参数 + 时间分配参数
        )

        # 创建输出目录
        output_path = os.path.join('output_files', sup.folder_id())
        print(f"创建输出目录: {output_path}")

        # 再次检查测试日志是否为空
        if self.log_test.empty:
            print("[严重错误] 测试日志数据为空！无法进行仿真")
            return

        # -------------------- 5. 多次重复仿真 --------------------
        for rep_num in range(0, self.parms['gl']['exp_reps']):
            print(f"===== 开始第 {rep_num + 1}/{self.parms['gl']['exp_reps']} 次重复仿真 =====")

            # 5.1 生成序列
            seq_gen.generate(self.log_test, start_time)

            # 如果生成的序列为空，跳过本次重复
            if seq_gen.gen_seqs is None or seq_gen.gen_seqs.empty:
                print(f"[错误] 第{rep_num + 1}次重复生成的序列为空!")
                continue

            print(f"生成的序列记录数: {len(seq_gen.gen_seqs)}")
            print(f"生成的序列案例数: {seq_gen.gen_seqs['caseid'].nunique()}")


            # 5.2 生成到达间隔时间（案例之间的间隔）
            # 统计测试日志的案例数，然后生成相应数量的间隔
            case_count = len(self.log_test.caseid.unique())
            print(f"生成到达间隔时间，案例数量: {case_count}")
            if case_count == 0:
                print("[错误] 测试日志中的案例数量为0!")
                continue
            inter_arrival = generator.generate(case_count, start_time)

            if inter_arrival is None or len(inter_arrival) == 0:
                print(f"[错误] 第{rep_num}次重复生成的到达间隔时间为空!")
                continue

            # 5.3 清理时间戳
            seq_gen.clean_time_stamps()

            # 5.4 将“序列 + 到达间隔时间”映射为完整事件日志
            event_log = times_allocator.generate(seq_gen.gen_seqs, inter_arrival)

            if event_log is None or len(event_log) == 0:
                print(f"[错误] 第{rep_num}次重复生成的事件日志为空!")
                continue

            event_log = pd.DataFrame(event_log)

            # 5.5 导出 & 评估
            print(f"第{rep_num}次重复生成的事件日志记录数: {len(event_log)}")
            print(f"第{rep_num}次重复生成的事件日志案例数: {event_log['caseid'].nunique()}")

            self._export_log(event_log, output_path, rep_num)  # 保存日志到磁盘
            if self.parms['gl']['evaluate']:
                self.sim_values.extend(
                    self._evaluate_logs(self.parms, self.log_test, event_log, rep_num)
                )

        # -------------------- 6. 结束 --------------------
        self._export_results(output_path)
        print("-- 试验结束 --")

    @timeit
    @safe_exec
    def _read_inputs(self, **_kwargs) -> None:
        try:
            one_ts = self.parms['gl']['read_options']['one_timestamp']
            key = 'end_timestamp' if one_ts else 'start_timestamp'
            ts_cols = ['end_timestamp'] if one_ts else ['start_timestamp', 'end_timestamp']

            # 优先使用 exp_data 预分割文件（无需时区转换和分割）
            log_name = self.parms['gl']['file'].split('.')[0]
            exp_data_dir = os.path.join(self.parms['gl']['event_logs_path'], 'exp_data', log_name)
            train_path = os.path.join(exp_data_dir, 'log_train.xes')
            test_path = os.path.join(exp_data_dir, 'log_test.xes')

            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"使用预分割数据: {exp_data_dir}")

                # exp_data XES 使用 start:timestamp + time:timestamp 双时间戳格式，
                # 没有 lifecycle:transition 属性，不能用标准 LogReader 读取。
                # 直接用 pm4py 读取并手动构造 LogReader 兼容对象。
                print(f"读取训练日志: {train_path}")
                self.log_train, train_df = self._read_exp_xes(train_path, one_ts)
                train_sorted = pd.DataFrame(self.log_train.data).sort_values(key, ascending=True).reset_index(drop=True)
                self.log_train.set_data(train_sorted.to_dict('records'))
                print(f"训练集: {train_df['caseid'].nunique()} cases, {len(train_df)} records")

                print(f"读取测试日志: {test_path}")
                test_log_reader, test_df = self._read_exp_xes(test_path, one_ts)
                test_sorted = pd.DataFrame(test_log_reader.data).sort_values(key, ascending=True).reset_index(drop=True)
                self.log_test = test_sorted
                print(f"测试集: {test_df['caseid'].nunique()} cases, {len(test_df)} records")

                return True

            print(f"[错误] 未找到 exp_data 预分割文件: {train_path} / {test_path}")
            print(f"请确保 exp_data/{log_name}/ 目录下存在 log_train.xes 和 log_test.xes")
            return False
        except Exception as e:
            print(f"读取输入数据时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    def _read_exp_xes(self, xes_path, one_ts):
        """读取 exp_data XES 文件并构造 LogReader 兼容对象。

        exp_data XES 使用 start:timestamp + time:timestamp 双时间戳格式，
        没有 lifecycle:transition 属性，因此不能用标准 LogReader 读取。
        """
        from pm4py import read_xes as pm4py_read_xes
        import itertools as it
        from datetime import timedelta

        df = pm4py_read_xes(xes_path)

        # 重命名列
        rename_map = {
            'case:concept:name': 'caseid',
            'concept:name': 'task',
            'org:resource': 'user',
        }
        if 'start:timestamp' in df.columns:
            rename_map['start:timestamp'] = 'start_timestamp'
        if 'time:timestamp' in df.columns:
            rename_map['time:timestamp'] = 'end_timestamp'
        df = df.rename(columns=rename_map)

        # 确保 user 列存在
        if 'user' not in df.columns:
            df['user'] = ''

        # 过滤 Start/End 任务
        df = df[~df.task.isin(['Start', 'End', 'start', 'end'])].reset_index(drop=True)

        # 去除时区信息（exp_data XES 时间戳已为本地时间，以 +00:00 偏移存储）
        ts_cols = ['end_timestamp'] if one_ts else ['start_timestamp', 'end_timestamp']
        for col in ts_cols:
            if col in df.columns and hasattr(df[col].dtype, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

        # 只保留必要列
        if self.parms['gl']['read_options'].get('filter_d_attrib', True):
            keep_cols = ['caseid', 'task', 'user']
            if one_ts:
                keep_cols.append('end_timestamp')
            else:
                keep_cols.extend(['start_timestamp', 'end_timestamp'])
            df = df[[c for c in keep_cols if c in df.columns]]

        # 添加 Start/End 虚拟事件（模拟 LogReader.append_csv_start_end）
        records = df.to_dict('records')
        end_start_times = dict()
        df_temp = pd.DataFrame(records)
        for case, group in df_temp.groupby('caseid'):
            if one_ts:
                end_start_times[(case, 'Start')] = group.end_timestamp.min() - timedelta(microseconds=1)
            else:
                end_start_times[(case, 'Start')] = group.start_timestamp.min() - timedelta(microseconds=1)
            end_start_times[(case, 'End')] = group.end_timestamp.max() + timedelta(microseconds=1)

        new_data = list()
        records_sorted = sorted(records, key=lambda x: x['caseid'])
        for ckey, group in it.groupby(records_sorted, key=lambda x: x['caseid']):
            trace = list(group)
            # Start 虚拟事件
            start_event = {
                'caseid': trace[0]['caseid'],
                'task': 'Start',
                'user': 'Start',
                'end_timestamp': end_start_times[(ckey, 'Start')],
            }
            if not one_ts:
                start_event['start_timestamp'] = end_start_times[(ckey, 'Start')]
            trace.insert(0, start_event)
            # End 虚拟事件
            end_event = {
                'caseid': trace[-1]['caseid'],
                'task': 'End',
                'user': 'End',
                'end_timestamp': end_start_times[(ckey, 'End')],
            }
            if not one_ts:
                end_event['start_timestamp'] = end_start_times[(ckey, 'End')]
            trace.append(end_event)
            new_data.extend(trace)

        # 创建 raw_data（模拟 LogReader.split_event_transitions）
        raw_data = list()
        for event in df.to_dict('records'):
            if one_ts:
                e = event.copy()
                e['timestamp'] = e.pop('end_timestamp')
                e['event_type'] = 'complete'
                raw_data.append(e)
            else:
                e_start = {k: v for k, v in event.items() if k != 'end_timestamp'}
                e_start['timestamp'] = e_start.pop('start_timestamp')
                e_start['event_type'] = 'start'
                raw_data.append(e_start)

                e_complete = {k: v for k, v in event.items() if k != 'start_timestamp'}
                e_complete['timestamp'] = e_complete.pop('end_timestamp')
                e_complete['event_type'] = 'complete'
                raw_data.append(e_complete)

        # 创建 LogReader 对象（不调用 __init__，避免触发 XES 解析）
        log_reader = object.__new__(lr.LogReader)
        log_reader.data = new_data
        log_reader.raw_data = raw_data
        log_reader.one_timestamp = one_ts
        log_reader.column_names = self.parms['gl']['read_options']['column_names'].copy()
        log_reader.filter_d_attrib = self.parms['gl']['read_options'].get('filter_d_attrib', True)
        log_reader.timeformat = self.parms['gl']['read_options']['timeformat']
        log_reader.verbose = True
        log_reader.input = xes_path
        log_reader.file_name = xes_path
        log_reader.file_extension = '.xes'

        return log_reader, df

    @timeit
    @safe_exec
    def _read_bpmn(self, **_kwargs) -> None:
        try:
            bpmn_path = os.path.join(self.parms['gl']['bpmn_models'], self.parms['gl']['file'].split('.')[0] + '.bpmn')
            print(f"读取BPMN模型: {bpmn_path}")

            self.bpmn = br.BpmnReader(bpmn_path)
            self.process_graph = gph.create_process_structure(self.bpmn)

            # 检查BPMN模型是否有效
            if not self.process_graph.nodes:
                print("[警告] BPMN模型未包含任何节点！")

            print(f"BPMN节点数量: {len(self.process_graph.nodes)}")
            return True
        except Exception as e:
            print(f"读取BPMN时发生错误: {str(e)}")
            traceback.print_exc()
            return False

    @staticmethod
    def _evaluate_logs(parms, log, sim_log, rep_num):
        """评估日志相似度"""
        print(f"\n===== 开始评估第{rep_num}次重复的日志相似度 =====")

        # 检查日志是否为空
        if log.empty:
            print(f"[错误] 第{rep_num}次重复评估中，真实日志为空！")
            return []

        if sim_log.empty:
            print(f"[错误] 第{rep_num}次重复评估中，模拟日志为空！")
            return []

        print(f"真实日志记录数: {len(log)}")
        print(f"模拟日志记录数: {len(sim_log)}")

        sim_values = list()
        log = copy.deepcopy(log)

        # 过滤掉开始/结束任务
        original_size = len(log)
        log = log[~log.task.isin(['Start', 'End'])]
        filtered_size = len(log)
        print(f"过滤后真实日志记录数: {filtered_size} (移除了 {original_size - filtered_size} 条Start/End记录)")

        log['source'] = 'log'
        log.rename(columns={'user': 'resource'}, inplace=True)
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']

        # 检查日志中是否包含必要列
        required_columns = ['caseid', 'task', 'start_timestamp', 'end_timestamp']
        for col in required_columns:
            if col not in log.columns:
                print(f"[错误] 真实日志缺少必要列: {col}")
                return []
            if col not in sim_log.columns:
                print(f"[错误] 模拟日志缺少必要列: {col}")
                return []

        evaluator = sim.SimilarityEvaluator(log, sim_log, parms['gl'], max_cases=1000)
        metrics = [parms['gl']['sim_metric']]
        if 'add_metrics' in parms['gl'].keys():
            metrics = list(set(list(parms['gl']['add_metrics']) + metrics))

        for metric in metrics:
            print(f"\n评估指标: {metric}")
            try:
                evaluator.measure_distance(metric)
                sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
                print(f"{metric} 评估完成，相似度值: {evaluator.similarity['sim_val']}")
            except Exception as e:
                print(f"评估 {metric} 时发生错误: {str(e)}")
                traceback.print_exc()

        return sim_values

    def _export_log(self, event_log, output_path, r_num) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        file_name = os.path.join(output_path,
                                 'gen_' + self.parms['gl']['file'].split('.')[0] + '_' + str(r_num + 1) + '.csv')
        event_log.to_csv(file_name, index=False)
        print(f"已导出仿真日志到: {file_name}")

    @staticmethod
    def clustering_method(dataframe, method, k=3):

        cols = [x for x in dataframe.columns if 'id_' in x]
        x = dataframe[cols]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=30).fit(x)
            dataframe['cluster'] = kmeans.labels_
        elif method == 'mean_shift':
            ms = MeanShift(bandwidth=k, bin_seeding=True).fit(x)
            dataframe['cluster'] = ms.labels_
        elif method == 'gaussian_mixture':
            dataframe['cluster'] = GaussianMixture(
                n_components=k, covariance_type='spherical', random_state=30).fit_predict(x)

        return dataframe

    @staticmethod
    def decomposition_method(dataframe, method):

        cols = [x for x in dataframe.columns if 'id_' in x]
        X = dataframe[cols]

        if method == 'pca':
            dataframe[['x', 'y', 'z']] = PCA(n_components=3).fit_transform(X)
        elif method == 'truncated_svd':
            dataframe[['x', 'y', 'z']] = TruncatedSVD(n_components=3).fit_transform(X)
        elif method == 'dictionary_learning':
            dataframe[['x', 'y', 'z']] = DictionaryLearning(n_components=3,
                                                            transform_algorithm='lasso_lars').fit_transform(X)

        return dataframe

    def _clustering_metrics(self, params):
        """
        计算聚类评估指标，用于评估嵌入向量的聚类效果

        参数:
            params: 参数字典，包含配置信息

        返回:
            包含最佳聚类结果的DataFrame（转置后）
        """

        # 1. 从参数中获取必要信息
        file_name = params['gl']['file']  # 文件名
        embedded_path = params['gl']['embedded_path']  # 嵌入矩阵路径
        concat_method = params['t_gen']['concat_method']  # 连接方法
        include_times = params['t_gen']['include_times']  # 是否包含时间信息
        emb_method = params['t_gen']['emb_method']  # 嵌入方法

        # 2. 构建嵌入矩阵文件路径
        emb_path = os.path.join(
            embedded_path,
            # 使用EmbeddingMethods类方法生成标准化的嵌入矩阵文件名
            cm.EmbeddingMethods.get_matrix_file_name(
                emb_method,
                include_times,
                concat_method,
                file_name
            )
        )

        # 3. 加载嵌入矩阵数据
        df_embeddings = pd.read_csv(emb_path, header=None)  # 无表头读取CSV
        n_cols = len(df_embeddings.columns)  # 获取列数

        # 4. 设置列名:
        # 第一列是id, 第二列是任务名, 其余列是嵌入向量维度(id_1, id_2...)
        df_embeddings.columns = ['id', 'task_name'] + ['id_{}'.format(idx) for idx in range(1, n_cols - 1)]
        df_embeddings['task_name'] = df_embeddings['task_name'].str.lstrip()  # 去除任务名左侧空格

        # 5. 定义聚类评估配置
        # (原代码中有更多选项，但被注释掉了，实际只用kmeans和pca)
        clustering_ms = ['kmeans']  # 使用的聚类方法
        decomposition_ms = ['pca']  # 使用的降维方法
        KS = [3]  # 要测试的聚类数量

        # 6. 评估不同聚类组合的效果
        metrics = []  # 存储评估结果
        for clustering_m in clustering_ms:
            for decomposition_m in decomposition_ms:
                for K in KS:
                    # 6.1 应用聚类方法
                    df_embeddings_tmp = self.clustering_method(df_embeddings, clustering_m, K)

                    # 6.2 应用降维方法
                    df_embeddings_tmp = self.decomposition_method(df_embeddings_tmp, decomposition_m)

                    # 6.3 计算聚类评估指标
                    # 轮廓系数(值越大越好，范围[-1,1])
                    s_score = silhouette_score(
                        df_embeddings_tmp[['x', 'y', 'z']],  # 使用降维后的三维特征
                        df_embeddings_tmp['cluster'],  # 聚类标签
                        metric='euclidean'  # 使用欧式距离
                    )

                    # Calinski-Harabasz指数(值越大越好)
                    ch_score = calinski_harabasz_score(
                        df_embeddings_tmp[['x', 'y', 'z']],
                        df_embeddings_tmp['cluster']
                    )

                    # 6.4 记录结果
                    metrics.append([
                        clustering_m,  # 聚类方法
                        decomposition_m,  # 降维方法
                        K,  # 聚类数量
                        s_score,  # 轮廓系数
                        ch_score  # Calinski-Harabasz指数
                    ])

        # 7. 整理评估结果
        metrics_df = pd.DataFrame(
            data=metrics,
            columns=[
                'clustering_method',  # 聚类方法
                'decomposition_method',  # 降维方法
                'number_clusters',  # 聚类数量
                'silhouette_score',  # 轮廓系数
                'calinski_harabasz_score'  # CH指数
            ]
        )

        # 8. 选择最佳结果(按轮廓系数和CH指数排序)
        best = metrics_df.sort_values(
            by=['silhouette_score', 'calinski_harabasz_score'],
            ascending=False  # 升序排序(实际上应该False获取最大值?)
        ).head(1)  # 取第一名

        # 9. 返回转置后的结果(方便后续处理)
        return best.T.reset_index()

    def _export_results(self, output_path) -> None:
        """
        将仿真结果、测试日志以及训练好的各类模型统一导出到 output_path 目录下。
        """

        # 1) 计算聚类/结构相似度指标（如活动分布、轨迹距离等）
        clust_mets = self._clustering_metrics(self.parms)  # 返回 DataFrame
        clust_mets.columns = ['metric', 'sim_val']  # 统一列名
        clust_mets['run_num'] = 0.0  # 加一列用于标记重复编号（这里固定 0）

        # 2) 把多次重复仿真得到的各类评价指标合并
        sim_values_df = pd.DataFrame(self.sim_values).sort_values(by='metric')  # 已存于 self.sim_values
        results_df = pd.concat([sim_values_df, clust_mets])  # 行拼接
        # ===== 在这里打印 =====
        print('[clust_mets]')
        print(clust_mets[['metric', 'sim_val']])
        print('[results_df]')
        print(results_df[['metric', 'sim_val', 'run_num']])

        # 计算每个指标多次运行的均值
        mean_rows = []
        if not sim_values_df.empty:
            sim_values_df['_sim_val_num'] = pd.to_numeric(sim_values_df['sim_val'], errors='coerce')
            means = sim_values_df.groupby('metric')['_sim_val_num'].mean()
            for metric_name, mean_val in sorted(means.items()):
                # 格式化为普通小数，避免科学记数法
                formatted_val = f'{mean_val:.6f}'
                mean_rows.append({'metric': metric_name, 'sim_val': formatted_val, 'run_num': 'mean'})

            # 打印均值
            mean_summary = pd.DataFrame([{'metric': k, 'mean_sim_val': f'{v:.6f}'}
                                         for k, v in sorted(means.items())])
            print('\n[各指标多次运行均值]')
            print(mean_summary.to_string(index=False))

        # 将均值行追加到 results_df，一并保存
        if mean_rows:
            results_df = pd.concat([results_df, pd.DataFrame(mean_rows)], ignore_index=True)
            results_df['run_num'] = results_df['run_num'].fillna(0.0)

        # 3) 将指标结果保存为 csv
        self._save_embedding_metrics_results(output_path, results_df)

        # 4) 保存“原始测试日志”（去掉 Start / End 两类人工事件）
        log_test = self.log_test[~self.log_test.task.isin(['Start', 'End'])]
        log_test.to_csv(
            os.path.join(output_path,
                         'tst_' + self.parms['gl']['file'].split('.')[0] + '.csv'),
            index=False
        )

        # 5) 若配置要求保存模型，则复制相关文件
        if self.parms['gl']['save_models']:
            # 5-a) 需要检索的文件夹列表
            paths = ['bpmn_models', 'embedded_path', 'ia_gen_path',
                     'seq_flow_gen_path', 'times_gen_path']
            sources = []  # 收集所有待复制的文件完整路径

            # 遍历上述文件夹，找到与当前日志同名的文件
            for path in paths:
                for root, dirs, files in os.walk(self.parms['gl'][path]):
                    for file in files:
                        if self.parms['gl']['file'].split('.')[0] in file:
                            sources.append(os.path.join(root, file))

            # 5-b) 逐个文件复制
            for source in sources:
                # 目标文件夹：在 output_path 下再建同名子文件夹
                base_folder = os.path.join(output_path,
                                           os.path.basename(os.path.dirname(source)))
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)
                destination = os.path.join(base_folder, os.path.basename(source))

                # --- 复制深度学习时间模型 ---
                allowed_ext = self._define_model_path(
                    {**self.parms['gl'], **self.parms['t_gen']})
                is_dual = self.parms['t_gen']['model_type'] == 'dual_inter'

                if is_dual and ('times_gen_models' in source) and \
                        any([x in source for x in allowed_ext]):
                    shutil.copyfile(source, destination)
                elif not is_dual and ('times_gen_models' in source) and \
                        any([self.parms['gl']['file'].split('.')[0] + x in source
                             for x in allowed_ext]):
                    shutil.copyfile(source, destination)

                # --- 复制嵌入矩阵或嵌入模型 ---
                if 'embedded_matrix' in source:
                    self._copy_embeddings(
                        destination, source,
                        self.parms['t_gen']['emb_method'],
                        self.parms['t_gen']['include_times'],
                        self.parms['t_gen']['concat_method'],
                        self.parms['gl']['file'])

                # --- 复制其他 BPMN / 概率模型 ---
                folders = ['bpmn_models', 'ia_gen_models']
                allowed_ext = ['.bpmn', '_mpdf.json', '_prf.json', '_prf_meta.json',
                               '_mpdf_meta.json', '_meta.json']
                if any([x in source for x in folders]) and \
                        any([self.parms['gl']['file'].split('.')[0] + x in source
                             for x in allowed_ext]):
                    shutil.copyfile(source, destination)

    @staticmethod
    def _copy_embeddings(destination, source, emb_method, include_times, concat_method, file_name):
        emb_file_name = cm.EmbeddingMethods.get_matrix_file_name(emb_method, include_times, concat_method, file_name)
        if emb_file_name in source:
            shutil.copyfile(source, destination)
        if emb_method not in cm.EmbeddingMethods.W2VEC:
            model_file_name = cm.EmbeddingMethods.get_model_file_name(emb_method, include_times, file_name)
            if model_file_name in source:
                shutil.copyfile(source, destination)

    def _save_embedding_metrics_results(self, output_path, results_df):
        """
        保存嵌入模型的评估指标结果到CSV文件

        参数:
            output_path: 输出目录路径
            results_df: 包含评估指标的DataFrame
        """

        # 1. 保存原始结果数据（纵向格式）
        # 使用sup.file_id()生成带'SE_'前缀的文件名，并保存到output_path
        results_df.to_csv(
            os.path.join(output_path, sup.file_id(prefix='SE_')),
            index=False  # 不保存行索引
        )

        # 2. 转换数据为横向格式（每个指标作为一列）
        # - 先将'metric'列设为索引
        # - 转置DataFrame（T）
        # - 重置索引并丢弃原索引（metric名称现在变为列名）
        results_df_t = results_df.set_index('metric').T.reset_index(drop=True)

        # 3. 添加元数据信息到横向格式的DataFrame
        # 获取嵌入方法的输入类型和时间包含方式
        input_method, include_times = cm.EmbeddingMethods.get_input_and_times_method(
            self.parms['t_gen']['emb_method'],  # 嵌入方法
            self.parms['t_gen']['include_times'],  # 是否包含时间信息
            self.parms['t_gen']['concat_method']  # 连接方法
        )

        # 添加各种元数据列
        results_df_t['input_method'] = input_method  # 输入数据方法
        results_df_t['embedding_method'] = cm.EmbeddingMethods.get_base_model(
            self.parms['t_gen']['emb_method']  # 基础嵌入模型类型
        )
        results_df_t['log_name'] = self.parms['gl']['file']  # 日志文件名
        results_df_t['times_included'] = include_times  # 是否包含时间信息

        # 4. 保存横向格式的结果数据到output_files目录
        results_df_t.to_csv(
            os.path.join(
                'output_files',
                # 使用EmbeddingMethods类方法生成特定格式的文件路径
                cm.EmbeddingMethods.get_metrics_file_path(
                    self.parms['t_gen']['emb_method'],  # 嵌入方法
                    self.parms['t_gen']['emb_method'],  # 再次传入嵌入方法（可能是为了统一接口）
                    self.parms['t_gen']['include_times'],  # 是否包含时间
                    self.parms['t_gen']['concat_method']  # 连接方法
                )
            ),
            index=False  # 不保存行索引
        )

    @staticmethod
    def _define_model_path(parms):
        inter = parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']
        is_dual = parms['model_type'] == 'dual_inter'
        next_ac = parms['model_type'] == 'inter_nt'
        arpool = parms['all_r_pool']
        if inter:
            if is_dual:
                if arpool:
                    return ['_dpiapr', '_dwiapr', '_diapr']
                else:
                    return ['_dpispr', '_dwispr', '_dispr']
            else:
                if next_ac:
                    if arpool:
                        return ['_inapr']
                    else:
                        return ['_inspr']
                else:
                    if arpool:
                        return ['_iapr']
                    else:
                        return ['_ispr']
        else:
            return ['.h5', '_scaler.pkl', '_meta.json']

    @staticmethod
    def _save_times(times, parms):
        times = [{**{'output': parms['output']}, **times}]
        log_file = os.path.join('output_files', 'execution_times.csv')
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    # =============================================================================
    # Support methods
    # =============================================================================
    @staticmethod
    def _get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), data)), key=itemgetter(order_key))
            traces.append(trace)
        return traces
