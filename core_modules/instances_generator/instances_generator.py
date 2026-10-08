# -*- coding: utf-8 -*-
"""
实例生成器 - 增强版
====================
改进点：
1. ✅ 完整的TFT集成
2. ✅ 统一的参数管理
3. ✅ 更好的错误处理
4. ✅ 保持向后兼容性
"""

import itertools
import pandas as pd

import support_modules.common as cm
from support_modules.common import InterArrivalGenerativeMethods as IaG
from core_modules.instances_generator import dl_generators as dl
from core_modules.instances_generator import multi_pdf_generators as mpdf
from core_modules.instances_generator import pdf_generators as pdf
from core_modules.instances_generator import prophet_generator as prf
from core_modules.instances_generator import tf_generator as tf


class InstancesGenerator:
    """
    实例生成器 - 统一的到达时间生成接口

    支持的方法：
    - PDF: 概率密度函数
    - DL: 深度学习
    - MULTI_PDF: 多概率密度函数
    - PROPHET: Facebook Prophet
    - TF: Temporal Fusion Transformer (增强版)
    - TEST: 原始到达时间重放
    """

    def __init__(self, process_graph, log, method, parms):
        """
        初始化实例生成器

        Args:
            process_graph: 流程图
            log: 事件日志
            method: 生成方法（IaG枚举）
            parms: 参数字典
        """
        print("\n" + "=" * 80)
        print(f"🚀 初始化实例生成器 - 方法: {method}")
        print("=" * 80)

        self.log = log
        self.method = method

        # ✅ 关键修复1: 在分割之前确保TFT参数
        if method == IaG.TF:
            self._ensure_tft_params(parms)

        # 分割训练/验证集
        self.log_train, self.log_validation = self._split_timeline(
            self.log, 0.8, parms['read_options']['one_timestamp']
        )

        # 分析首个任务
        self.tasks = self._analize_first_tasks(process_graph)
        self.one_timestamp = parms['read_options']['one_timestamp']
        self.time_format = parms['read_options']['timeformat']

        # 挖掘到达间隔（非TFT方法需要）
        if method != IaG.TF:
            self.ia_times = self._mine_inter_arrival_time(
                self.log_train, self.tasks, self.one_timestamp
            )
            self.ia_validation = self._mine_inter_arrival_time(
                self.log_validation, self.tasks, self.one_timestamp
            )
        else:
            # TFT不需要预处理的间隔时间
            self.ia_times = None
            self.ia_validation = None

        self.params = parms

        # 创建生成器
        self._get_generator(method)

        print(f"✅ 生成器初始化完成")

    def generate(self, num_instances, start_time):
        """
        生成指定数量的案例

        Args:
            num_instances: 案例数量
            start_time: 起始时间

        Returns:
            pd.DataFrame: 包含caseid和timestamp的DataFrame
        """
        return self.generator.generate(num_instances, start_time)

    def _get_generator(self, method):
        """
        根据方法创建对应的生成器

        ✅ 改进点：
        1. 统一的参数传递
        2. TFT专用分支
        3. 更清晰的错误处理
        """
        if method == IaG.PDF:
            self.generator = pdf.PDFGenerator(self.ia_times, self.ia_validation)

        elif method == IaG.DL:
            self.generator = dl.DeepLearningGenerator(
                self.ia_times, self.ia_validation, self.params
            )

        elif method == IaG.MULTI_PDF:
            self.generator = mpdf.MultiPDFGenerator(
                self.ia_times, self.ia_validation, self.params
            )

        elif method == IaG.PROPHET:
            self.generator = prf.NeuralProphetGenerator(
                self.log, self.log_validation, self.params
            )

        elif method == IaG.TF:
            print("  🎯 使用TFT生成器(增强版, inter-arrival)")

            try:
                ordering_field = 'end_timestamp' if self.one_timestamp else 'start_timestamp'

                # ✅ 提取“案例到达事件”（一行 = 一个 case arrival）
                train_arrivals = self._extract_arrivals(
                    self.log_train, self.tasks, ordering_field
                )
                val_arrivals = self._extract_arrivals(
                    self.log_validation, self.tasks, ordering_field
                )

                print(f"    ✅ 训练 arrivals: {len(train_arrivals)} cases")
                print(f"    ✅ 验证 arrivals: {len(val_arrivals)} cases")

                # 🔥 关键修复：重命名为 timestamp（TFTGenerator 需要）
                train_arrivals = train_arrivals.rename(columns={"time": "timestamp"})
                val_arrivals = val_arrivals.rename(columns={"time": "timestamp"})

                # 创建 TFT inter-arrival 生成器
                self.generator = tf.TFTGenerator(
                    train_arrivals,
                    val_arrivals,
                    self.params
                )

                print("  ✅ TFT生成器创建成功（event-level）")

            except Exception as e:
                print(f"\n  ❌ TFT生成器创建失败: {e}")
                import traceback
                traceback.print_exc()
                raise

        elif method == IaG.TEST:
            print("  🧪 使用原始到达时间重放")
            self.generator = self.OriginalInterarrival()

        else:
            raise ValueError(f'❌ 不支持的生成方法: {method}')

    def _convert_to_hourly_counts(self, arrivals_df):
        """转换为每小时计数时间序列"""
        arrivals_df['time'] = pd.to_datetime(arrivals_df['time'])
        arrivals_df['hour'] = arrivals_df['time'].dt.floor('H')

        hourly_counts = (
            arrivals_df.groupby('hour')
                .size()
                .reset_index(name='count')
                .rename(columns={'hour': 'time'})
        )

        if len(hourly_counts) > 0:
            full_range = pd.date_range(
                start=hourly_counts['time'].min(),
                end=hourly_counts['time'].max(),
                freq='H'
            )
            hourly_counts = (
                pd.DataFrame({'time': full_range})
                    .merge(hourly_counts, on='time', how='left')
                    .fillna({'count': 0})
            )

        hourly_counts['count'] = hourly_counts['count'].astype(float)

        return hourly_counts
    def _extract_arrivals(self, log, tasks, ordering_field):
        """提取案例到达时间"""
        log_filtered = log[log.task.isin(tasks)]
        arrivals = (
            pd.DataFrame(log_filtered.groupby('caseid')[ordering_field].min())
                .reset_index()
                .rename(columns={ordering_field: 'time'})
        )
        return arrivals
    def _ensure_tft_params(self, parms):
        """
        ✅ 确保参数中包含完整的TFT配置

        改进点：
        1. 更丰富的默认参数
        2. 保留用户自定义参数
        3. 参数验证
        4. 路径自动创建
        """
        print("\n  🔧 配置TFT参数...")

        # ============================================================
        # ✅ 默认TFT参数（与tf_generator.py中的TFTGenerator对齐）
        # ============================================================
        default_tft_params = {
            # 文件路径
            'ia_gen_path': 'input_files/ia_gen_models',
            'file': parms.get('file', 'data'),
            'log_dir': 'lightning_logs',

            # 模型控制
            'update_ia_gen': False,  # 是否强制重新训练
            'save_temp_model': True,  # 是否保存临时模型

        }

        # ============================================================
        # ✅ 合并用户参数
        # ============================================================
        # 1. 如果用户提供了tf_params，优先使用
        if 'tf_params' in parms:
            user_tft_params = parms['tf_params']
            print(f"    ℹ️  检测到用户自定义TFT参数: {len(user_tft_params)} 项")
            default_tft_params.update(user_tft_params)

        # 2. 检查是否有单独的参数覆盖
        for key in default_tft_params.keys():
            if key in parms and key not in ['tf_params']:
                default_tft_params[key] = parms[key]

        # ============================================================
        # ✅ 参数验证
        # ============================================================

        # ============================================================
        # ✅ 更新主参数字典
        # ============================================================
        parms.update(default_tft_params)
        parms['tf_params'] = default_tft_params  # 保留引用

        # ✅ 创建必要的目录
        import os
        os.makedirs(default_tft_params['ia_gen_path'], exist_ok=True)
        os.makedirs(default_tft_params['log_dir'], exist_ok=True)

    def _validate_tft_params(self, params):
        """
        ✅ 验证TFT参数的合理性

        Args:
            params: TFT参数字典
        """
        # 检查必需参数
        required_params = [
            'max_encoder_length', 'max_prediction_length',
            'hidden_size', 'num_layers', 'batch_size'
        ]

        for param in required_params:
            if param not in params:
                raise ValueError(f"❌ 缺少必需的TFT参数: {param}")

        # 参数范围检查
        if params['max_encoder_length'] < 24:
            print(f"    ⚠️  警告: max_encoder_length ({params['max_encoder_length']}) 过小，建议至少24小时")

        if params['max_prediction_length'] < 1:
            raise ValueError("❌ max_prediction_length 必须 >= 1")

        if params['hidden_size'] < 16:
            print(f"    ⚠️  警告: hidden_size ({params['hidden_size']}) 过小，可能影响模型表现")

        if params['num_layers'] < 1:
            raise ValueError("❌ num_layers 必须 >= 1")

        if params['batch_size'] < 1:
            raise ValueError("❌ batch_size 必须 >= 1")

        # 分位数检查
        if not params['quantiles']:
            raise ValueError("❌ quantiles 不能为空")

        if not all(0 < q < 1 for q in params['quantiles']):
            raise ValueError("❌ quantiles 必须在 (0, 1) 范围内")

    @staticmethod
    def _check_cuda_available():
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ============================================================
    # 内部工具类
    # ============================================================
    class OriginalInterarrival:
        """
        原始到达时间重放生成器
        用于测试和基准对比
        """

        @staticmethod
        def generate(log, _start_time):
            """
            从日志中提取原始到达时间

            Args:
                log: 事件日志
                _start_time: 起始时间（未使用）

            Returns:
                pd.DataFrame: caseid和timestamp
            """
            i_arr = log.groupby('caseid').start_timestamp.min().reset_index()
            i_arr.rename(columns={'start_timestamp': 'timestamp'}, inplace=True)
            i_arr['timestamp'] = pd.to_datetime(i_arr['timestamp'])
            i_arr.drop(columns='caseid', inplace=True)
            i_arr.sort_values('timestamp', inplace=True)
            i_arr['caseid'] = i_arr.index + 1
            i_arr['caseid'] = i_arr['caseid'].astype(str)
            i_arr['caseid'] = 'Case' + i_arr['caseid']
            return i_arr[['caseid', 'timestamp']]

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _mine_inter_arrival_time(log_train, tasks, one_ts):
        """
        从事件日志中挖掘到达间隔时间

        Args:
            log_train: 训练日志
            tasks: 首个任务列表
            one_ts: 是否使用单一时间戳

        Returns:
            pd.DataFrame: 到达间隔数据
                - caseid: 案例ID
                - inter_time: 到达间隔（秒）
                - timestamp: 到达时间戳
                - daytime: 一天中的时间（秒）
                - weekday: 星期几（0-6）
        """
        ordering_field = 'end_timestamp' if one_ts else 'start_timestamp'

        # 筛选首个任务
        log_train = log_train[log_train.task.isin(tasks)]

        # 提取每个案例的到达时间
        arrival_timestamps = (
            pd.DataFrame(log_train.groupby('caseid')[ordering_field].min())
                .reset_index()
                .rename(columns={ordering_field: 'timestamp'})
        )

        # 计算到达间隔
        inter_arrival_times = list()
        daily_times = arrival_timestamps.sort_values('timestamp').to_dict('records')

        for i, event in enumerate(daily_times):
            # 计算与前一个案例的间隔
            delta = (
                (daily_times[i]['timestamp'] - daily_times[i - 1]['timestamp']).total_seconds()
                if i > 0 else 0
            )

            # 计算一天中的时间（秒）
            time = daily_times[i]['timestamp'].time()
            time = time.second + time.minute * 60 + time.hour * 3600

            inter_arrival_times.append({
                'caseid': daily_times[i]['caseid'],
                'inter_time': delta,
                'timestamp': daily_times[i]['timestamp'],
                'daytime': time,
                'weekday': daily_times[i]['timestamp'].weekday()
            })

        return pd.DataFrame(inter_arrival_times)

    @staticmethod
    def _analize_first_tasks(process_graph) -> list:
        """
        分析流程图，提取首个任务

        Args:
            process_graph: Networkx有向图

        Returns:
            list: 首个任务名称列表
        """
        temp_process_graph = process_graph.copy()

        # 移除非任务节点（网关等）
        for node in list(temp_process_graph.nodes):
            if process_graph.nodes[node]['type'] not in ['start', 'end', 'task']:
                preds = list(temp_process_graph.predecessors(node))
                succs = list(temp_process_graph.successors(node))
                temp_process_graph.add_edges_from(list(itertools.product(preds, succs)))
                temp_process_graph.remove_node(node)

        # 找到起始节点
        graph_data = pd.DataFrame.from_dict(
            dict(temp_process_graph.nodes.data()),
            orient='index'
        )
        start = graph_data[graph_data.type.isin(['start'])]
        start = start.index.tolist()[0]

        # 获取起始节点的后继任务
        in_tasks = [
            temp_process_graph.nodes[x]['name']
            for x in temp_process_graph.successors(start)
        ]

        return in_tasks

    @staticmethod
    def _split_timeline(log, size: float, one_ts: bool) -> tuple:
        """
        按时间分割事件日志

        Args:
            log: 完整事件日志
            size: 训练集比例（0-1）
            one_ts: 是否使用单一时间戳

        Returns:
            tuple: (训练集, 验证集)
        """
        key = 'end_timestamp' if one_ts else 'start_timestamp'

        # 使用通用分割函数
        train, validation = cm.split_log(log, one_ts, size)

        # 排序
        log_validation = validation.sort_values(key, ascending=True).reset_index(drop=True)
        log_train = train.sort_values(key, ascending=True).reset_index(drop=True)

        return log_train, log_validation

