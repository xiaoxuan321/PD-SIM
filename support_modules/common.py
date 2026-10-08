# -*- coding: utf-8 -*-
# 引入 dataclass 以简化数据类的声明
from dataclasses import dataclass
# 引入 List 类型注解
from typing import List

import pandas as pd
# 日志切分器，用于将日志划分为训练集和验证集
from readers.log_splitter import LogSplitter


# 文件扩展名常量
@dataclass
class FileExtensions:
    BPMN: str = '.bpmn'   # BPMN 模型文件
    H5: str = '.h5'       # HDF5 权重/模型文件
    XES: str = '.xes'     # XES 事件日志文件
    CSV: str = '.csv'     # CSV 事件日志文件
    JSON: str = '.json'   # JSON 配置文件
    EMB: str = '.emb'     # 嵌入向量文件


# 日志中常用字段的统一命名
@dataclass
class LogAttributes:
    CASE_ID: str = 'caseid'        # 案例 ID
    ACTIVITY: str = 'task'         # 活动名称
    START_TIME: str = 'start_timestamp'  # 活动开始时间
    END_TIME: str = 'end_timestamp'      # 活动完成时间
    RESOURCE: str = 'user'         # 资源（执行者）
    ROLE: str = 'role'             # 角色
    TIMESTAMP: str = 'timestamp'   # 通用时间戳字段


# 序列生成方法枚举
@dataclass
class SequencesGenerativeMethods:
    PROCESS_MODEL: str = 'stochastic_process_model'  # 基于随机过程模型
    TEST: str = 'test'                               # 测试方法

    def get_methods(self) -> List[str]:
        """返回所有可用的序列生成方法"""
        return list(self.__dict__.values())


# 到达间隔时间（Inter-Arrival）生成方法枚举
@dataclass
class InterArrivalGenerativeMethods:
    PDF: str = 'pdf'            # 概率密度函数
    DL: str = 'dl'              # 深度学习方法
    MULTI_PDF: str = 'mul_pdf'  # 多元 PDF
    TEST: str = 'test'          # 测试方法
    TF: str = 'tf'              # TensorFlow 方法
    PROPHET: str = 'prophet'    # Facebook Prophet 方法

    def get_methods(self) -> List[str]:
        """返回所有可用的到达间隔生成方法"""
        return list(self.__dict__.values())


# Word2Vec 拼接方式枚举
@dataclass
class W2VecConcatMethod:
    SINGLE_SENTENCE: str = 'single_sentence'  # 单句拼接
    FULL_SENTENCE: str = 'full_sentence'      # 全句拼接
    WEIGHTING: str = 'weighting'              # 加权拼接

    def get_methods(self) -> List[str]:
        """返回所有可用的拼接方法"""
        return list(self.__dict__.values())


# SplitMiner 算法版本枚举
@dataclass
class SplitMinerVersion:
    SM_V1: str = 'sm1'  # SplitMiner 第一版
    SM_V2: str = 'sm2'  # SplitMiner 第二版
    SM_V3: str = 'sm3'  # SplitMiner 第三版

    def get_methods(self) -> List[str]:
        """返回所有可用的算法版本"""
        return list(self.__dict__.values())


# 嵌入方法枚举及相关工具函数
@dataclass
class EmbeddingMethods:
    DOT_PROD: str = 'emb_dot_product'                 # 点积嵌入
    DOT_PROD_TIMES: str = 'emb_dot_product_times'     # 带时间的点积嵌入
    W2VEC: str = 'emb_w2vec'                          # Word2Vec 嵌入
    DOT_PROD_ACT_WEIGHT: str = 'emb_dot_product_act_weighting'  # 活动加权的点积嵌入

    # ------------------------------------------------------------------
    # 根据方法返回基础模型名称
    # ------------------------------------------------------------------
    @classmethod
    def get_base_model(cls, method):
        if method in [cls.DOT_PROD, cls.DOT_PROD_TIMES, cls.DOT_PROD_ACT_WEIGHT]:
            return 'Dot product'
        elif method == cls.W2VEC:
            return 'Word2vec'

    # ------------------------------------------------------------------
    # 获取输入类型及是否包含时间特征
    # ------------------------------------------------------------------
    @classmethod
    def get_input_and_times_method(cls, method, include_times, concat_method):
        if method == cls.DOT_PROD:
            return 'N/A', False
        elif method == cls.DOT_PROD_TIMES:
            return 'Times', True
        elif method == cls.W2VEC:
            return concat_method, include_times
        elif method == cls.DOT_PROD_ACT_WEIGHT:
            return 'Activity weighting', include_times

    # ------------------------------------------------------------------
    # 根据方法、是否包含时间、拼接方式等生成评估指标文件名
    # ------------------------------------------------------------------
    @classmethod
    def get_metrics_file_path(cls, method, include_times, concat_method, file_name):
        name = file_name.split('.')[0]
        _, inc_times = cls.get_input_and_times_method(method, include_times, concat_method)
        times = 'times' if inc_times else 'no_times'
        if method in [cls.DOT_PROD, cls.DOT_PROD_TIMES]:
            return f"ac_DP_{times}_{name}.csv"
        elif method == cls.DOT_PROD_ACT_WEIGHT:
            return f"ac_DP_act_weighting_{times}_{name}.csv"
        elif method == cls.W2VEC:
            return f"ac_W2V_{concat_method}_{times}_{name}.csv"

    # ------------------------------------------------------------------
    # 生成嵌入矩阵文件名
    # ------------------------------------------------------------------
    @classmethod
    def get_matrix_file_name(cls, method, include_times, concat_method, file_name):
        name = file_name.split('.')[0]
        if method == cls.DOT_PROD:
            return f"ac_DP_{name}{FileExtensions.EMB}"
        elif method == cls.W2VEC:
            return f"ac_W2V_{concat_method}_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_TIMES:
            return f"ac_DP_times_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and include_times:
            return f"ac_DP_act_weighting_times_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and not include_times:
            return f"ac_DP_act_weighting_no_times_{name}{FileExtensions.EMB}"

    # ------------------------------------------------------------------
    # 生成模型文件名
    # ------------------------------------------------------------------
    @classmethod
    def get_model_file_name(cls, method, include_times, file_name):
        name = file_name.split('.')[0]
        if method == cls.DOT_PROD:
            return f"ac_DP_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_TIMES:
            return f"ac_DP_times_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and include_times:
            return f"ac_DP_act_weighting_times_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and not include_times:
            return f"ac_DP_act_weighting_no_times_{name}_emb{FileExtensions.H5}"
        else:
            return None

    # 返回所有可用嵌入类型
    def get_types(self) -> List[str]:
        return list(self.__dict__.values())


# ------------------------------------------------------------------
# 日志切分工具函数：将日志按时间线切分为训练集和验证集
# ------------------------------------------------------------------
def split_log(log, one_ts, size):
    """
    log   : LogReader 对象
    one_ts: 是否单时间戳
    size  : 验证集比例（0~1）
    返回  : (train_df, validation_df)
    """
    splitter = LogSplitter(log.data)
    # 使用 timeline_trace（案例级分割），不丢弃跨分割点的案例
    train, validation = splitter.split_log('timeline_trace', size, one_ts)

    validation = pd.DataFrame(validation)
    train = pd.DataFrame(train)
    return train, validation


# 输出文件夹名称常量
OUTPUT_FILES = 'output_files'