# -*- coding: utf-8 -*-

import pandas as pd
import itertools
import numpy as np

from nltk.util import ngrams


# 移除 keras.utils.np_utils 引用，不再需要 One-Hot
# from keras.utils import np_utils as ku


class SequencesCreator():

    def __init__(self, one_timestamp, ac_index):
        """constructor"""
        self.one_timestamp = one_timestamp
        self.ac_index = ac_index

    def vectorize(self, model_type, log, params):
        vectorizer = self._get_vectorizer(model_type)
        return vectorizer(log, params)

    def _get_vectorizer(self, model_type):
        if model_type in ['basic', 'inter']:
            return self._vectorize_seq
        elif model_type == 'dual_inter':
            return self._dual_vectorize_seq
        elif model_type == 'inter_nt':
            return self._vectorize_nt_seq
        else:
            raise ValueError('Unexistent vectorizer')

    def _vectorize_seq(self, log, params):
        """
        [修改版] 通用单模型向量化
        移除显式 One-Hot，直接使用预处理好的特征
        """
        vec = {'pref': dict(), 'next': dict()}
        columns = [x for x in log.columns if x != 'caseid']
        log = self.reformat_events(log, columns, self.one_timestamp)

        # N-gram 生成
        for i, _ in enumerate(log):
            for x in columns:
                serie = list(ngrams(log[i][x], params['n_size'],
                                    pad_left=True, left_pad_symbol=0))
                if x in ['processing_time', 'waiting_time']:
                    y_serie = [x[-1] for x in serie]
                    vec['next'][x] = (vec['next'][x] + y_serie if i > 0 else y_serie)
                    serie.insert(0, tuple([0 for i in range(params['n_size'])]))
                    serie.pop(-1)
                vec['pref'][x] = (vec['pref'][x] + serie if i > 0 else serie)

        # --- 修改：移除 st_weekday 的 One-Hot 处理 ---
        # 此时 columns 列表里已经包含了 st_weekday_sin, st_weekday_cos 等
        # 它们会被当做普通的数值特征处理，直接 reshape 即可

        # 处理 ac_index
        vec['pref']['ac_index'] = np.array(vec['pref']['ac_index'])
        if 'ac_index' in columns: columns.remove('ac_index')

        # 移除可能残留的旧列名 (以防万一)
        if 'st_weekday' in columns: columns.remove('st_weekday')

        # 初始化 features 数组 (此时还没有数据)
        features = None

        # 遍历所有剩余特征列 (包括新的 Sin/Cos 列)
        for value in columns:
            vec['pref'][value] = np.array(vec['pref'][value])
            vec['pref'][value] = vec['pref'][value].reshape(
                (vec['pref'][value].shape[0], vec['pref'][value].shape[1], 1))

            # 拼接特征
            if features is None:
                features = vec['pref'][value]
            else:
                features = np.concatenate((features, vec['pref'][value]), axis=2)

            # 清理
            vec['pref'].pop(value, None)

        vec['pref']['features'] = features

        # Output array
        vec['next']['processing_time'] = (
            vec['next']['processing_time']
                .reshape((vec['next']['processing_time'].shape[0], 1)))
        vec['next']['waiting_time'] = (
            vec['next']['waiting_time']
                .reshape((vec['next']['waiting_time'].shape[0], 1)))
        vec['next']['expected'] = np.concatenate(
            (vec['next']['processing_time'],
             vec['next']['waiting_time']), axis=1)
        vec['next'].pop('processing_time', None)
        vec['next'].pop('waiting_time', None)
        return vec

    def _dual_vectorize_seq(self, log, params):
        """
        [修改版] 双模型向量化 (Dual Model)
        核心修改：移除 create_vector 中的 One-Hot 逻辑
        """
        ngram_size = params['n_size']
        vec = {'proc_model': dict(), 'waiting_model': dict()}

        # 修改：移除了 week_col 参数
        def create_vector(df, exp_col):
            dt_prefixes = list()
            dt_expected = list()
            cols = list(df.columns)

            # 按案例分组生成 N-gram
            for key, group in df.groupby('caseid'):
                dt_prefix = pd.DataFrame(0, index=range(ngram_size),
                                         columns=cols + ['ngram_num'])
                dt_prefix['caseid'] = key
                dt_prefix = pd.concat([dt_prefix, group], axis=0)
                dt_prefix = dt_prefix.iloc[:-1]
                dt_expected.append(group[exp_col])
                for nr_events in range(0, len(group)):
                    tmp = dt_prefix.iloc[nr_events:nr_events + ngram_size].copy()
                    tmp['ngram_num'] = nr_events
                    dt_prefixes.append(tmp)

            dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
            dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)

            # --- 修改：彻底移除 One-Hot Encoding ---
            # 因为 df 已经包含了 _sin, _cos, _oc 等数值列，直接使用即可

            # 准备 Reshape
            num_samples = len(
                dt_prefixes[['caseid', 'ngram_num']].drop_duplicates())
            dt_prefixes.drop(columns={'caseid', 'ngram_num'}, inplace=True)
            num_columns = len(dt_prefixes.columns)

            dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples,
                                                         ngram_size,
                                                         num_columns)
            dt_expected = dt_expected.to_numpy().reshape((num_samples, 1))
            return dt_prefixes, dt_expected

        caseid_col = ['caseid']

        # 这里的筛选逻辑会自动包含 st_weekday_sin, st_daytime_cos 等
        st_cols = (caseid_col + ['ac_index', 'processing_time'] +
                   [c_n for c_n in log.columns if 'st_' in c_n])
        end_cols = (caseid_col + ['n_ac_index', 'waiting_time'] +
                    [c_n for c_n in log.columns if 'end_' in c_n])

        # 调用时不再传 week_col
        st_train, st_expected = create_vector(log[st_cols], 'processing_time')

        # 组装 Processing Model 输入
        vec['proc_model']['pref'] = dict()
        # 假设 ac_index 是第一列 (索引0)
        vec['proc_model']['pref']['ac_index'] = st_train[:, :, 0]
        # 其余列为特征 (WIP, Resource, Time_Sin, Time_Cos...)
        vec['proc_model']['pref']['features'] = st_train[:, :, 1:]
        vec['proc_model']['next'] = st_expected

        # 调用 Waiting Model 向量化
        end_train, end_expected = create_vector(log[end_cols], 'waiting_time')

        # 组装 Waiting Model 输入
        vec['waiting_model'] = dict()  # 修复原代码笔误 vec['waiting_model']: dict()
        vec['waiting_model']['pref'] = dict()
        vec['waiting_model']['pref']['ac_index'] = end_train[:, :, 0]
        vec['waiting_model']['pref']['features'] = end_train[:, :, 1:]
        vec['waiting_model']['next'] = end_expected

        return vec

    def _vectorize_nt_seq(self, log, params):
        """
        [修改版] Next-Task 向量化
        """
        ngram_size = params['n_size']

        # 移除显式的 week_col
        caseid_col = ['caseid']
        exp_col = ['processing_time', 'waiting_time']
        cols = (caseid_col + ['ac_index', 'n_ac_index'] + exp_col +
                [c_n for c_n in log.columns if 'st_' in c_n])

        log = log[cols]
        dt_prefixes = list()
        dt_expected = list()

        for key, group in log.groupby('caseid'):
            dt_prefix = pd.DataFrame(0, index=range(ngram_size),
                                     columns=cols + ['ngram_num'])
            dt_prefix['caseid'] = key
            dt_prefix = pd.concat([dt_prefix, group], axis=0)
            dt_prefix = dt_prefix.iloc[:-1]
            dt_expected.append(group[exp_col])
            for nr_events in range(0, len(group)):
                tmp = dt_prefix.iloc[nr_events:nr_events + ngram_size].copy()
                tmp['ngram_num'] = nr_events
                dt_prefixes.append(tmp)

        dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
        dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)

        # --- 修改：移除 One-Hot ---
        num_samples = len(
            dt_prefixes[['caseid', 'ngram_num']].drop_duplicates())
        dt_prefixes.drop(columns={'caseid', 'ngram_num'}, inplace=True)
        num_columns = len(dt_prefixes.columns)

        dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples,
                                                     ngram_size,
                                                     num_columns)
        dt_expected = dt_expected.to_numpy().reshape((num_samples, 2))

        vec = {'pref': dict(), 'next': dt_expected}

        # 根据列顺序提取特征
        # 假设 0: ac_index, 1: n_ac_index, 2+: features
        vec['pref']['ac_index'] = dt_prefixes[:, :, 0]
        vec['pref']['n_ac_index'] = dt_prefixes[:, :, 1]
        vec['pref']['features'] = dt_prefixes[:, :, 2:]

        return vec

    # =============================================================================
    # Reformat events
    # =============================================================================
    def reformat_events(self, log, columns, one_timestamp):
        """Creates series of activities, roles and relative times per trace."""
        temp_data = list()
        log_df = log.to_dict('records')
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'waiting_time':
                    serie.pop(0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data