# -*- coding: utf-8 -*-


# === 基础库 ===
import copy
import csv
import os
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import utils.support as sup

# === 超参数优化库 ===
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe

# === 内部模块：数据向量化 + 不同模型 ===
from core_modules.times_allocator import samples_creator as sc
from core_modules.times_allocator.models import basic_model as bsc
from core_modules.times_allocator.models import basic_model_nt as innt
from core_modules.times_allocator.models import dual_model as dual


class TimesModelOptimizer:
    """
    超参数优化器：
    使用 Bayesian Optimization（TPE）自动搜索最优超参数组合
    """

    class Decorators(object):
        """装饰器：包装执行方法，捕获异常并返回状态"""

        @classmethod
        def safe_exec(cls, method):
            """
            Decorator to safe execute methods and return the state
            ----------
            method : Any method.
            Returns
            -------
            dict : execution status
            """

            def safety_check(*args, **kw):
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        response['status'] = STATUS_FAIL
                return response
            return safety_check

    # ------------------------------------------------------------------
    # 构造函数：准备数据 & 路径 & 试验记录
    # ------------------------------------------------------------------
    def __init__(self, parms, log_train, log_valdn, ac_index, embedding_file_name):
        """构造函数"""
        self.space = self.define_search_space(parms)  # === 定义搜索空间 ===
        self.log_train = copy.deepcopy(log_train)     # === 训练日志 ===
        self.log_valdn = copy.deepcopy(log_valdn)     # === 验证日志 ===
        self.ac_index = ac_index                      # === 活动索引字典 ===
        self.index_ac = {v: k for k, v in self.ac_index.items()}  # === 反向索引 ===
        self.embedding_file_name = embedding_file_name  # === 词向量文件名 ===
        self.ac_weights = None                        # === 预训练嵌入矩阵 ===

        # === 读取配置 ===
        self.parms = parms
        self.read_embeddings()                        # === 加载活动嵌入 ===
        self.temp_output = parms['output']            # === 临时输出目录 ===
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)

        # === 结果保存文件 ===
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()

        # === 试验记录器 ===
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_parms = dict()
        self.best_loss = 1

    # ------------------------------------------------------------------
    # 1. 定义超参数搜索空间
    # ------------------------------------------------------------------
    @staticmethod
    def define_search_space(parms):
        """把 parms 里给出的网格变成 HyperOpt 的搜索空间"""
        space = {'n_size': hp.choice('n_size', parms['n_size']),
                 'l_size': hp.choice('l_size', parms['l_size']),
                 'lstm_act': hp.choice('lstm_act', parms['lstm_act']),
                 'dense_act': hp.choice('dense_act', parms['dense_act']),
                 'optim': hp.choice('optim', parms['optim']),
                 'imp': parms['imp'], 'file': parms['file'],
                 'batch_size': parms['batch_size'], 'epochs': parms['epochs']}
        return space

    # ------------------------------------------------------------------
    # 2. 主流程：启动 Bayesian Optimization
    # ------------------------------------------------------------------
    def execute_trials(self):
        """执行多次试验，寻找最优超参数"""
        def exec_pipeline(trial_stg):
            print(trial_stg)   # === 打印当前超参数 ===
            trial_stg['all_r_pool'] = self.parms['all_r_pool']  # === 资源池 ===
            status = STATUS_OK

            # === 临时路径重定义 ===
            rsp = self._temp_path_redef(trial_stg, status=status)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg

            # === 数据向量化 ===
            vectorizer = sc.SequencesCreator(self.parms['read_options']['one_timestamp'], self.ac_index)
            train_vec = vectorizer.vectorize(self.parms['model_type'], self.log_train, trial_stg)
            valdn_vec = vectorizer.vectorize(self.parms['model_type'], self.log_valdn, trial_stg)

            # === 训练模型 ===
            trainer = self._get_trainer(self.parms['model_type'])
            tf.compat.v1.reset_default_graph()  # === 清空图 ===
            model = trainer(self.ac_weights, train_vec, valdn_vec, trial_stg)

            # === 评估验证集 ===
            acc = self.evaluate_model(self.parms['model_type'], model, valdn_vec)
            print(acc)

            # === 保存本次结果 ===
            rsp = self._define_response(trial_stg, status, acc['loss'])
            print("-- End of trial --")
            return rsp

        # === 开始 TPE 优化 ===
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.parms['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)

        # === 保存最优结果 ===
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                       .sort_values('loss', ascending=True))
            result = results[results.status == 'ok'].head(1).iloc[0]
            self.best_output = result.output
            self.best_loss = result.loss
            self.best_parms = {k: self.parms[k][v] for k, v in best.items()}
        except Exception as e:
            print(e)
            pass

    # ------------------------------------------------------------------
    # 3. 根据 model_type 返回训练器
    # ------------------------------------------------------------------
    def _get_trainer(self, model_type):
        """选择对应模型的训练函数"""
        if model_type in ['basic', 'inter']:
            return bsc._training_model
        elif model_type == 'inter_nt':
            return innt._training_model
        elif model_type == 'dual_inter':
            return dual._training_model
        else:
            raise ValueError(model_type)

    # ------------------------------------------------------------------
    # 4. 临时输出路径重定义（装饰器包裹）
    # ------------------------------------------------------------------
    @Decorators.safe_exec
    def _temp_path_redef(self, settings, **kwargs) -> dict:
        """为每次试验创建独立输出文件夹"""
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
        return settings

    # ------------------------------------------------------------------
    # 5. 读取活动嵌入矩阵
    # ------------------------------------------------------------------
    def read_embeddings(self):
        """加载预训练活动嵌入"""
        path = os.path.join(self.parms['embedded_path'], self.embedding_file_name)
        if os.path.exists(path):
            self.ac_weights = self.load_embedded(self.index_ac, self.parms['embedded_path'], self.embedding_file_name)

    # ------------------------------------------------------------------
    # 6. 记录本次试验结果
    # ------------------------------------------------------------------
    def _define_response(self, parms, status, loss, **kwargs) -> None:
        """把试验结果写进 csv"""
        print(loss)
        response = dict()
        measurements = list()
        data = {'n_size': parms['n_size'],
                'l_size': parms['l_size'],
                'lstm_act': parms['lstm_act'],
                'dense_act': parms['dense_act'],
                'optim': parms['optim']}
        response['output'] = parms['output']

        if status == STATUS_OK:
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL
            measurements.append({**{'loss': loss,
                                    'sim_metric': 'mae',
                                    'status': response['status']},
                                 **data})
        else:
            response['status'] = status
            measurements.append({**{'loss': 1,
                                    'sim_metric': 'mae',
                                    'status': response['status']},
                                 **data})

        # === 写 csv ===
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response

    # ------------------------------------------------------------------
    # 7. 根据模型类型评估验证集
    # ------------------------------------------------------------------
    def evaluate_model(self, model_type, model, valdn_vec):
        """返回验证集上的 loss"""
        if model_type in ['inter', 'basic']:
            return model.evaluate(
                x={'ac_input': valdn_vec['pref']['ac_index'],
                   'features': valdn_vec['pref']['features']},
                y={'time_output': valdn_vec['next']['expected']},
                return_dict=True)
        elif model_type == 'inter_nt':
            return model.evaluate(
                x={'ac_input': valdn_vec['pref']['ac_index'],
                   'n_ac_input': valdn_vec['pref']['n_ac_index'],
                   'features': valdn_vec['pref']['features']},
                y={'time_output': valdn_vec['next']},
                return_dict=True)
        elif model_type == 'dual_inter':
            acc_proc = model['proc_model']['model'].evaluate(
                x={'ac_input': valdn_vec['proc_model']['pref']['ac_index'],
                   'features': valdn_vec['proc_model']['pref']['features']},
                y={'time_output': valdn_vec['proc_model']['next']},
                return_dict=True)
            acc_wait = model['wait_model']['model'].evaluate(
                x={'ac_input': valdn_vec['waiting_model']['pref']['ac_index'],
                   'features': valdn_vec['waiting_model']['pref']['features']},
                y={'time_output': valdn_vec['waiting_model']['next']},
                return_dict=True)
            return {'loss': 0.5 * acc_proc['loss'] + 0.5 * acc_wait['loss']}
        else:
            raise ValueError('Unexistent model')

    # ------------------------------------------------------------------
    # 8. 静态方法：加载预训练嵌入
    # ------------------------------------------------------------------
    @staticmethod
    def load_embedded(index, input_folder, filename):
        """从 csv 加载预训练嵌入矩阵"""
        weights = []
        path = os.path.join(input_folder, filename)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                cat_ix = int(row[0])
                if index[cat_ix] == row[1].strip():
                    weights.append([float(x) for x in row[2:]])
            csvfile.close()
        return np.array(weights)