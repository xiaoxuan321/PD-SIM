# -*- coding: utf-8 -*-
"""
固定参数训练器：直接使用预定义超参数训练模型，跳过贝叶斯优化。
可通过 parms['times_config'] 选择内置配置。
"""

import copy
import csv
import os
import numpy as np
import tensorflow as tf
import utils.support as sup

# === 内部模块：数据向量化 + 不同模型 ===
from core_modules.times_allocator import samples_creator as sc
from core_modules.times_allocator.models import basic_model as bsc
from core_modules.times_allocator.models import basic_model_nt as innt
from core_modules.times_allocator.models import dual_model as dual


class TimesModelTrainer:
    """
    固定参数训练器：超参数配置内置，由 times_config 选择。
    """

    # ========================================================
    # 🔧 配置切换开关
    # ========================================================
    # 🔥 切换至 ACR 最佳配置
    ACTIVE_CONFIG = 'BPI17W_REAL_LARGE'

    # ========================================================
    # 超参数配置字典
    # ========================================================
    HYPERPARAMETERS = {
        # ========================================================
        # 🎓 ACR 数据集：学术/行政流程最佳实践
        # 来源：基于 Loss 0.023 的历史最佳模型复现
        # ========================================================
            'acr_academic_best': {
                'n_size': 5,
                'l_size': 100,
                'lstm_act': 'tanh',
                'dense_act': 'linear',
                'optim': 'Nadam',
                'learning_rate': 0.0005,
                'batch_size': 16,
                'epochs': 200,

                # 原0.2对这么小的数据会有一定信息损失，
                # 降到0.10，让模型更充分利用有限上下文。
                'dropout': 0.10,

                # Processing不是主要矛盾，保持稳健。
                'huber_delta': 1.0,
                'proc_huber_delta': 1.0,

                # 关键：
                # 增大waiting Huber delta，
                # 让多天/多周长等待样本对训练产生更强影响。
                'wait_huber_delta': 2.0,

                # 小数据集继续冻结活动embedding，降低过拟合。
                'embedding_trainable': False,

                'imp': 1,

                'description': (
                    'ACR short-trace heavy-tail timing profile: '
                    'short history window, deterministic processing model, '
                    'tail-sensitive waiting-time model'
                )
            },
        'BPI12_W': {
            'n_size': 15,
            'l_size': 100,
            'lstm_act': 'tanh',
            'dense_act': 'linear',
            'optim': 'Nadam',
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 250,
            'dropout': 0.10,
            'huber_delta': 1.0,
            'proc_huber_delta': 1.0,
            'wait_huber_delta': 2.0,
            'embedding_trainable': False,

            'imp': 1,

            'description': (
                'BPI12W dual-LSTM timing profile: short-to-medium traces with strong '
                'rework; highly skewed processing times and multi-hour/multi-day '
                'waiting times; log1p targets, calendar/WIP features, robust '
                'processing loss and tail-sensitive waiting loss.'
            )
        },
        # ========================================================
        # MP 数据集配置 (保留备用)
        # ========================================================
        'mp_manufacturing_best_v2': {

    'n_size':8,

    'l_size':100,

    'lstm_act':'tanh',

    'dense_act':'linear',

    'optim':'Nadam',

    'learning_rate':0.0005,

    'batch_size':16,

    'epochs':300,

    'dropout':0.05,

    'proc_huber_delta':1.0,

    'wait_huber_delta':5.0,

    'embedding_trainable':True,

    'imp':1
},
        'cfs_synthetic_best': {
            'n_size': 15,  # [关键] 历史窗口 15：覆盖大部分 Trace 长度，利用合成数据的全知视角
            'l_size': 100,  # [关键] 单元数 100：对 32 种活动进行高维编码，确保无信息损失
            'lstm_act': 'tanh',  # 合成数据通常数值规范，tanh 收敛极快
            'dense_act': 'linear',
            'optim': 'Nadam',  # Nadam 对这种干净的规律性数据（Momentum）加速效果显著
            'batch_size': 32,  # 数据量大且密集（2万+事件），大 Batch 训练更稳更快
            'epochs': 200,  # 200 轮足够让 Loss 降到 0.01 以下
            'imp': 1
        },
        'CVS_REAL_BIGDATA': {

            # CVS 主流程约10~11个事件，10步已经覆盖绝大多数有效历史
            'n_size': 10,

            # 足够建模 activity + WIP + resource/calendar interaction，
            # 无需进一步扩大
            'l_size': 100,

            # 标准且稳定的 LSTM 激活
            'lstm_act': 'tanh',

            # 配合 Log1p target；允许预测接近0，逆变换后再截断
            'dense_act': 'linear',

            # 对长尾时间回归保持稳定
            'optim': 'Nadam',

            # 显式固定学习率，避免依赖优化器默认值
            'learning_rate': 0.0005,

            # 保留 rare branch / manual activity 的梯度信息
            'batch_size': 32,

            # 若 dual_model 没有 EarlyStopping，250 比300更稳妥
            'epochs': 250,

            # 轻度正则；不破坏少数人工异常路径
            'dropout': 0.10,

            # 默认兼容值
            'huber_delta': 1.0,

            # processing: 大量0 + 分钟级主体 + 少数极端尾部
            'proc_huber_delta': 1.0,

            # waiting: 多小时/多天等待属于真实业务结构，应更关注长等待
            'wait_huber_delta': 2.0,

            # CVS 数据量足够，允许 activity embedding 向时间预测任务微调
            'embedding_trainable': True,

            'imp': 1,

            'description': (
                'CVS dense pharmacy timing profile: stable short processing times '
                'with zero inflation and rare long tails; strongly bimodal/long '
                'waiting times driven by queue, calendar and manual branches.'
            )
        },
        'BPI17W_REAL_LARGE': {
            'n_size': 15,  # 保持
            'l_size': 128,  # 保持

            'lstm_act': 'tanh',
            'dense_act': 'linear',

            'optim': 'Nadam',
            'learning_rate': 0.0005,  # 保持

            # 64 -> 32
            # 让稀有的长等待样本在梯度中不那么容易被平均掉
            'batch_size': 32,

            # 最大训练轮数提高一点，但实际由 EarlyStopping 控制
            'epochs': 300,

            # 0.15 -> 0.10
            # 当前模型明显过于平滑，不需要这么强的正则
            'dropout': 0.10,

            'huber_delta': 1.0,
            'proc_huber_delta': 1.0,

            # 2.0 -> 5.0
            # waiting target 已经经过 log1p，
            # 增大 delta 可以更强惩罚长等待被严重低估
            'wait_huber_delta': 2.0,

            # 暂时保持 True，不同时改变太多变量
            'embedding_trainable': True,

            # dual_model.py 已经支持下面三个参数
            'early_stopping_patience': 40,
            'lr_patience': 10,
            'min_learning_rate': 1e-6,

            'imp': 1
        },
        'purchase_to_pay': {
            # P2P 训练案例长度中位数约为 18。窗口 8 会截断约一半
            # 的训练事件历史；窗口 12 在上下文覆盖率与训练成本间更平衡。
            'n_size': 8,

            # 4756 个训练样本可以支撑适度增大的隐藏层。
            # 100 与该日志历史较优结构一致，避免 64 单元欠拟合长分支。
            'l_size': 100,

            'lstm_act': 'tanh',
            'dense_act': 'linear',

            'optim': 'Nadam',
            'learning_rate': 0.0005,

            'batch_size': 16,
            'epochs': 300,

            'dropout': 0.1,
            'huber_delta': 1.0,
            # 处理时间分布相对稳定；等待时间存在更强的长尾和时间漂移。
            # dual_model.py 会按目标分别读取下面两个参数。
            'proc_huber_delta': 1.0,
            'wait_huber_delta': 2.0,
            'embedding_trainable': False,

            'imp': 1,

            'description': (
                'P2P长短分支混合日志：训练案例中位长度约18，'
                '测试等待时间具有显著长尾与时间分布漂移'
            )
        },

        # === 默认配置 ===
        'default': {
            'n_size': 5,
            'l_size': 50,
            'lstm_act': 'selu',
            'dense_act': 'linear',
            'optim': 'Nadam',
            'batch_size': 32,
            'epochs': 150,
            'imp': 1
        }
    }

    def __init__(self, parms, log_train, log_valdn, ac_index, embedding_file_name):
        """构造函数"""
        print("[TimesModelTrainer] 初始化固定参数训练器")

        self.log_train = copy.deepcopy(log_train)
        self.log_valdn = copy.deepcopy(log_valdn)
        self.ac_index = ac_index
        self.index_ac = {v: k for k, v in self.ac_index.items()}
        self.embedding_file_name = embedding_file_name
        self.ac_weights = None

        self.parms = parms
        self.active_config = self.parms.get(
            'times_config',
            self.ACTIVE_CONFIG
        )

        # === 优先使用 parms['times_config']，否则使用 ACTIVE_CONFIG ===
        self.hyperparams = self._select_hyperparameters()

        # 读取嵌入
        self.read_embeddings()

        # === 创建输出目录 ===
        self.temp_output = parms['output']
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)

        # === 训练结果 ===
        self.best_output = None
        self.best_parms = dict()
        self.best_loss = 1.0

        print(f"[TimesModelTrainer] 输出目录: {self.temp_output}")
        self._print_config_banner()

    def _select_hyperparameters(self):
        config_name = self.active_config

        if config_name in self.HYPERPARAMETERS:
            hyperparams = self.HYPERPARAMETERS[config_name].copy()
        else:
            print(f"[⚠️ 警告] 未找到配置 {config_name}, 将回退至 default")
            hyperparams = self.HYPERPARAMETERS['default'].copy()

        # 强制同步 self.parms
        for key in hyperparams.keys():
            if key in self.parms:
                self.parms[key] = hyperparams[key]
        return hyperparams

    def _print_config_banner(self):
        """打印配置横幅"""
        print("")
        print("=" * 70)
        print(f"  🎯 当前激活配置: {self.active_config}")
        print("-" * 70)
        print("  超参数详情:")
        for key, value in self.hyperparams.items():
            print(f"    • {key:15s} = {value}")
        print("=" * 70)
        print("")

    def execute_trials(self):
        """执行单次训练"""
        print(f"[execute_trials] 开始执行训练，使用配置: {self.active_config}")

        trial_settings = copy.deepcopy(self.parms)
        trial_settings.update(self.hyperparams)

        trial_settings['file'] = self.parms['file']
        trial_settings['all_r_pool'] = self.parms.get('all_r_pool', False)
        trial_settings['output'] = os.path.join(self.temp_output, sup.folder_id())

        if not os.path.exists(trial_settings['output']):
            os.makedirs(trial_settings['output'])

        vectorizer = sc.SequencesCreator(
            self.parms['read_options']['one_timestamp'],
            self.ac_index
        )

        train_vec = vectorizer.vectorize(
            self.parms['model_type'],
            self.log_train,
            trial_settings
        )
        valdn_vec = vectorizer.vectorize(
            self.parms['model_type'],
            self.log_valdn,
            trial_settings
        )

        self._print_vectorization_stats(train_vec, valdn_vec)

        trainer = self._get_trainer(self.parms['model_type'])
        tf.compat.v1.reset_default_graph()

        model = trainer(
            self.ac_weights,
            train_vec,
            valdn_vec,
            trial_settings
        )

        acc = self.evaluate_model(self.parms['model_type'], model, valdn_vec)
        print(f"[execute_trials] 验证集损失: {acc['loss']:.6f}")

        self.best_output = trial_settings['output']
        self.best_loss = acc['loss']
        self.best_parms = self.hyperparams.copy()

        self._print_result_banner()

    def _print_result_banner(self):
        print("")
        print("=" * 70)
        print(f"  ✅ 训练完成！")
        print("=" * 70)
        print(f"  配置名称: {self.active_config}")
        print(f"  验证损失: {self.best_loss:.6f}")
        print(f"  模型路径: {self.best_output}")
        print("=" * 70)
        print("")

    def _print_vectorization_stats(self, train_vec, valdn_vec):
        if self.parms['model_type'] in ['basic', 'inter', 'inter_nt']:
            train_samples = len(train_vec['pref']['ac_index']) if 'pref' in train_vec else 0
            valdn_samples = len(valdn_vec['pref']['ac_index']) if 'pref' in valdn_vec else 0
            print(f"[execute_trials] 向量化完成: 训练集={train_samples}, 验证集={valdn_samples}")
        elif self.parms['model_type'] == 'dual_inter':
            proc_train = len(train_vec['proc_model']['pref']['ac_index']) if 'proc_model' in train_vec else 0
            wait_train = len(train_vec['waiting_model']['pref']['ac_index']) if 'waiting_model' in train_vec else 0
            print(f"[execute_trials] 向量化完成: 处理模型训练集={proc_train}, 等待模型训练集={wait_train}")

    def _get_trainer(self, model_type):
        if model_type in ['basic', 'inter']:
            return bsc._training_model
        elif model_type == 'inter_nt':
            return innt._training_model
        elif model_type == 'dual_inter':
            return dual._training_model
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    def read_embeddings(self):
        path = os.path.join(
            self.parms['embedded_path'],
            self.embedding_file_name
        )
        if os.path.exists(path):
            self.ac_weights = self.load_embedded(
                self.index_ac,
                self.parms['embedded_path'],
                self.embedding_file_name
            )

    def evaluate_model(self, model_type, model, valdn_vec):
        if model_type in ['inter', 'basic']:
            return model.evaluate(
                x={
                    'ac_input': valdn_vec['pref']['ac_index'],
                    'features': valdn_vec['pref']['features']
                },
                y={'time_output': valdn_vec['next']['expected']},
                return_dict=True
            )
        elif model_type == 'inter_nt':
            return model.evaluate(
                x={
                    'ac_input': valdn_vec['pref']['ac_index'],
                    'n_ac_input': valdn_vec['pref']['n_ac_index'],
                    'features': valdn_vec['pref']['features']
                },
                y={'time_output': valdn_vec['next']},
                return_dict=True
            )
        elif model_type == 'dual_inter':
            acc_proc = model['proc_model']['model'].evaluate(
                x={
                    'ac_input': valdn_vec['proc_model']['pref']['ac_index'],
                    'features': valdn_vec['proc_model']['pref']['features']
                },
                y={'time_output': valdn_vec['proc_model']['next']},
                return_dict=True
            )
            acc_wait = model['wait_model']['model'].evaluate(
                x={
                    'ac_input': valdn_vec['waiting_model']['pref']['ac_index'],
                    'features': valdn_vec['waiting_model']['pref']['features']
                },
                y={'time_output': valdn_vec['waiting_model']['next']},
                return_dict=True
            )
            combined_loss = 0.5 * acc_proc['loss'] + 0.5 * acc_wait['loss']
            return {'loss': combined_loss}
        else:
            raise ValueError(f'不存在的模型类型: {model_type}')

    @staticmethod
    def load_embedded(index, input_folder, filename):
        weights = []
        path = os.path.join(input_folder, filename)
        try:
            with open(path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for row in reader:
                    cat_ix = int(row[0])
                    if cat_ix in index and index[cat_ix] == row[1].strip():
                        weights.append([float(x) for x in row[2:]])
                csvfile.close()
            return np.array(weights) if len(weights) > 0 else None
        except Exception as e:
            print(f"[load_embedded] 加载嵌入时出错: {e}")
            return None

    @classmethod
    def list_available_configs(cls):
        print("=" * 70)
        print("  📋 可用的配置 (ACTIVE_CONFIG 可选值):")
        print("=" * 70)
        for name in cls.HYPERPARAMETERS.keys():
            current = " ⭐ (当前激活)" if name == cls.ACTIVE_CONFIG else ""
            print(f"  • {name}{current}")
        print("=" * 70)
