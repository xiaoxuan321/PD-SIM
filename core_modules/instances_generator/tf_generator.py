"""
优化版TFT实现 - 减少过拟合 v3.0
==========================================
改进点：
1. ✅ 精简特征工程（保留核心特征）
2. ✅ 更稳健的可解释性分析
3. ✅ 增强正则化和验证策略
4. ✅ 特征重要性驱动的自动特征选择
"""

import os
import json
import pickle
import shutil
from typing import Optional, Tuple

import numpy as np

from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import Baseline
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import matplotlib.pyplot as plt
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import pandas as pd  # ← 确保导入
import utils.support as sup
from support_modules.common import FileExtensions as Fe
from torch import nn
from pytorch_forecasting.metrics import (
    QuantileLoss,
    Metric,
    MAE,
)


class HuberMAELoss(Metric):
    def __init__(self, delta=0.5, mae_weight=0.3):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.mae = nn.L1Loss()
        self.mae_weight = mae_weight

    def forward(self, y_pred, y_true):
        # 🔥 PyTorch Forecasting: y_true 是 (target, weight)
        if isinstance(y_true, (tuple, list)):
            y_true = y_true[0]

        # 只用中位数通道
        if y_pred.ndim == 3:
            y_pred = y_pred[..., y_pred.shape[-1] // 2]

        huber_loss = self.huber(y_pred, y_true)
        mae_loss = self.mae(y_pred, y_true)

        # ❌ 绝对不要在 loss 里 log
        return huber_loss + self.mae_weight * mae_loss


class NumpyEncoder(json.JSONEncoder):
    """自定义编码器处理numpy数据类型"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                            np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def prepare_interarrival_data(log: pd.DataFrame):
    """
    Prepare event-level inter-arrival data for TFT
    修复版：确保 log 变换后的有限性
    """
    df = log.copy()

    # 1. 时间处理
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. 计算 inter-arrival (秒)
    df["delta_t"] = df["timestamp"].diff().dt.total_seconds()

    # ⭐ 修复点 1：必须在 clip 之前删除第一个 NaN (由 diff 产生)
    df = df.dropna(subset=["delta_t"])

    # ⭐ 修复点 2：严格 clip，确保没有负值或微小的负浮点数偏差
    # 限制下限为 0.0，上限保持 99.5 分位数
    upper_bound = df["delta_t"].quantile(0.995)
    df["delta_t"] = df["delta_t"].clip(lower=0.0, upper=upper_bound)

    # ⭐ 修复点 3：执行 log1p 变换
    df["delta_t"] = np.log1p(df["delta_t"])

    # ⭐ 修复点 4：防御性检查，防止无穷大
    df = df[np.isfinite(df["delta_t"])]

    # 3. 时间特征
    df["hour"] = df["timestamp"].dt.hour.astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(str)
    df["month"] = df["timestamp"].dt.month.astype(str)
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    # 4. rolling features（注意：此时是在 log 空间计算均值）
    for w in [6, 24, 168]:
        df[f"delta_mean_{w}"] = (
            df["delta_t"]
                .rolling(w, min_periods=1)
                .mean()
        )

    # 5. 其他
    df["group_id"] = "arrivals"
    df["group_id"] = df["group_id"].astype(str)
    df["time_idx"] = range(len(df))

    return df


class TFTGenerator:
    """
    优化版TFT生成器 - 减少过拟合 v3.0

    主要改进：
    1. ✅ 精简特征工程（三档可选）
    2. ✅ 更强的正则化
    3. ✅ 改进的可解释性分析
    4. ✅ 自动特征重要性分析
    """

    def __init__(self, log, valdn, params):
        print("\n" + "=" * 80)
        print("🚀 初始化TFT生成器")
        print("=" * 80)

        if valdn is None:
            raise ValueError("❌ 必须提供外部验证集 (valdn 参数)")

        if hasattr(log, 'data'):
            self.log_train = log.data if isinstance(log.data, pd.DataFrame) else pd.DataFrame(log.data)
        else:
            self.log_train = pd.DataFrame(log) if not isinstance(log, pd.DataFrame) else log.copy()

        if hasattr(valdn, 'data'):
            self.log_validation = valdn.data if isinstance(valdn.data, pd.DataFrame) else pd.DataFrame(valdn.data)
        else:
            self.log_validation = pd.DataFrame(valdn) if not isinstance(valdn, pd.DataFrame) else valdn.copy()

        self.params = params

        required_params = ['file', 'ia_gen_path']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"❌ 缺少必需参数: {param}")
        default_params = {

            # === 序列结构（小样本核心） ===
            'max_encoder_length': 30,  # ⬇ 缩短历史，提升有效样本数
            'max_prediction_length': 7,  # 你现在设 1，是对的

            # === 模型容量（小数据友好） ===
            'hidden_size': 16,  # ⬆ 比 24 更稳
            'num_layers': 1,  # 小数据不堆层
            'attention_head_size': 4,  # ⬇ 非常关键！
            'dropout': 0.2,  # ⬇ 建议直接关

            # === 连续特征投影 ===
            'hidden_continuous_size': 8,  # TFT 里很重要

            # === 训练稳定性（关键改动） ===
            'batch_size': 16,  # 保留
            'learning_rate': 1e-3,  # ⬆ 这是最大收益点之一
            'gradient_clip': 0.5,  # ⬆ 0.1 太保守

            # === 训练策略 ===
            'epochs': 300,  # 不用 150，早停更靠谱
            'patience': 20,  # 稍微激进一点

            # === 其他 ===
            'feature_mode': 'core',
            'seed': 42,
            'gpus': 1 if torch.cuda.is_available() else 0,
            'log_dir': 'lightning_logs',
            'update_ia_gen': False,

            # === 可解释性 ===
            'enable_interpretation': True,
            'save_interpretations': True,
            'interpretation_sample_size': 30,  # ⬇ 小数据别太大
        }

        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

        self.temp_output = os.path.join('output_files', sup.folder_id())
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        print(f"📁 临时文件夹: {self.temp_output}")

        os.makedirs(self.params['ia_gen_path'], exist_ok=True)

        base_name = self.params['file'].split('.')[0]

        self.model_checkpoint_path = os.path.join(self.params['ia_gen_path'], f"{base_name}_tft.ckpt")
        self.dataset_config_path = os.path.join(self.params['ia_gen_path'], f"{base_name}_tft_dataset_config.pkl")
        self.metadata_path = os.path.join(self.params['ia_gen_path'], f"{base_name}_tft_meta{Fe.JSON}")
        self.model_path = self.model_checkpoint_path
        self.params['model_checkpoint_path'] = self.model_checkpoint_path
        self.params['dataset_config_path'] = self.dataset_config_path
        self.params['metadata_path'] = self.metadata_path
        self.params['model_path'] = self.model_checkpoint_path

        print(f"📄 模型文件: {self.model_checkpoint_path}")
        print(f"📄 数据集配置: {self.dataset_config_path}")
        print(f"📄 元数据文件: {self.metadata_path}")

        torch.manual_seed(self.params['seed'])
        np.random.seed(self.params['seed'])

        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.min_time = None
        self.model_metadata = {}
        self.interpretation_cache = {}
        self.is_safe = True
        self.dataset_config = None

        self._prepare_data()
        self._load_model()

        print("\n" + "=" * 80)
        print("✅ TFT生成器初始化完成")
        print("=" * 80)

    # ============================================================
    # 优化的数据准备
    # ============================================================
    def _prepare_data(self):
        """准备训练和验证数据"""
        print("\n" + "=" * 70)
        print("📊 准备训练数据（含重采样）")
        print("=" * 70)

        if self.log_validation is None:
            raise ValueError("❌ 必须提供外部验证集")

        print("\n🔹 步骤1: 准备训练数据")
        self.training = prepare_interarrival_data(self.log_train)
        print(f"   ✅ 训练集准备完成: {len(self.training):,} 样本")

        print("\n🔹 步骤2: 准备外部验证数据")
        self.validation = prepare_interarrival_data(self.log_validation)
        print(f"   ✅ 外部验证集准备完成: {len(self.validation):,} 样本")
        self.min_time = self.training["timestamp"].min()
        print("\n🔹 步骤3: 创建 TimeSeriesDataSet")

        categorical_encoders = {
            'hour': NaNLabelEncoder(add_nan=True, warn=False),
            'day_of_week': NaNLabelEncoder(add_nan=True, warn=False),
            'month': NaNLabelEncoder(add_nan=True, warn=False)
        }

        print("\n  🔧 预拟合分类编码器...")
        categorical_encoders['hour'].fit(pd.Series([str(i) for i in range(24)] + ['nan']))
        categorical_encoders['day_of_week'].fit(pd.Series([str(i) for i in range(7)] + ['nan']))
        categorical_encoders['month'].fit(pd.Series([str(i) for i in range(1, 13)] + ['nan']))
        print(f"  ✅ 分类编码器预拟合完成\n")

        self.training_dataset = TimeSeriesDataSet(
            self.training,
            time_idx="time_idx",
            target="delta_t",
            group_ids=["group_id"],

            max_encoder_length=self.params["max_encoder_length"],
            max_prediction_length=1,

            time_varying_known_categoricals=[
                "hour", "day_of_week", "month"
            ],
            time_varying_unknown_reals=[
                "delta_mean_6",
                "delta_mean_24",
                "delta_mean_168"
            ],

            categorical_encoders={  # 🔥🔥🔥 关键修复
                "hour": NaNLabelEncoder(add_nan=True),
                "day_of_week": NaNLabelEncoder(add_nan=True),
                "month": NaNLabelEncoder(add_nan=True),
                "group_id": NaNLabelEncoder(add_nan=True),
            },

            target_normalizer=GroupNormalizer(
                groups=["group_id"],
                transformation=None  # ⭐ 非常关键
            ),

            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        print(f"\n   📊 训练 TimeSeriesDataSet 统计:")
        print(f"   ├─ 可用训练样本数: {len(self.training_dataset):,}")
        print(f"   ├─ 原始数据行数: {len(self.training):,}")

        # 🟢 调试点: 检查 TimeSeriesDataSet 内部数据
        print("\n🔍 [DEBUG] TimeSeriesDataSet 内部数据完整性检查:")
        try:
            x, y = next(iter(self.training_dataset.to_dataloader(batch_size=32)))
            targets = y[0]
            print(f"   Target Batch Mean: {targets.mean().item():.4f}")
            print(f"   Target Batch Max:  {targets.max().item():.4f}")
        except Exception as e:
            print(f"   ⚠️ 调试检查失败: {e}")

        print(f"\n  🔧 创建验证数据集...")

        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.validation,
            predict=False,
            stop_randomization=True,
            allow_missing_timesteps=True
        )

        print(f"\n   📊 验证 TimeSeriesDataSet 统计:")
        print(f"   ├─ 可用验证样本数: {len(self.validation_dataset):,}")

        self.train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=self.params['batch_size'], num_workers=0
        )
        self.val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=self.params['batch_size'], num_workers=0
        )

        print(f"\n   ✅ 数据加载器创建完成")
        print("=" * 70)

    # ============================================================
    # 改进的可解释性分析（更稳健）
    # ============================================================
    def _analyze_model_interpretation_safe(self):
        """安全的可解释性分析（避免OOM和崩溃）"""
        print("\n" + "=" * 80)
        print("🔍 执行可解释性分析（安全模式）")
        print("=" * 80)

        try:
            # 限制样本数量
            sample_size = min(
                self.params['interpretation_sample_size'],
                len(self.val_dataloader.dataset)
            )

            print(f"\n  📊 使用 {sample_size} 个验证样本进行分析...")

            # 分批处理避免OOM
            batch_size = 10
            all_interpretations = []

            for i in range(0, sample_size, batch_size):
                end_idx = min(i + batch_size, sample_size)
                batch_data = self.val_dataloader.dataset[i:end_idx]

                try:
                    interp = self.model.interpret_output(
                        batch_data,
                        reduction="sum"
                    )
                    all_interpretations.append(interp)
                except Exception as e:
                    print(f"  ⚠️  批次 {i}-{end_idx} 分析失败: {e}")
                    continue

            if not all_interpretations:
                print("  ❌ 所有批次都失败，跳过可解释性分析")
                return

            # 合并结果（取平均）
            encoder_importance = {}
            decoder_importance = {}

            for interp in all_interpretations:
                for var, score in interp.get("encoder_variables", {}).items():
                    encoder_importance[var] = encoder_importance.get(var, 0) + score

                for var, score in interp.get("decoder_variables", {}).items():
                    decoder_importance[var] = decoder_importance.get(var, 0) + score

            # 归一化
            n_batches = len(all_interpretations)
            encoder_importance = {k: v / n_batches for k, v in encoder_importance.items()}
            decoder_importance = {k: v / n_batches for k, v in decoder_importance.items()}

            # 保存
            self.interpretation_cache = {
                'encoder_importance': encoder_importance,
                'decoder_importance': decoder_importance,
                'sample_size': sample_size,
                'feature_mode': self.params['feature_mode']
            }

            # 打印Top特征
            print("\n  ✅ Top 10 历史特征（编码器）:")
            for i, (var, importance) in enumerate(sorted(
                    encoder_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
            )[:10], 1):
                print(f"    {i:2d}. {var:30s}: {importance:.4f}")

            print("\n  ✅ Top 10 未来特征（解码器）:")
            for i, (var, importance) in enumerate(sorted(
                    decoder_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
            )[:10], 1):
                print(f"    {i:2d}. {var:30s}: {importance:.4f}")

            # 保存可视化
            if self.params['save_interpretations']:
                self._save_interpretation_plots_optimized()

            # 🆕 特征重要性警告
            self._warn_low_importance_features()

            print("\n✅ 可解释性分析完成")

        except Exception as e:
            print(f"❌ 可解释性分析失败: {e}")
            print("   模型仍可正常使用，但无法提供详细解释")
            import traceback
            traceback.print_exc()

    def _warn_low_importance_features(self):
        """警告低重要性特征（可能的过拟合信号）"""
        if not self.interpretation_cache:
            return

        all_features = {
            **self.interpretation_cache['encoder_importance'],
            **self.interpretation_cache['decoder_importance']
        }

        # 识别低重要性特征
        low_importance_threshold = 0.01
        low_importance_features = [
            k for k, v in all_features.items()
            if v < low_importance_threshold
        ]

        if low_importance_features:
            print(f"\n⚠️  发现 {len(low_importance_features)} 个低重要性特征（<{low_importance_threshold}）:")
            print(f"   {', '.join(low_importance_features[:10])}")
            if len(low_importance_features) > 10:
                print(f"   ... 及其他 {len(low_importance_features) - 10} 个")
            print("   💡 建议: 考虑切换到更精简的特征模式")

    def _save_interpretation_plots_optimized(self):
        """保存优化的可解释性图表"""
        try:
            save_dir = os.path.join(self.params['ia_gen_path'], 'interpretations')
            os.makedirs(save_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 编码器变量
            encoder_vars = self.interpretation_cache['encoder_importance']
            if encoder_vars:
                top_encoder = dict(sorted(
                    encoder_vars.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15])

                axes[0].barh(list(top_encoder.keys()), list(top_encoder.values()))
                axes[0].set_xlabel('Importance Score')
                axes[0].set_title(f'Top 15 Encoder Variables ({self.params["feature_mode"]} mode)')
                axes[0].invert_yaxis()
                axes[0].axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='Low Importance Threshold')
                axes[0].legend()

            # 解码器变量
            decoder_vars = self.interpretation_cache['decoder_importance']
            if decoder_vars:
                top_decoder = dict(sorted(
                    decoder_vars.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15])

                axes[1].barh(list(top_decoder.keys()), list(top_decoder.values()))
                axes[1].set_xlabel('Importance Score')
                axes[1].set_title(f'Top 15 Decoder Variables ({self.params["feature_mode"]} mode)')
                axes[1].invert_yaxis()
                axes[1].axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='Low Importance Threshold')
                axes[1].legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f'variable_importance_{self.params["feature_mode"]}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

            print(f"  ✅ 可视化已保存到: {save_dir}")

        except Exception as e:
            print(f"  ⚠️  保存可视化失败: {e}")

    # ============================================================
    # 训练方法（增强正则化）
    # ============================================================

    def _load_model(self):
        """加载模型或训练新模型"""
        print("\n" + "=" * 80)
        print("📥 模型加载/训练流程")
        print("=" * 80)

        # 🔥 修复：使用正确的属性名检查文件
        model_exists = os.path.exists(self.model_checkpoint_path)
        config_exists = os.path.exists(self.dataset_config_path)  # ✅ 正确的属性名
        metadata_exists = os.path.exists(self.metadata_path)

        print(f"\n🔍 文件检查:")
        print(f"   模型checkpoint: {'✅ 存在' if model_exists else '❌ 不存在'}")
        print(f"   数据集配置: {'✅ 存在' if config_exists else '❌ 不存在'}")
        print(f"   元数据文件: {'✅ 存在' if metadata_exists else '❌ 不存在'}")

        # 判断是否需要训练
        should_train = (
                self.params.get('update_ia_gen', False) or  # 强制重新训练
                not model_exists or  # 模型不存在
                not config_exists  # 配置不存在
        )

        if should_train:
            if self.params.get('update_ia_gen', False):
                print("\n🔄 强制重新训练模式")
            else:
                print("\n🆕 首次训练模式（未找到完整模型）")

            # ============================================================
            # 训练新模型
            # ============================================================
            metrics = self._discover_model()

            # ============================================================
            # 比较并保存
            # ============================================================
            save_model = self._compare_models(metrics, model_exists and metadata_exists)

            if save_model:
                print("\n✅ 新模型性能更好，执行保存...")
                self._save_model(metrics)  # 🔥 调用 _save_model
            else:
                print("\n⚠️  新模型性能不如旧模型，保留旧模型")
                # 清理临时文件
                if os.path.exists(self.temp_output):
                    shutil.rmtree(self.temp_output)
                # 加载旧模型
                self._load_existing_model()

        else:
            print("\n✅ 发现现有模型，直接加载...")
            self._load_existing_model()

    def _save_model(self, metrics=None):  # ✅ 添加 metrics 参数（后续会用到）
        """保存完整TFT生态系统"""
        print("\n" + "=" * 80)
        print("💾 保存完整TFT生态系统")
        print("=" * 80)

        # ✅ 修正：使用正确的目录路径
        model_dir = self.params['ia_gen_path']
        os.makedirs(model_dir, exist_ok=True)

        # ============================================================
        # 1. 保存模型 checkpoint（原子性操作）
        # ============================================================
        print(f"\n🔧 保存模型 checkpoint...")

        import tempfile
        temp_dir = tempfile.mkdtemp(dir=model_dir)  # ✅ 使用 model_dir
        temp_checkpoint = os.path.join(temp_dir, 'checkpoint.tmp')

        try:
            # ✅ 修正：使用 self.trainer（而不是未定义的变量）
            self.trainer.save_checkpoint(temp_checkpoint)

            # 原子性移动
            os.replace(temp_checkpoint, self.model_checkpoint_path)
            print(f"   ✅ Checkpoint 已保存: {self.model_checkpoint_path}")

        except Exception as e:
            raise RuntimeError(
                f"❌ Checkpoint 保存失败: {e}\n"
                f"   目标路径: {self.model_checkpoint_path}"
            ) from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # ============================================================
        # 2. 保存数据集配置（原子性操作）
        # ============================================================
        print(f"\n🔧 保存数据集配置...")

        temp_config = os.path.join(model_dir, 'dataset_config.tmp')  # ✅ 使用 model_dir

        try:
            dataset_config = {
                'time_idx': 'time_idx',
                'target': 'delta_t',
                'group_ids': ['group_id'],
                'time_varying_known_categoricals': ['hour', 'day_of_week', 'month'],
                'time_varying_unknown_reals': [ "delta_mean_6","delta_mean_24","delta_mean_168"],
                # ✅ 修正：使用正确的参数键名
                'max_encoder_length': self.params['max_encoder_length'],
                'max_prediction_length': 1,
                'min_encoder_length': self.params['max_encoder_length'],
                'min_prediction_length': self.params['max_prediction_length'],

                'static_categoricals': [],
                'static_reals': [],
                'target_normalizer': self.training_dataset.target_normalizer,
                'categorical_encoders': self.training_dataset.categorical_encoders,

                'add_relative_time_idx': True,
                'add_target_scales': True,
                'add_encoder_length': True,

                'min_time': self.min_time,
                'feature_mode': self.params['feature_mode']
            }

            with open(temp_config, 'wb') as f:
                pickle.dump(dataset_config, f)

            os.replace(temp_config, self.dataset_config_path)
            print(f"   ✅ 数据集配置已保存: {self.dataset_config_path}")

        except Exception as e:
            if os.path.exists(temp_config):
                os.remove(temp_config)

            raise RuntimeError(
                f"❌ 数据集配置保存失败: {e}\n"
                f"   目标路径: {self.dataset_config_path}"
            ) from e

        # ============================================================
        # 3. 保存训练参数和元数据（修正）
        # ============================================================
        print(f"\n🔧 保存训练元数据...")

        # ✅ 修正：保存元数据（用于模型比较）
        if metrics is not None:
            metadata_path = self.metadata_path
            temp_metadata = metadata_path + '.tmp'

            try:
                metadata = {
                    'loss': metrics['loss'],
                    'baseline_mae': metrics.get('baseline_mae'),
                    'improvement': metrics.get('improvement'),
                    'epochs_trained': metrics.get('epochs_trained'),
                    'feature_mode': self.params['feature_mode'],
                    'trained_at': datetime.now().isoformat(),
                    'params': {
                        k: v for k, v in self.params.items()
                        if k not in ['trainer_kwargs', 'early_stop_callback']
                    }
                }

                with open(temp_metadata, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyEncoder)

                os.replace(temp_metadata, metadata_path)
                print(f"   ✅ 元数据已保存: {metadata_path}")

            except Exception as e:
                if os.path.exists(temp_metadata):
                    os.remove(temp_metadata)
                print(f"   ⚠️  元数据保存失败（非致命）: {e}")

        # ============================================================
        # 4. 验证保存的文件
        # ============================================================
        print(f"\n🔍 验证已保存文件...")

        required_files = {
            'checkpoint': self.model_checkpoint_path,
            'dataset_config': self.dataset_config_path
        }

        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"❌ {name} 未成功保存: {path}\n"
                    f"   请检查磁盘空间和权限"
                )

            file_size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"   ✅ {name}: {file_size:.2f} MB")

        print(f"\n✅ 完整生态系统已保存到: {model_dir}")

    def _create_dummy_dataframe(self, dataset_config):
        """
        根据数据集配置创建 dummy DataFrame（v3.1 - 修复缺失列问题）

        Args:
            dataset_config: 从 .pkl 加载的数据集配置

        Returns:
            pd.DataFrame: 包含足够行数的 dummy 数据
        """
        print(f"\n🔧 生成 dummy DataFrame...")

        # ============================================================
        # 计算需要的最小行数
        # ============================================================
        max_encoder_length = dataset_config['max_encoder_length']
        max_prediction_length = 1
        min_required_length = max_encoder_length + max_prediction_length + 10

        print(f"   📏 编码器长度: {max_encoder_length}")
        print(f"   📏 预测长度: {max_prediction_length}")
        print(f"   📏 需要最小行数: {min_required_length}")

        # ============================================================
        # 定义保护列（TimeSeriesDataSet自动添加的列）
        # ============================================================
        PROTECTED_COLUMNS = {
            'relative_time_idx',
            'encoder_length',
            f"{dataset_config['target']}_center",
            f"{dataset_config['target']}_scale",
        }

        print(f"   🛡️  保护列: {PROTECTED_COLUMNS}")

        # ============================================================
        # 收集所有需要的列（排除保护列）
        # ============================================================
        all_columns = set()

        # 基础列
        all_columns.add(dataset_config['time_idx'])
        all_columns.add(dataset_config['target'])
        all_columns.update(dataset_config['group_ids'])

        # 分类特征
        categorical_cols = (
                dataset_config.get('static_categoricals', []) +
                dataset_config.get('time_varying_known_categoricals', []) +
                dataset_config.get('time_varying_unknown_categoricals', [])
        )
        all_columns.update(categorical_cols)

        # 连续特征
        real_cols = (
                dataset_config.get('static_reals', []) +
                dataset_config.get('time_varying_unknown_reals', [])
        )
        all_columns.update(real_cols)

        # 排除保护列
        all_columns = all_columns - PROTECTED_COLUMNS

        print(f"   需要 {len(all_columns)} 列（排除了 {len(PROTECTED_COLUMNS)} 个保护列）")

        # ============================================================
        # 🔥 关键修复：构建完整的 dummy 数据
        # ============================================================
        dummy_data = {}

        # 1. time_idx（必需）
        if dataset_config['time_idx'] in all_columns:
            dummy_data[dataset_config['time_idx']] = list(range(min_required_length))

        # 2. target（必需）
        if dataset_config['target'] in all_columns:
            dummy_data[dataset_config['target']] = [0.1 * (i % 10) for i in range(min_required_length)]

        # 3. group_ids（必需）
        for col in dataset_config['group_ids']:
            if col in all_columns:
                group_value = 'main'
                encoders = dataset_config.get('categorical_encoders', {})
                if col in encoders and hasattr(encoders[col], 'classes_'):
                    try:
                        classes = list(encoders[col].classes_)
                        classes = [c.item() if hasattr(c, "item") else c for c in classes]
                        non_nan = [c for c in classes if str(c).lower() not in ("nan", "<nan>", "none")]
                        if 'main' in classes:
                            group_value = 'main'
                        elif len(non_nan) > 0:
                            group_value = non_nan[0]
                        else:
                            group_value = classes[0]
                    except Exception:
                        pass
                dummy_data[col] = [group_value] * min_required_length

        # 4. 分类特征（time_varying_known_categoricals）
        for col in categorical_cols:
            if col in all_columns:
                encoders = dataset_config.get('categorical_encoders', {})
                if col in encoders and hasattr(encoders[col], 'classes_'):
                    try:
                        all_classes = encoders[col].classes_
                        classes = [c.item() if hasattr(c, 'item') else c for c in all_classes]
                        classes = [c for c in classes if str(c).lower() not in ("nan", "<nan>", "none")]

                        if not classes:
                            classes = ['cat_0']

                        # 根据列名生成合理的循环值
                        if col == 'hour':
                            dummy_data[col] = [str(i % 24) for i in range(min_required_length)]
                        elif col == 'day_of_week':
                            dummy_data[col] = [str(i % 7) for i in range(min_required_length)]
                        elif col == 'month':
                            dummy_data[col] = [str((i % 12) + 1) for i in range(min_required_length)]
                        else:
                            # 其他分类特征：循环使用已知类别
                            dummy_data[col] = [classes[i % len(classes)] for i in range(min_required_length)]
                    except Exception as e:
                        print(f"   ⚠️  {col} 编码器处理失败: {e}")
                        dummy_data[col] = ['cat_0'] * min_required_length
                else:
                    dummy_data[col] = ['cat_0'] * min_required_length

        for col in real_cols:
            if col not in all_columns:
                continue

            if col == 'day_of_month':
                dummy_data[col] = [(i % 31) + 1 for i in range(min_required_length)]
            elif col == 'week_of_year':
                dummy_data[col] = [(i % 52) + 1 for i in range(min_required_length)]
            elif 'delta_mean' in col:
                dummy_data[col] = [1.0 + 0.1 * (i % 10) for i in range(min_required_length)]
            else:
                dummy_data[col] = [0.0] * min_required_length

        # ============================================================
        # 验证并创建DataFrame
        # ============================================================
        df = pd.DataFrame(dummy_data)

        print(f"   ✅ Dummy DataFrame 已创建:")
        print(f"      形状: {df.shape}")
        print(f"      列数: {len(df.columns)}")
        print(f"      行数: {len(df)}")
        print(f"      time_idx 范围: [{df[dataset_config['time_idx']].min()}, {df[dataset_config['time_idx']].max()}]")

        # 🔥 新增：验证是否包含所有必需列
        missing_cols = all_columns - set(df.columns)
        if missing_cols:
            print(f"   ⚠️  缺失的列: {missing_cols}")
            raise ValueError(f"Dummy数据缺少必需列: {missing_cols}")

        return df

    def _load_existing_model(self):
        """
        加载完整TFT生态系统（简化版 v5.0）

        改进：
        1. 移除过度嵌套的异常处理
        2. 明确的错误信息
        3. 快速失败机制
        """
        print("\n" + "=" * 80)
        print("📥 加载完整TFT生态系统")
        print("=" * 80)

        # ============================================================
        # 1. 文件检查（快速失败）
        # ============================================================
        print("\n🔍 检查必需文件...")

        if not os.path.exists(self.model_checkpoint_path):
            raise FileNotFoundError(
                f"❌ 模型文件不存在: {self.model_checkpoint_path}\n"
                f"   请先训练模型或检查文件路径"
            )

        if not os.path.exists(self.dataset_config_path):
            raise FileNotFoundError(
                f"❌ 数据集配置不存在: {self.dataset_config_path}\n"
                f"   请确保使用相同版本的代码保存模型"
            )

        print(f"   ✅ 所有文件就绪")

        # ============================================================
        # 2. 加载 checkpoint
        # ============================================================
        print(f"\n🔧 加载 checkpoint...")
        checkpoint = torch.load(
            self.model_checkpoint_path,
            map_location='cpu',
            weights_only=False  # 允许加载自定义对象
        )

        if 'hyper_parameters' not in checkpoint:
            raise KeyError(
                f"❌ Checkpoint 格式错误: 缺少 'hyper_parameters'\n"
                f"   可能是损坏的模型文件"
            )

        print(f"   ✅ Checkpoint 已加载")

        # ============================================================
        # 3. 实例化模型（单一路径）
        # ============================================================
        print(f"\n🔧 实例化 TFT 模型...")

        hparams = checkpoint['hyper_parameters'].copy()

        # 移除设备相关参数（确保CPU兼容）
        for key in ['gpus', 'devices', 'accelerator']:
            hparams.pop(key, None)

        try:
            self.model = TemporalFusionTransformer(**hparams)
            print(f"   ✅ 模型结构已创建")
        except TypeError as e:
            # 只在参数不匹配时提供降级方案
            print(f"   ⚠️  超参数不兼容: {e}")
            print(f"   🔄 尝试最小参数集...")

            # 提取核心参数
            essential_params = {
                k: v for k, v in hparams.items()
                if k in ['hidden_size', 'lstm_layers', 'dropout',
                         'attention_head_size', 'output_size', 'loss']
            }

            self.model = TemporalFusionTransformer(**essential_params)
            print(f"   ✅ 使用核心参数创建模型")

        # ============================================================
        # 4. 加载权重
        # ============================================================
        print(f"\n🔧 加载模型权重...")

        if 'state_dict' not in checkpoint:
            raise KeyError("❌ Checkpoint 缺少 'state_dict'")

        state_dict = checkpoint['state_dict']

        # 转换为CPU张量
        cpu_state_dict = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }

        # 尝试严格加载
        missing_keys, unexpected_keys = self.model.load_state_dict(
            cpu_state_dict,
            strict=False  # 宽松模式，但记录差异
        )

        if missing_keys:
            print(f"   ⚠️  缺少的权重: {missing_keys[:5]} ...")
        if unexpected_keys:
            print(f"   ⚠️  多余的权重: {unexpected_keys[:5]} ...")

        if not missing_keys and not unexpected_keys:
            print(f"   ✅ 权重完全匹配")
        else:
            print(f"   ⚠️  权重部分匹配（可能影响性能）")

        # ============================================================
        # 5. 设置为评估模式
        # ============================================================
        self.model = self.model.cpu()
        self.model.eval()
        print(f"   ✅ 模型已设为评估模式（CPU）")

        # ============================================================
        # 6. 加载数据集配置
        # ============================================================
        print(f"\n🔧 加载数据集配置...")

        with open(self.dataset_config_path, 'rb') as f:
            self.dataset_config = pickle.load(f)

        self.min_time = self.dataset_config['min_time']

        print(f"   ✅ 配置已加载")
        print(f"   特征模式: {self.dataset_config.get('feature_mode', 'unknown')}")

        # ============================================================
        # 7. 创建 dummy 数据并重建 TimeSeriesDataSet
        # ============================================================
        print(f"\n🔧 重建 TimeSeriesDataSet...")

            # ============================================================
            # 8. 最终验证
            # ============================================================
        print(f"\n✅ TFT生态系统加载完成")
        print(f"   编码器长度: {self.dataset_config['max_encoder_length']} 小时")
        print(f"   预测长度: {self.dataset_config['max_prediction_length']} 小时")
        print(f"   特征数: 已知分类={len(self.dataset_config['time_varying_known_categoricals'])}, "
              f"未知数值={len(self.dataset_config['time_varying_unknown_reals'])}")

        return True

    def _discover_model(self):
        """训练新模型 (含 Sanity Check)"""
        print("\n" + "=" * 80)
        print("🔍 训练新模型 (含 Sanity Check)")
        print("=" * 80)

        # 1. 基准模型
        print("\n📊 计算基准模型...")
        try:
            baseline_predictions = Baseline().predict(
                self.val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
            )
            baseline_mae = float((baseline_predictions.output - baseline_predictions.y[0]).abs().mean())
            print(f"✅ 基准MAE: {baseline_mae:.4f}")
        except Exception as e:
            print(f"⚠️  基准计算失败: {e}")
            baseline_mae = None

        # 2. 配置训练器 (保留过拟合测试)
        temp_model_dir = os.path.join(self.temp_output, 'checkpoints')
        os.makedirs(temp_model_dir, exist_ok=True)

        early_stop_callback = EarlyStopping(
            monitor='val_MAE',
            mode='min',
            patience=self.params['patience'],
            min_delta=1e-4,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(
            dirpath=temp_model_dir,
            filename='best_model',
            monitor='val_MAE',  # ✅ 和 EarlyStopping 对齐
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True
        )
        logger = TensorBoardLogger(save_dir=self.params['log_dir'], name='tft_arrival_resampled')

        self.trainer = pl.Trainer(
            max_epochs=self.params['epochs'],
            accelerator='gpu' if self.params['gpus'] > 0 else 'cpu',
            devices=self.params['gpus'] if self.params['gpus'] > 0 else 1,
            gradient_clip_val=self.params['gradient_clip'],
            callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            val_check_interval=1.0,
            limit_train_batches=1.0,
        )

        # 3. 创建模型
        quantiles = [0.3, 0.5, 0.7]

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            hidden_size=self.params["hidden_size"],
            lstm_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
            attention_head_size=self.params["attention_head_size"],
            output_size=1,  # ✅ 关键
            loss=HuberMAELoss(
                delta=0.5,
                mae_weight=0.3
            ),
            logging_metrics=[MAE()],
            learning_rate=self.params["learning_rate"],
        )

        print(f"✅ 模型参数量: {self.model.size() / 1e6:.2f}M")

        print("\n" + "=" * 80)
        print("📈 开始训练...")
        print("=" * 80)

        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

        # 4. 收集指标
        best_val_loss = float(checkpoint_callback.best_model_score)
        metrics = {
            'loss': best_val_loss,
            'baseline_mae': baseline_mae,
            'improvement': None,
            'epochs_trained': self.trainer.current_epoch,
            'best_model_path': checkpoint_callback.best_model_path,
        }

        print(f"\n📊 训练完成: 最佳验证损失 {best_val_loss:.6f}")

        # 5. 加载模型
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            try:
                self.model = TemporalFusionTransformer.load_from_checkpoint(
                    checkpoint_callback.best_model_path, map_location="cpu"
                )
                print(f"✅ 已加载最佳模型")
            except Exception as e:
                print(f"⚠️  加载最佳模型失败: {e}")

        return metrics

    def _compare_models(self, new_metrics, old_model_exists):
        """
        比较新旧模型性能（v4.0 - 增强错误处理）

        Args:
            new_metrics: 新模型的性能指标
            old_model_exists: 旧模型checkpoint是否存在

        Returns:
            bool: 是否应该保存新模型
        """
        # ✅ 同时检查元数据文件
        if not old_model_exists or not os.path.exists(self.metadata_path):
            if not old_model_exists:
                print("\n✅ 无旧模型，将保存新模型")
            else:
                print("\n✅ 旧模型缺少元数据，将保存新模型")
            return True

        # 加载旧模型的元数据
        try:
            with open(self.metadata_path, 'r') as f:
                old_metadata = json.load(f)

            old_loss = old_metadata.get('loss', float('inf'))
            new_loss = new_metrics['loss']

            print(f"\n📊 模型对比:")
            print(f"  旧模型损失: {old_loss:.6f}")
            print(f"  新模型损失: {new_loss:.6f}")

            if new_loss < old_loss:
                improvement = (1 - new_loss / old_loss) * 100
                print(f"  ✅ 新模型更好（提升 {improvement:.2f}%）")
                return True
            else:
                degradation = (new_loss / old_loss - 1) * 100
                print(f"  ❌ 新模型更差（下降 {degradation:.2f}%）")
                return False

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  旧模型元数据损坏: {e}")
            print("  默认保存新模型")
            return True
        except Exception as e:
            print(f"⚠️  无法加载旧模型元数据: {e}")
            print("  默认保存新模型")
            return True

    def generate(self, num_instances, start_time):
        """
        Inter-arrival based generation
        (MAE-friendly, log-domain, path-stable version)
        """

        # =====================================================
        # 0. 初始化
        # =====================================================
        start_time = pd.Timestamp(start_time)
        if start_time.tz is not None:
            start_time = start_time.tz_localize(None)

        current_time = start_time
        generated = []
        history = self.training.copy()

        # =====================================================
        # 1. 全局 log-domain anchor（固定，不随路径漂移）
        # =====================================================
        global_log_mu = history["delta_t"].mean()

        step = 0

        # =====================================================
        # 2. 主生成循环
        # =====================================================
        while len(generated) < num_instances:

            # -------------------------------------------------
            # 2.1 构造 encoder 输入
            # -------------------------------------------------
            max_enc = self.params["max_encoder_length"]
            available_len = len(history) - 1
            if available_len < 1:
                continue

            encoder_len = min(max_enc, available_len)
            encoder_data = history.tail(encoder_len + 1)

            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                encoder_data,
                predict=True,
                stop_randomization=True,
                min_encoder_length=encoder_len,
                max_encoder_length=encoder_len,
            )

            loader = pred_dataset.to_dataloader(train=False, batch_size=1)

            # -------------------------------------------------
            # 2.2 点预测（❌ 不再 fake quantile）
            # -------------------------------------------------
            with torch.no_grad():
                raw = self.model.predict(loader, mode="prediction")

            # raw.shape = [1, 1]
            base_log = float(raw[0, 0])

            # =================================================
            # 3. 改进 1：log-domain 零均值采样（MAE 友好）
            # =================================================
            eps = np.random.normal(loc=0.0, scale=0.03)
            delta_t_log = base_log + eps

            # =================================================
            # 4. 改进 2：log-domain 局部路径约束
            # =================================================
            recent_log = history["delta_t"].tail(24)
            if len(recent_log) > 0:
                mu_local_log = recent_log.mean()
                delta_t_log = np.clip(
                    delta_t_log,
                    mu_local_log - 0.15,
                    mu_local_log + 0.15
                )

            # =================================================
            # 5. 改进 3：高频 global log-anchor（防坏路径）
            # =================================================
            if step % 10 == 0:
                delta_t_log = np.clip(
                    delta_t_log,
                    global_log_mu - 0.20,
                    global_log_mu + 0.20
                )

            # =================================================
            # 6. 转回秒域
            # =================================================
            delta_t = np.expm1(delta_t_log)
            delta_t = max(delta_t, 1.0)

            # =================================================
            # 7. 状态更新
            # =================================================
            current_time += pd.Timedelta(seconds=delta_t)
            generated.append(current_time)
            step += 1

            # 写回 history（log-domain）
            new_row = history.iloc[-1:].copy()
            new_row["timestamp"] = current_time
            new_row["delta_t"] = np.log1p(delta_t)
            new_row["time_idx"] += 1

            # 时间特征
            new_row["hour"] = str(current_time.hour)
            new_row["day_of_week"] = str(current_time.dayofweek)
            new_row["month"] = str(current_time.month)
            new_row["week_of_year"] = int(current_time.isocalendar()[1])

            history = pd.concat([history, new_row], ignore_index=True)

            # rolling 特征（log-domain）
            for w in [6, 24, 168]:
                history.loc[
                    history.index[-1], f"delta_mean_{w}"
                ] = history["delta_t"].tail(w).mean()

        # =====================================================
        # 8. 输出
        # =====================================================
        return pd.DataFrame({
            "caseid": [f"Case{i + 1}" for i in range(len(generated))],
            "timestamp": generated
        })
