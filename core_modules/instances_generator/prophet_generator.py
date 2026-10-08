# -*- coding: utf-8 -*-

import os
import json
import shutil
import logging
import re
import random
from contextlib import redirect_stdout
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import torch
import optuna

from neuralprophet import NeuralProphet, set_log_level
from optuna.trial import TrialState
from pytorch_lightning.callbacks import Callback

import utils.support as sup
from support_modules.common import LogAttributes as La
from support_modules.common import FileExtensions as Fe


# =========================
# 日志静音
# =========================
logging.getLogger("neuralprophet").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logger = logging.getLogger("neuralprophet")
logger.setLevel(logging.CRITICAL)
set_log_level("ERROR")


class NeuralProphetPruningCallback(Callback):
    """
    用于 NeuralProphet (基于 PyTorch Lightning) 的 Optuna 剪枝回调
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "RMSE_val"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor in metrics:
            current_score = metrics[self.monitor].item()
            epoch = trainer.current_epoch
            self.trial.report(current_score, step=epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned(f"Trial pruned at epoch {epoch}.")




class NeuralProphetGenerator:
    """
    基于 NeuralProphet 的案例到达时间生成器。

    训练阶段：
    1. 每个案例只保留第一个活动的开始时间；
    2. 按小时统计案例到达数，并显式补齐无到达小时为 0；
    3. 使用 rolling-origin × 7/14/28d 多窗口验证，用 Time-CDF + Week-Hour + soft CountBias + Concentration 选择模型。

    生成阶段：
    1. 像原 ProphetGenerator 一样按连续时间块预测未来每小时到达强度；
    2. NeuralProphet 给出连续小时强度，经 rate_scale + calendar blend 稳定后进入生成器；
    3. 按时间向前累计强度并执行 global exact-N，严格生成指定案例数；
    4. 连续空预测时及时退出，并使用近期到达率和星期-小时分布进行日历感知兜底；
    5. 每次 generate 使用不同但可复现的随机种子。
    """

    # ==========================================================
    # 手动实验配置切换
    # ==========================================================
    # 初步对比实验阶段只需要修改 CURRENT_MODE 这一行。六个日志对应：
    #   MP            -> "MP_SPARSE_ADAPTIVE"
    #   P2P           -> "P2P_RIGID"
    #   ACR           -> "ACR_REAL_SHORT"
    #   CVS           -> "CVS_DENSE_SYNTHETIC"
    #   BPIC2012_W    -> "BPI12W_BALANCED"
    #   BPIC2017_W    -> "BPI17W_OPTIMIZED_FINAL"
    # 每个配置内部的 manual_best_params 会跳过模型参数搜索，只执行一组
    # 预设手动参数；rolling-origin × multi-horizon 仍用于验证/健康诊断。
    CURRENT_MODE = "MP_SPARSE_ADAPTIVE"

    # 所有数据集共享同一套“候选－验证－保存”协议。数据集配置和运行参数
    # 均可覆盖这些候选，但测试日志不能参与选择。
    # 生成校准仅保留 rate_scale + calendar_blend 两个参数。
    MODEL_CONFIGS = {
        "BPI12W_BALANCED": {
            "description": (
                "BPI12W: business-hour dominated real arrival process; "
                "strong daily/weekly calendar structure, short-term AR dependence, "
                "high hourly dispersion; compact TPE search."
            ),

            # ==========================================================
            # 1. NeuralProphet 主模型搜索空间
            # ==========================================================
            "param_grid": {

                # 优化参数，对数据结构本身不敏感，因此保持一个稳定的小范围。
                "learning_rate": [
                    0.003,
                    0.005,
                    0.008,
                ],

                # 测试期没有明显长期趋势，
                # 应避免把 weekly fluctuation 错误解释成趋势变化。
                "trend_reg": [
                    0.5,
                    1.0,
                    2.0,
                ],

                # BPI12W 的日/周季节波动非常强，
                # 因此使用较弱的 seasonality regularization。
                "seasonality_reg": [
                    0.1,
                    0.2,
                    0.4,
                ],

                # 测试期主要变化来自 calendar，而非频繁 structural trend break。
                # 因此降低趋势复杂度。
                "n_changepoints": [
                    5,
                    10,
                    15,
                ],

                # lag1、lag24、lag48 都存在明显相关。
                # 24h 已能同时观察最近小时和前一天；
                # 48h 用于判断第二天历史是否还能增加预测信息。
                "n_lags": [
                    24,
                    48,
                ],

                # AR 信号明显，不需要过强 sparsity penalty。
                "ar_reg": [
                    0.05,
                    0.10,
                    0.30,
                ],
            },

            # ==========================================================
            # 2. Seasonality
            # ==========================================================

            # 日内模式非常尖锐：
            # 夜间接近0，10点后突然进入高峰。
            # Fourier order=4 左右可能过度平滑这种边界。
            "daily_seasonality": 10,

            # weekday effect 很强，但 daily seasonality 已负责日内形状，
            # weekly component 不需要过高阶。
            "weekly_seasonality": 4,

            # 数据跨度不足以可靠学习年度规律。
            "yearly_seasonality": False,

            # BPI12W 的 sparsity 主要来自营业时间，而不是随机缺失。
            # 因此关闭目前基于 sparsity/burst penalty 的自动降阶。
            "adaptive_seasonality": False,

            # 搜索严格限制在 24/48h，不自动注入 168h。
            "allow_auto_lag_expansion": False,

            # 避免因为其它启发式规则扩大 trend complexity。
            "allow_auto_changepoint_expansion": False,

            # ==========================================================
            # 3. Week-hour calendar calibration
            # ==========================================================

            # 测试集 weekday-hour 结构极强，
            # 因此允许 calendar prior 比原来的 5-15% 更明显地参与。
            "calendar_blend_alpha_candidates": [
                0.70,
                0.85,
                1.00,
            ],

            # 训练日志较大，weekday-hour prior 不需要过度平滑。
            "calendar_smoothing_candidates": [
                0.10,
                0.30,
                0.50,
            ],

            "calendar_smoothing": 0.30,

            # ==========================================================
            # 4. Rate calibration
            # ==========================================================

            # 允许一定 train -> future rate drift，
            # 但避免 calibration 完全掩盖基础模型的数量预测错误。
            "rate_scale_min": 0.70,
            "rate_scale_max": 1.50,

            # ==========================================================
            # 5. Prediction cap
            # ==========================================================

            # test max=23/h，99%分位约20/h。
            # 1.75 相比原来的2.0更能抑制极端递归预测，
            # 同时留出足够的峰值空间。
            "cap_multiplier": 1.75,

            # ==========================================================
            # 6. Validation objective
            # ==========================================================

            # BPI12W 的 weekday-hour pattern 是非常重要的真实性指标。
            "objective_weights": {
                "time_cdf": 0.35,
                "week_hour": 0.40,
                "count": 0.10,
                "concentration": 0.15,
            },

            # ==========================================================
            # 7. Generation stochasticity
            # ==========================================================

            # BPI12W:
            # mean≈3.06, CV≈1.63, variance/mean≈8.14。
            # 不适合完全平滑的 deterministic intensity。
            #
            # triangular_calibrated 在预测区间内扰动，
            # 比无界 Poisson perturbation 更稳定；
            # 最后仍由 global exact-N 保证总案例数。
            "arrival_count_sampling": "triangular_calibrated",

            # ==========================================================
            # 8. Training
            # ==========================================================

            "epochs": 180,
            "batch_size": 64,

            "search_method": "tpe",

            # 紧凑 TPE：
            # 比原来的30次明显快，但又不至于只有几个随机trial。
            "n_trials": 18,
        },

        "BPI17W_OPTIMIZED_FINAL": {
            "description": (
                "BPI17W compact search: strong daily/weekly calendar structure, "
                "clear AR dependence, bounded local TPE search."
            ),

            "param_grid": {
                "learning_rate": [0.003, 0.005, 0.008],

                "trend_reg": [0.3, 0.5, 1.0],

                "seasonality_reg": [0.1, 0.3, 0.5],

                "n_changepoints": [20, 30],

                # BPI17W具有明确的日/周相关性，不再搜索无AR模型
                "n_lags": [24, 168],

                # 避免0正则导致168维AR过拟合，也去掉过强的0.5/1.0
                "ar_reg": [0.05, 0.10, 0.30],
            },

            "daily_seasonality": 10,
            "weekly_seasonality": 10,
            "yearly_seasonality": False,

            # BPI17W的稀疏主要是工作日历造成的，而不是随机稀疏
            "adaptive_seasonality": False,

            "allow_auto_lag_expansion": False,
            "allow_auto_changepoint_expansion": False,

            # alpha=1是纯NeuralProphet；
            # BPI17W周历结构很强，因此允许15%左右calendar prior参与
            "calendar_blend_alpha_candidates": [
                0.85,
                0.925,
                1.00,
            ],

            # 大样本 + 明确Sunday/weekday结构，不宜过度平滑
            "calendar_smoothing_candidates": [
                0.05,
                0.15,
                0.30,
            ],
            "calendar_smoothing": 0.15,

            # 不让rate calibration过度掩盖基础模型错误
            "rate_scale_min": 0.75,
            "rate_scale_max": 1.35,

            # 防止168h递归产生极端峰值
            "cap_multiplier": 1.75,

            "objective_weights": {
                "time_cdf": 0.35,
                "week_hour": 0.40,
                "count": 0.10,
                "concentration": 0.15,
            },

            # 比Poisson更适合保持已学到的calendar shape
            "arrival_count_sampling": "triangular_calibrated",

            "epochs": 200,
            "batch_size": 128,

            "search_method": "tpe",
            "n_trials": 15,

            # !!! 删除 manual_best_params !!!
        },
        "CVS_DENSE_SYNTHETIC": {
            "description": (
                "CVS dense synthetic arrivals: stable business-hour arrivals, "
                "strong daily/weekly calendar structure, low trend complexity, no AR."
            ),

            # =========================
            # 1. NeuralProphet 主模型搜索
            # =========================
            "param_grid": {
                "learning_rate": [0.003, 0.005, 0.008],
                "trend_reg": [1.0, 3.0, 5.0],
                "seasonality_reg": [0.3, 0.5, 0.8],
                "n_changepoints": [1, 3, 5],

                # CVS 的相关性主要来自营业日历，不需要 AR
                "n_lags": [0],
                "ar_reg": [0.0],
            },

            # =========================
            # 2. Seasonality
            # =========================
            "daily_seasonality": 10,
            "weekly_seasonality": 6,
            "yearly_seasonality": False,

            # CVS 结构明确，不让程序自动改变 Fourier 阶数
            "adaptive_seasonality": False,

            # 不自动加入 24h / 168h AR
            "allow_auto_lag_expansion": False,

            # 不自动增加 changepoints
            "allow_auto_changepoint_expansion": False,

            # =========================
            # 3. Calendar calibration
            # =========================
            "calendar_blend_alpha_candidates": [
                0.40,
                0.60,
                0.80,
                1.00,
            ],

            "calendar_smoothing_candidates": [
                0.00,
                0.05,
                0.10,
            ],

            "calendar_smoothing": 0.05,

            # =========================
            # 4. Prediction cap
            # =========================
            "cap_multiplier": 1.5,

            # =========================
            # 5. Training
            # =========================
            "epochs": 200,
            "batch_size": 64,

            # TPE，不做完整 grid
            "search_method": "tpe",
            "n_trials": 18,

            # CVS 建议明确采用 exact-N
            "arrival_count_sampling": "exact_n",
        },

        "MP_SPARSE_ADAPTIVE": {
            "description": (
                "MP sparse arrivals: low-complexity trend, no AR, stronger "
                "calendar stabilization, global exact-N generation."
            ),
            "param_grid": {
                "learning_rate": [0.005],
                "trend_reg": [1.0, 2.0, 4.0],
                "seasonality_reg": [1.0, 2.0, 4.0],
                "n_changepoints": [1, 2, 3],
                "n_lags": [0],
                "ar_reg": [0.0],
            },
            "daily_seasonality": 4,
            "weekly_seasonality": 2,
            "yearly_seasonality": False,
            "adaptive_seasonality": False,
            "allow_auto_lag_expansion": False,
            "allow_auto_changepoint_expansion": False,
            "search_method": "grid",
            "calendar_blend_alpha_candidates": [0.60, 0.75, 0.90, 1.00],
            "calendar_smoothing_candidates": [0.50, 0.65, 0.80],
            "calendar_smoothing": 0.65,
            "cap_multiplier": 3.0,
            # 推荐手动固定配置：小样本稀疏日志，强正则、低趋势复杂度、无AR。
            "manual_best_params": {
                "learning_rate": 0.005,
                "trend_reg": 4.0,
                "seasonality_reg": 2.0,
                "n_changepoints": 2,
                "n_lags": 0,
                "ar_reg": 0.0,
                "calendar_blend_alpha": 0.75,
                "calendar_smoothing": 0.80,
                "cap_multiplier": 3.0,
            },
            "arrival_count_sampling": "exact_n",
            "epochs": 200,
            "batch_size": 32,
            "n_trials": 6,
        },

        "ACR_REAL_SHORT": {
            "config_version": "acr_real_short_v4_rolling",
            "description": (
                "ACR sparse short real log: low-complexity trend, no AR, "
                "calendar-prior stabilization."
            ),
            "param_grid": {
                "learning_rate": [0.003, 0.005, 0.008],
                "trend_reg": [1.0, 2.0, 5.0],
                "seasonality_reg": [3.0],
                "n_changepoints": [1, 2, 3],
                "n_lags": [0],
                "ar_reg": [0.0],
            },
            "daily_seasonality": False,
            "weekly_seasonality": False,
            "yearly_seasonality": False,
            "adaptive_seasonality": False,
            "allow_auto_lag_expansion": False,
            "allow_auto_changepoint_expansion": False,
            "calendar_blend_alpha_candidates": [0.60, 0.75, 0.90, 1.00],
            "calendar_smoothing_candidates": [0.65, 0.80, 0.90],
            "calendar_smoothing": 0.80,
            "rate_scale_min": 0.70,
            "rate_scale_max": 1.50,
            "cap_multiplier": 3.0,
            # 推荐手动固定配置：短而稀疏，关闭显式季节性/AR，依靠日历先验稳定。
            "manual_best_params": {
                "learning_rate": 0.005,
                "trend_reg": 5.0,
                "seasonality_reg": 3.0,
                "n_changepoints": 2,
                "n_lags": 0,
                "ar_reg": 0.0,
                "calendar_blend_alpha": 0.75,
                "calendar_smoothing": 0.80,
                "cap_multiplier": 3.0,
            },
            "arrival_count_sampling": "exact_n",
            "epochs": 200,
            "batch_size": 32,
            "search_method": "grid",
            "n_trials": 27,
        },

        "P2P_RIGID": {
            "description": (
                "P2P sparse arrivals with downward trend; strongly regularized "
                "low-flexibility trend and no AR."
            ),
            "param_grid": {
                "learning_rate": [0.003, 0.005, 0.008],
                "trend_reg": [2.0, 5.0, 10.0],
                "seasonality_reg": [5.0],
                "n_changepoints": [1, 2, 3],
                "n_lags": [0],
                "ar_reg": [0.0],
            },
            "daily_seasonality": False,
            "weekly_seasonality": 3,
            "yearly_seasonality": False,
            "adaptive_seasonality": False,
            "allow_auto_lag_expansion": False,
            "allow_auto_changepoint_expansion": False,
            "calendar_blend_alpha_candidates": [0.75, 0.90, 1.00],
            "calendar_smoothing_candidates": [0.50, 0.70],
            "calendar_smoothing": 0.70,
            "cap_multiplier": 2.0,
            # 推荐手动固定配置：下降趋势优先，强趋势正则、低变点复杂度、无AR。
            "manual_best_params": {
                "learning_rate": 0.005,
                "trend_reg": 2.0,
                "seasonality_reg": 5.0,
                "n_changepoints": 2,
                "n_lags": 0,
                "ar_reg": 0.0,
                "calendar_blend_alpha": 0.8,
                "calendar_smoothing": 0.70,
                "cap_multiplier": 2.0,
            },
            "arrival_count_sampling": "exact_n",
            "epochs": 250,
            "batch_size": 32,
            "n_trials": 40,
        },
    }

    def __init__(self, log, valdn, parms):
        self.times = None
        self.temp_output = os.path.join("output_files", sup.folder_id())
        os.makedirs(self.temp_output, exist_ok=True)

        self.log = pd.DataFrame(log.data)
        self.valdn = valdn
        self.parms = parms
        self.model_metadata = dict()
        self.is_safe = True
        self.max_cap = None
        self._generation_call = 0

        self._load_model()

    # ==========================================================
    # 模型加载 / 训练
    # ==========================================================
    def _load_model(self) -> None:
        base_name = self.parms["file"].split(".")[0]
        model_path = os.path.join(self.parms["ia_gen_path"], f"{base_name}_nprf.np")

        self.model_path = model_path
        model_exist = os.path.exists(model_path)
        self.parms["model_path"] = model_path

        if (not model_exist) or self.parms["update_ia_gen"]:
            acc = self._discover_model()
            save, metadata_file = self._compare_models(acc, model_exist)
            if save:
                self._save_model(metadata_file, acc)
            else:
                shutil.rmtree(self.temp_output, ignore_errors=True)

    def _compare_models(self, acc, model_exist):
        base_name = self.parms["file"].split(".")[0]
        metadata_file = os.path.join(self.parms["ia_gen_path"], f"{base_name}_nprf_meta{Fe.JSON}")
        save = True

        if model_exist and os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            # 季节性模式或阶数发生变化时，新旧 RMSE 不再对应同一模型结构。
            # 例如旧 MP 模型实际为 daily=False，而修复后应为 daily=4；
            # 此时必须保存新模型，不能让旧模型因 RMSE 略低而继续被保留。
            signature_keys = [
                "arrival_model_config",
                "seasonality_mode",
                "daily_seasonality",
                "weekly_seasonality",
                "yearly_seasonality",
                "arrival_objective_version",
                "simple_generation_calibration_version",
                "cap_multiplier",
                "rolling_origin_validation",
            ]
            configuration_changed = any(
                data.get(key) != self.model_metadata.get(key)
                for key in signature_keys
            )
            old_calibration = data.get("simple_generation_calibration", {})
            new_calibration = self.model_metadata.get(
                "simple_generation_calibration", {}
            )
            calibration_signature_keys = [
                "rate_scale",
                "calendar_blend_alpha",
                "calendar_smoothing",
            ]
            calibration_changed = any(
                old_calibration.get(key) != new_calibration.get(key)
                for key in calibration_signature_keys
            )
            if configuration_changed or calibration_changed:
                print(
                    "[模型更新] 检测到到达模型或简化生成校准协议发生变化，"
                    "跳过旧 loss 比较并保存新模型。"
                )
                return True, metadata_file

            old_loss = data.get("loss", float("inf"))
            if old_loss < acc["loss"]:
                save = False
        return save, metadata_file

    def _select_training_config(self, profile, data_hours):
        """
        选择训练配置。

        默认 AUTO 仅依据当前训练到达序列的稀疏性、周期性和长度构造搜索空间，
        不再因为类常量写死为 P2P 而误用其他数据集的参数。用户仍可通过
        arrival_model_config 显式选择旧配置以复现实验。
        """
        requested = str(
            self.parms.get("arrival_model_config", self.CURRENT_MODE)
        ).strip().upper()

        if requested not in {"", "AUTO", "NONE"}:
            if requested not in self.MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown arrival_model_config={requested}. "
                    f"Available: AUTO, {sorted(self.MODEL_CONFIGS)}"
                )
            config = {
                k: ({pk: list(pv) if isinstance(pv, list) else pv
                     for pk, pv in v.items()} if k == "param_grid" else v)
                for k, v in self.MODEL_CONFIGS[requested].items()
            }
            print(f"[到达模型配置] 使用显式配置: {requested}")
            return config, requested

        mean_rate = float(profile.get("mean_rate", 0.0))
        nonzero_ratio = float(profile.get("nonzero_ratio", 0.0))
        cv_rate = float(profile.get("cv_rate", 0.0))
        daily_strength = float(profile.get("daily_strength", 0.0))
        weekly_strength = float(profile.get("weekly_strength", 0.0))

        sparse = mean_rate < 0.40 and nonzero_ratio < 0.35
        dense = mean_rate >= 1.0 or nonzero_ratio >= 0.55

        # 只有历史足够长且确实检测到相关性时才允许 AR。
        lag_candidates = [0]
        if data_hours >= 24 * 14 and daily_strength >= 0.12 and not sparse:
            lag_candidates.append(24)
        daily_order = 10 if daily_strength >= 0.08 else False
        weekly_order = 6 if weekly_strength >= 0.08 and data_hours >= 24 * 14 else False

        if sparse:
            profile_name = "AUTO_SPARSE"
            learning_rates = [0.003, 0.008, 0.015]
            trend_regs = [0.5, 1.0, 2.0]
            seasonality_regs = [1.0, 3.0, 5.0]
            changepoints = [3, 5, 8]
            batch_size = 16
            epochs = 250
        elif dense and cv_rate < 1.5:
            profile_name = "AUTO_DENSE"
            learning_rates = [0.005, 0.01, 0.02]
            trend_regs = [0.1, 0.5, 1.0]
            seasonality_regs = [0.1, 0.5, 1.0]
            changepoints = [10, 20, 30]
            batch_size = 128
            epochs = 200
        else:
            profile_name = "AUTO_GENERAL"
            learning_rates = [0.003, 0.008, 0.015]
            trend_regs = [0.3, 1.0, 2.0]
            seasonality_regs = [0.3, 1.0, 3.0]
            changepoints = [5, 10, 20]
            batch_size = 64
            epochs = 220

        config = {
            "description": "Data-driven NeuralProphet configuration",
            "param_grid": {
                "learning_rate": learning_rates,
                "trend_reg": trend_regs,
                "seasonality_reg": seasonality_regs,
                "n_changepoints": changepoints,
                "n_lags": sorted(set(lag_candidates)),
                "ar_reg": [0.0, 0.3, 1.0] if any(lag_candidates) else [0.0],
            },
            "daily_seasonality": daily_order,
            "weekly_seasonality": weekly_order,
            "yearly_seasonality": False,
            "epochs": int(self.parms.get("arrival_epochs", epochs)),
            "batch_size": int(self.parms.get("arrival_batch_size", batch_size)),
            "n_trials": int(self.parms.get("arrival_n_trials", 30)),
        }
        print(
            f"[到达模型配置] {profile_name}: "
            f"lags={config['param_grid']['n_lags']}, "
            f"daily={daily_order}, weekly={weekly_order}"
        )
        return config, profile_name

    # ==========================================================
    # 联合验证目标：Time-CDF + Week-Hour + soft CountBias + Concentration
    # ==========================================================
    def _get_arrival_objective_weights(self, config=None):
        """
        返回到达模型联合目标权重。

        固定案例数（exact-N）场景的统一默认权重：
            Time-CDF distance          0.40
            Week-Hour distribution     0.35
            Total count bias           0.10
            Concentration error         0.15

        CountBias 仍保留为软健康约束，用于识别原始 NeuralProphet 强度严重
        低估/高估的模型；评价权重不交给参数搜索自动修改。
        """
        configured = {}
        if isinstance(config, dict):
            cfg_weights = config.get("objective_weights", {})
            if isinstance(cfg_weights, dict):
                configured.update(cfg_weights)

        runtime_weights = self.parms.get("arrival_objective_weights", {})
        if isinstance(runtime_weights, dict):
            configured.update(runtime_weights)

        time_cdf_weight = float(
            self.parms.get(
                "arrival_objective_time_cdf_weight",
                configured.get("time_cdf", 0.40),
            )
        )
        week_hour_weight = float(
            self.parms.get(
                "arrival_objective_week_hour_weight",
                configured.get("week_hour", 0.35),
            )
        )
        count_weight = float(
            self.parms.get(
                "arrival_objective_count_weight",
                configured.get("count", 0.10),
            )
        )
        concentration_weight = float(
            self.parms.get(
                "arrival_objective_concentration_weight",
                configured.get("concentration", 0.15),
            )
        )

        weights = np.asarray(
            [time_cdf_weight, week_hour_weight, count_weight, concentration_weight],
            dtype=float,
        )
        if (not np.all(np.isfinite(weights))) or np.any(weights < 0):
            raise ValueError(
                "Arrival objective weights must be finite and non-negative."
            )

        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("At least one arrival objective weight must be > 0.")

        weights = weights / total
        return {
            "time_cdf": float(weights[0]),
            "week_hour": float(weights[1]),
            "count": float(weights[2]),
            "concentration": float(weights[3]),
        }

    def _arrival_rate_diagnostics(self, score):
        """数量相关指标只做诊断，不参与 trial 硬淘汰。"""
        actual_total = float(score.get("actual_total", 0.0))
        predicted_total = float(score.get("predicted_total", 0.0))
        eps = 1e-12
        if actual_total <= eps:
            count_ratio = 1.0 if predicted_total <= eps else float("inf")
        else:
            count_ratio = predicted_total / actual_total

        near_zero_threshold = float(
            self.parms.get("arrival_diagnostic_near_zero_ratio", 0.20)
        )
        near_zero = bool(
            actual_total > eps
            and np.isfinite(count_ratio)
            and count_ratio < near_zero_threshold
        )
        return {
            "count_ratio": float(count_ratio),
            "near_zero": near_zero,
            "near_zero_threshold": near_zero_threshold,
        }

    @staticmethod
    def _prepare_arrival_validation_frames(actual_df, pred_df):
        """
        对齐验证集真实小时计数和模型预测强度。

        actual_df 必须包含：ds, y
        pred_df   必须包含：ds, yhat1
        返回每个验证小时一行：ds, y, yhat1。
        """
        if actual_df is None or len(actual_df) == 0:
            raise ValueError("Arrival validation actual data is empty.")
        if pred_df is None or len(pred_df) == 0:
            raise ValueError("Arrival validation prediction is empty.")

        actual = actual_df[["ds", "y"]].copy()
        pred = pred_df[["ds", "yhat1"]].copy()

        actual["ds"] = pd.to_datetime(actual["ds"])
        pred["ds"] = pd.to_datetime(pred["ds"])

        if actual["ds"].dt.tz is not None:
            actual["ds"] = actual["ds"].dt.tz_localize(None)
        if pred["ds"].dt.tz is not None:
            pred["ds"] = pred["ds"].dt.tz_localize(None)

        actual["y"] = (
            pd.to_numeric(actual["y"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        pred["yhat1"] = (
            pd.to_numeric(pred["yhat1"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

        # 以真实验证窗口为基准；缺失预测按 0 处理。
        merged = actual.merge(
            pred.drop_duplicates(subset=["ds"], keep="last"),
            on="ds",
            how="left",
        )
        merged["yhat1"] = merged["yhat1"].fillna(0.0).clip(lower=0.0)
        merged = merged.sort_values("ds").reset_index(drop=True)

        if merged.empty:
            raise ValueError("No aligned arrival validation hours were found.")

        return merged

    @staticmethod
    def _time_cdf_distance(aligned_df):
        """
        比较真实与预测到达质量沿时间轴的累计分布。

        对稀疏日志，单个案例错开 1-2 小时不应像逐小时 WAPE 那样被双重惩罚；
        CDF 距离更关注整批案例在时间轴上的前后位置与节奏。
        取值约在 [0, 1]，越小越好。
        """
        actual = pd.to_numeric(
            aligned_df["y"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        predicted = pd.to_numeric(
            aligned_df["yhat1"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)

        actual_total = float(actual.sum())
        predicted_total = float(predicted.sum())
        eps = 1e-12
        if actual_total <= eps and predicted_total <= eps:
            return 0.0
        if actual_total <= eps or predicted_total <= eps:
            return 1.0

        actual_cdf = np.cumsum(actual / actual_total)
        predicted_cdf = np.cumsum(predicted / predicted_total)
        return float(np.mean(np.abs(actual_cdf - predicted_cdf)))

    @staticmethod
    def _week_hour_distribution_distance(aligned_df):
        """
        计算真实/预测到达量在 7×24=168 个星期-小时槽位上的分布距离。

        使用 Total Variation Distance (TVD)：
            TVD = 0.5 * sum(|p_k - q_k|)

        其中 p/q 分别是真实与预测在 168 个槽位上的归一化到达质量。
        取值范围 [0, 1]，越小越好：
            0 = 星期-小时分布完全一致
            1 = 两个分布完全不重叠

        这里使用预测强度 yhat1，而不是随机采样后的整数案例数，
        避免把采样噪声混入“模型选择”目标。
        """
        data = aligned_df[["ds", "y", "yhat1"]].copy()
        data["weekday"] = data["ds"].dt.dayofweek
        data["hour"] = data["ds"].dt.hour

        full_index = pd.MultiIndex.from_product(
            [range(7), range(24)],
            names=["weekday", "hour"],
        )

        actual_mass = (
            data.groupby(["weekday", "hour"])["y"]
            .sum()
            .reindex(full_index, fill_value=0.0)
            .astype(float)
        )
        pred_mass = (
            data.groupby(["weekday", "hour"])["yhat1"]
            .sum()
            .reindex(full_index, fill_value=0.0)
            .astype(float)
        )

        actual_total = float(actual_mass.sum())
        pred_total = float(pred_mass.sum())
        eps = 1e-12

        if actual_total <= eps and pred_total <= eps:
            return 0.0
        if actual_total <= eps or pred_total <= eps:
            return 1.0

        p = actual_mass.to_numpy(dtype=float) / actual_total
        q = pred_mass.to_numpy(dtype=float) / pred_total
        return float(0.5 * np.abs(p - q).sum())

    def _arrival_concentration_statistics(self, aligned_df):
        """
        计算真实/预测小时强度的集中程度及其误差。

        对长度为 H 的验证窗口，取强度最高的前 ceil(H * top_fraction) 个小时，
        计算这些小时占总到达质量的比例。真实与预测分别独立排序，因此该指标
        只刻画“是否过平/过尖”，具体发生在哪个 weekday-hour 仍由 WeekHour
        distance 约束。
        """
        top_fraction = float(
            self.parms.get("arrival_concentration_top_fraction", 0.10)
        )
        top_fraction = float(np.clip(top_fraction, 0.01, 0.50))

        actual = pd.to_numeric(
            aligned_df["y"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        predicted = pd.to_numeric(
            aligned_df["yhat1"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)

        def top_mass(values):
            values = np.asarray(values, dtype=float)
            total = float(values.sum())
            if total <= 1e-12 or len(values) == 0:
                return 0.0
            k = max(1, int(np.ceil(len(values) * top_fraction)))
            if k >= len(values):
                return 1.0
            # partition 比完整排序更省内存，适用于 BPI17W 等长序列。
            top_values = np.partition(values, len(values) - k)[-k:]
            return float(top_values.sum() / total)

        actual_concentration = top_mass(actual)
        predicted_concentration = top_mass(predicted)
        error = abs(actual_concentration - predicted_concentration)
        return {
            "error": float(error),
            "actual": float(actual_concentration),
            "predicted": float(predicted_concentration),
            "top_fraction": float(top_fraction),
        }

    def _arrival_joint_validation_score(
            self,
            actual_df,
            pred_df,
            weights=None):
        """
        联合验证目标（越小越好）：

            J = w1 * TimeCDF
              + w2 * WeekHour_TVD
              + w3 * Total_Count_Bias
              + w4 * ConcentrationError

        Hourly WAPE 仍计算并记录，但仅作为诊断，不参与模型排名。
        """
        if weights is None:
            weights = self._get_arrival_objective_weights()

        aligned = self._prepare_arrival_validation_frames(
            actual_df=actual_df,
            pred_df=pred_df,
        )

        actual = aligned["y"].to_numpy(dtype=float)
        predicted = aligned["yhat1"].to_numpy(dtype=float)

        actual_total = float(actual.sum())
        predicted_total = float(predicted.sum())
        denominator = max(actual_total, 1.0)

        hourly_wape = float(
            np.abs(actual - predicted).sum() / denominator
        )
        time_cdf_distance = self._time_cdf_distance(aligned)
        count_bias = float(
            abs(predicted_total - actual_total) / denominator
        )
        week_hour_distance = self._week_hour_distribution_distance(aligned)
        concentration = self._arrival_concentration_statistics(aligned)

        joint_score = float(
            weights["time_cdf"] * time_cdf_distance
            + weights["week_hour"] * week_hour_distance
            + weights["count"] * count_bias
            + weights["concentration"] * concentration["error"]
        )

        signed_count_bias = float(
            (predicted_total - actual_total) / denominator
        )

        return {
            "joint_score": joint_score,
            "time_cdf_distance": float(time_cdf_distance),
            "hourly_wape": hourly_wape,
            "week_hour_distance": float(week_hour_distance),
            "count_bias": count_bias,
            "concentration_error": float(concentration["error"]),
            "actual_concentration": float(concentration["actual"]),
            "predicted_concentration": float(concentration["predicted"]),
            "concentration_top_fraction": float(concentration["top_fraction"]),
            "signed_count_bias": signed_count_bias,
            "actual_total": actual_total,
            "predicted_total": predicted_total,
            "validation_hours": int(len(aligned)),
        }

    def _predict_validation_arrivals(
            self,
            model,
            train_df,
            validation_df,
            max_cap=None):
        """
        在内部时间验证窗口上生成连续小时到达强度。

        n_lags=0：直接预测整个验证区间；
        n_lags>0：逐小时递归预测，每一步都把上一小时预测值反馈为下一步历史，
        且上下文只来自当前 origin 之前的数据，避免未来标签泄漏。
        """
        if validation_df is None or validation_df.empty:
            return pd.DataFrame(columns=["ds", "yhat1"])

        validation_start = pd.to_datetime(validation_df["ds"].min())
        validation_end = pd.to_datetime(validation_df["ds"].max())
        n_lags = int(getattr(model, "n_lags", 0))
        effective_cap = self.max_cap if max_cap is None else float(max_cap)

        if n_lags <= 0:
            return self._predict_density_no_lags(
                m=model,
                forecast_start=validation_start,
                horizon_end=validation_end,
                freq="1h",
                max_cap=effective_cap,
            )

        train_cases_proxy = max(
            1,
            int(round(pd.to_numeric(train_df["y"], errors="coerce")
                      .fillna(0.0).clip(lower=0.0).sum()))
        )
        train_rate = float(
            pd.to_numeric(train_df["y"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).mean()
        )
        validation_cases_proxy = max(
            1,
            int(round(max(train_rate, 0.0) * max(len(validation_df), 1)))
        )
        profile = self._build_generation_profile(
            df_history=train_df,
            num_instances=validation_cases_proxy,
            total_cases=train_cases_proxy,
        )

        pred_df, _ = self._predict_lagged_block(
            m=model,
            context_df=train_df[["ds", "y"]].copy(),
            block_start=validation_start,
            block_end=validation_end,
            profile=profile,
            max_cap=effective_cap,
        )
        return pred_df

    def _get_multi_horizon_validation_plan(self, n_hours, valid_p):
        """
        构造统一的多时间尺度内部验证方案。

        默认目标窗口为 7/14/28 天，对应权重 0.2/0.3/0.5。
        若日志历史不足以容纳某个窗口，则自动删除该窗口并重新归一化权重；
        若连 7 天都无法容纳，则退化为一个按 valid_p 计算的短窗口。

        这里只读取训练分区长度和运行参数，不读取任何 validation/test 标签。
        """
        n_hours = int(max(0, n_hours))
        valid_p = float(np.clip(valid_p, 0.01, 0.50))

        raw_days = self.parms.get(
            "arrival_validation_horizon_days",
            [7, 14, 28],
        )
        raw_weights = self.parms.get(
            "arrival_validation_horizon_weights",
            [0.20, 0.30, 0.50],
        )

        if isinstance(raw_days, str):
            raw_days = [
                item.strip() for item in raw_days.split(",") if item.strip()
            ]
        if not isinstance(raw_days, (list, tuple, set, np.ndarray)):
            raw_days = [raw_days]

        days = []
        for item in raw_days:
            try:
                value = float(item)
            except Exception:
                continue
            if np.isfinite(value) and value > 0:
                days.append(value)

        if not days:
            days = [7.0, 14.0, 28.0]

        if isinstance(raw_weights, str):
            raw_weights = [
                item.strip() for item in raw_weights.split(",") if item.strip()
            ]
        if not isinstance(raw_weights, (list, tuple, np.ndarray)):
            raw_weights = [raw_weights]

        parsed_weights = []
        for item in raw_weights:
            try:
                value = float(item)
            except Exception:
                value = float("nan")
            parsed_weights.append(value)

        if (
            len(parsed_weights) != len(days)
            or not np.all(np.isfinite(parsed_weights))
            or np.any(np.asarray(parsed_weights, dtype=float) < 0)
            or float(np.sum(parsed_weights)) <= 0.0
        ):
            parsed_weights = [1.0] * len(days)

        pairs = sorted(
            {
                (int(round(day * 24.0)), float(weight))
                for day, weight in zip(days, parsed_weights)
                if int(round(day * 24.0)) > 0
            },
            key=lambda x: x[0],
        )

        # 至少保留两周训练历史；对很短日志则允许自动缩短。
        configured_min_train = int(
            self.parms.get(
                "arrival_validation_min_train_hours",
                24 * 14,
            )
        )
        configured_min_train = max(24, configured_min_train)
        min_train_hours = min(
            configured_min_train,
            max(24, n_hours // 2),
        )

        max_available_horizon = max(0, n_hours - min_train_hours)
        selected = [
            (hours, weight)
            for hours, weight in pairs
            if hours <= max_available_horizon
        ]

        fallback = False
        if not selected:
            fallback = True
            # 对短日志保留至少约一半历史用于拟合，同时至少验证 1 小时。
            fallback_train = max(24, min(n_hours // 2, configured_min_train))
            max_fallback_horizon = max(1, n_hours - fallback_train)
            fallback_hours = max(
                1,
                int(np.ceil(n_hours * valid_p)),
            )
            fallback_hours = min(
                fallback_hours,
                max_fallback_horizon,
            )
            selected = [(int(fallback_hours), 1.0)]

        total_weight = float(sum(weight for _, weight in selected))
        if total_weight <= 0:
            normalized = [1.0 / len(selected)] * len(selected)
        else:
            normalized = [weight / total_weight for _, weight in selected]

        horizons = []
        for (hours, _), weight in zip(selected, normalized):
            if hours % 24 == 0:
                label = f"{hours // 24}d"
            else:
                label = f"{hours}h"
            horizons.append({
                "label": label,
                "hours": int(hours),
                "weight": float(weight),
            })

        return {
            "version": "multi_horizon_v2_long_weighted",
            "fallback_single_horizon": bool(fallback),
            "horizons": horizons,
            "max_horizon_hours": int(max(x["hours"] for x in horizons)),
            "min_train_hours": int(min_train_hours),
            "source": "training_partition_length_only",
        }

    def _get_rolling_origin_validation_plan(self, n_hours, validation_plan):
        """
        在 training partition 内构造 rolling-origin 验证起点。

        默认使用 3 个 origin，相邻 origin 间隔 14 天；每个 origin 都向未来
        评估同一套 7/14/28d multi-horizon。历史不足时自动退化为 2 个或 1 个
        origin。这里只依据训练序列长度，不读取任何验证/test 标签。
        """
        n_hours = int(max(0, n_hours))
        max_horizon = int(validation_plan["max_horizon_hours"])
        min_train = int(max(24, validation_plan.get("min_train_hours", 24)))

        n_origins = max(
            1,
            int(self.parms.get("arrival_validation_n_origins", 3)),
        )
        step_hours = max(
            24,
            int(round(float(
                self.parms.get("arrival_validation_origin_step_days", 14)
            ) * 24.0)),
        )

        latest_origin = n_hours - max_horizon
        if latest_origin < min_train:
            # multi-horizon planner 理论上已保证该条件；这里保留防御式回退。
            latest_origin = max(24, latest_origin)

        origins_desc = []
        cursor = int(latest_origin)
        while cursor >= min_train and len(origins_desc) < n_origins:
            origins_desc.append(cursor)
            cursor -= step_hours

        if not origins_desc:
            origins_desc = [int(latest_origin)]

        origins = sorted(set(int(x) for x in origins_desc if x >= 24))
        if not origins:
            origins = [max(24, int(latest_origin))]

        return {
            "version": "rolling_origin_v1_x_multi_horizon",
            "n_origins_requested": int(n_origins),
            "n_origins_used": int(len(origins)),
            "origin_step_hours": int(step_hours),
            "origin_indices": origins,
            "earliest_origin": int(min(origins)),
            "latest_origin": int(max(origins)),
            "max_horizon_hours": int(max_horizon),
            "source": "training_partition_length_only",
        }

    def _score_multi_horizon_validation(
            self,
            actual_full,
            pred_full,
            validation_plan,
            weights):
        """
        对同一个预测起点同时计算多个 horizon 的验证分数。

        例如 7d/14d/28d 都从同一个训练截止点向未来预测，
        这样可以直接检查短期表现和长期外推稳定性是否一致。
        """
        horizons = list(validation_plan.get("horizons", []))
        if not horizons:
            raise ValueError("Multi-horizon validation plan is empty.")

        actual_full = (
            actual_full[["ds", "y"]]
            .copy()
            .sort_values("ds")
            .reset_index(drop=True)
        )

        horizon_scores = []
        for item in horizons:
            hours = int(item["hours"])
            weight = float(item["weight"])
            actual_slice = actual_full.iloc[:hours].copy()

            score = self._arrival_joint_validation_score(
                actual_df=actual_slice,
                pred_df=pred_full,
                weights=weights,
            )
            score.update({
                "label": str(item["label"]),
                "hours": hours,
                "weight": weight,
            })
            horizon_scores.append(score)

        def weighted_average(key):
            return float(sum(
                float(item["weight"]) * float(item[key])
                for item in horizon_scores
            ))

        # 数量指标使用最长 horizon 仅用于诊断；不再作为硬约束。
        longest = max(horizon_scores, key=lambda x: int(x["hours"]))

        aggregate = {
            "joint_score": weighted_average("joint_score"),
            "time_cdf_distance": weighted_average("time_cdf_distance"),
            "hourly_wape": weighted_average("hourly_wape"),
            "week_hour_distance": weighted_average("week_hour_distance"),
            "count_bias": weighted_average("count_bias"),
            "concentration_error": weighted_average("concentration_error"),
            "actual_concentration": weighted_average("actual_concentration"),
            "predicted_concentration": weighted_average("predicted_concentration"),
            "signed_count_bias": weighted_average("signed_count_bias"),
            "actual_total": float(longest["actual_total"]),
            "predicted_total": float(longest["predicted_total"]),
            "validation_hours": int(longest["hours"]),
            "longest_horizon_label": str(longest["label"]),
            "horizon_scores": horizon_scores,
        }
        return aggregate

    @staticmethod
    def _set_training_seed(seed=42):
        """固定模型训练随机性，使不同参数 trial 的比较尽量只反映参数差异。"""
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _parse_float_candidates(value, default):
        if value is None:
            value = default
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",") if x.strip()]
        if not isinstance(value, (list, tuple, set, np.ndarray)):
            value = [value]
        parsed = []
        for item in value:
            try:
                v = float(item)
            except Exception:
                continue
            if np.isfinite(v):
                parsed.append(v)
        return parsed or list(default)

    def _apply_rate_calendar_blend(
            self,
            pred_df,
            week_hour_scale,
            calendar_blend_alpha=1.0,
            rate_scale=1.0):
        """
        简化生成校准：
          1) rate_scale 只修正总到达质量；
          2) alpha 在 NeuralProphet 时间形状与训练日志 calendar prior 间做凸组合。

        alpha=1 完全使用 NeuralProphet；alpha=0 完全使用 weekday-hour prior。
        该操作保持块内总质量为 rate_scale * sum(original yhat1)。
        """
        if pred_df is None or pred_df.empty:
            return pred_df

        result = pred_df.copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)

        alpha = float(np.clip(calendar_blend_alpha, 0.0, 1.0))
        rate_scale = float(rate_scale)
        if not np.isfinite(rate_scale) or rate_scale <= 0.0:
            rate_scale = 1.0

        point = pd.to_numeric(
            result["yhat1"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        original_total = float(point.sum())
        if original_total <= 1e-12:
            return result

        np_prob = point / original_total
        calendar_score = np.asarray([
            float(week_hour_scale.get(
                (int(ts.dayofweek), int(ts.hour)), 1.0
            ))
            for ts in result["ds"]
        ], dtype=float)
        calendar_score = np.nan_to_num(
            calendar_score, nan=1.0, posinf=1.0, neginf=0.0
        )
        calendar_score = np.clip(calendar_score, 0.0, None)
        calendar_total = float(calendar_score.sum())
        if calendar_total <= 1e-12:
            calendar_prob = np.full(len(point), 1.0 / max(len(point), 1))
        else:
            calendar_prob = calendar_score / calendar_total

        blended_prob = alpha * np_prob + (1.0 - alpha) * calendar_prob
        blended_total = float(blended_prob.sum())
        if blended_total <= 1e-12 or not np.isfinite(blended_total):
            blended_prob = np_prob
        else:
            blended_prob = blended_prob / blended_total

        target_total = original_total * rate_scale
        transformed = np.maximum(target_total * blended_prob, 0.0)
        result["yhat1"] = transformed

        # 若其他配置仍使用预测区间，对分位数同步做近似缩放。
        ratio = np.ones_like(point, dtype=float)
        positive = point > 1e-12
        ratio[positive] = transformed[positive] / point[positive]
        low_col, high_col = self._quantile_columns(result)
        for col, default_factor in ((low_col, 0.8), (high_col, 1.2)):
            if col is None or col not in result.columns:
                continue
            values = pd.to_numeric(
                result[col], errors="coerce"
            ).fillna(0.0).clip(lower=0.0).to_numpy(float)
            adjusted = values * ratio
            adjusted[~positive] = transformed[~positive] * default_factor
            result[col] = np.maximum(adjusted, 0.0)

        return result

    def _fit_simple_generation_calibration(
            self,
            df_train,
            best_params,
            final_epochs,
            batch_size,
            daily_seasonality,
            weekly_seasonality,
            yearly_seasonality,
            validation_plan,
            rolling_plan,
            objective_weights,
            config,
            cap_multiplier):
        """
        在 training partition 内使用 rolling-origin × multi-horizon 学习生成校准：
          - rate_scale：所有 origin/horizon 的 actual/predicted 比值中位数；
          - calendar_blend_alpha：NeuralProphet 与 weekday-hour prior 的混合权重；
          - calendar_smoothing：weekday-hour prior 的平滑强度。

        alpha/smoothing 使用小规模候选网格选择，不新增贝叶斯优化；正式 test 日志
        始终不参与选择。
        """
        default_smoothing = float(config.get("calendar_smoothing", 0.50))
        default = {
            "version": "simple_rate_calendar_v2_rolling",
            "enabled": False,
            "rate_scale": 1.0,
            "calendar_blend_alpha": 1.0,
            "calendar_smoothing": default_smoothing,
            "source": "training_internal_rolling_origin_multi_horizon_validation",
        }
        try:
            origins = list(rolling_plan.get("origin_indices", []))
            max_horizon = int(validation_plan["max_horizon_hours"])
            if not origins:
                default["reason"] = "empty_rolling_origin_plan"
                return default

            manual = self.parms.get("arrival_manual_best_params") or config.get("manual_best_params") or {}
            if not isinstance(manual, dict):
                manual = {}

            alpha_source = self.parms.get(
                "arrival_calendar_blend_alpha_candidates",
                config.get("calendar_blend_alpha_candidates", [0.70, 0.85, 1.00]),
            )
            manual_alpha_fixed = "calendar_blend_alpha" in manual
            if manual_alpha_fixed:
                alpha_source = [manual["calendar_blend_alpha"]]
            alpha_candidates = sorted({
                float(np.clip(x, 0.0, 1.0))
                for x in self._parse_float_candidates(alpha_source, [0.70, 0.85, 1.00])
            })
            # 自动/候选模式保留 alpha=1 作为“纯 NeuralProphet”基线；
            # 手动配置模式则严格使用用户/配置指定的 alpha。
            if (not manual_alpha_fixed) and 1.0 not in alpha_candidates:
                alpha_candidates.append(1.0)
                alpha_candidates = sorted(alpha_candidates)

            smoothing_source = self.parms.get(
                "arrival_calendar_smoothing_candidates",
                config.get(
                    "calendar_smoothing_candidates",
                    [config.get("calendar_smoothing", 0.50)],
                ),
            )
            if "calendar_smoothing" in manual:
                smoothing_source = [manual["calendar_smoothing"]]
            smoothing_candidates = sorted({
                float(np.clip(x, 0.0, 1.0))
                for x in self._parse_float_candidates(smoothing_source, [default_smoothing])
            })

            fold_records = []
            rate_ratios = []
            training_seed = int(self.parms.get("arrival_training_seed", 42))

            for fold_idx, origin in enumerate(origins):
                origin = int(origin)
                cal_train = df_train.iloc[:origin][["ds", "y"]].copy().reset_index(drop=True)
                cal_valid = df_train.iloc[origin:origin + max_horizon][["ds", "y"]].copy().reset_index(drop=True)
                if len(cal_train) < 24 or cal_valid.empty:
                    continue

                fold_cap = self._calculate_max_cap(cal_train, cap_multiplier)
                self._set_training_seed(training_seed + fold_idx)
                model = NeuralProphet(
                    **best_params,
                    quantiles=[0.1, 0.9],
                    global_normalization=True,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    growth="linear",
                    batch_size=batch_size,
                    trainer_config={
                        "enable_checkpointing": False,
                        "accelerator": "cpu",
                        "logger": False,
                    },
                )
                model.fit(
                    cal_train,
                    progress="none",
                    freq="h",
                    epochs=final_epochs,
                )
                pred = self._predict_validation_arrivals(
                    model=model,
                    train_df=cal_train,
                    validation_df=cal_valid,
                    max_cap=fold_cap,
                )
                if pred is None or pred.empty or "yhat1" not in pred.columns:
                    continue

                for item in validation_plan.get("horizons", []):
                    hours = int(item["hours"])
                    actual_slice = cal_valid.iloc[:hours].copy()
                    aligned = self._prepare_arrival_validation_frames(actual_slice, pred)
                    actual_total = float(aligned["y"].sum())
                    predicted_total = float(aligned["yhat1"].sum())
                    if predicted_total > 1e-12 and actual_total > 1e-12:
                        rate_ratios.append(actual_total / predicted_total)

                fold_records.append({
                    "fold": int(fold_idx),
                    "origin": int(origin),
                    "train": cal_train,
                    "valid": cal_valid,
                    "pred": pred,
                })

            if not fold_records:
                raise ValueError("no valid rolling-origin calibration folds")

            raw_rate_scale = float(np.median(rate_ratios)) if rate_ratios else 1.0
            min_scale = float(self.parms.get(
                "arrival_rate_scale_min", config.get("rate_scale_min", 0.50)
            ))
            max_scale = float(self.parms.get(
                "arrival_rate_scale_max", config.get("rate_scale_max", 2.00)
            ))
            rate_scale = float(np.clip(raw_rate_scale, min_scale, max_scale))

            candidate_scores = []
            for smoothing in smoothing_candidates:
                for alpha in alpha_candidates:
                    fold_scores = []
                    for rec in fold_records:
                        calendar_scale = self._build_week_hour_scale(
                            history=rec["train"],
                            smoothing=smoothing,
                        )
                        calibrated = self._apply_rate_calendar_blend(
                            pred_df=rec["pred"],
                            week_hour_scale=calendar_scale,
                            calendar_blend_alpha=alpha,
                            rate_scale=rate_scale,
                        )
                        score = self._score_multi_horizon_validation(
                            actual_full=rec["valid"],
                            pred_full=calibrated,
                            validation_plan=validation_plan,
                            weights=objective_weights,
                        )
                        fold_scores.append(score)

                    def avg(key):
                        return float(np.mean([float(x[key]) for x in fold_scores]))

                    candidate_scores.append({
                        "alpha": float(alpha),
                        "smoothing": float(smoothing),
                        "score": avg("joint_score"),
                        "time_cdf_distance": avg("time_cdf_distance"),
                        "week_hour_distance": avg("week_hour_distance"),
                        "count_bias": avg("count_bias"),
                        "concentration_error": avg("concentration_error"),
                        "n_folds": int(len(fold_scores)),
                    })

            best = min(candidate_scores, key=lambda x: x["score"])
            if manual_alpha_fixed:
                # 手动实验：alpha 已固定，不再用 alpha=1 基线覆盖人工配置。
                baseline = best
                relative_improvement = 0.0
            else:
                baseline_candidates = [
                    x for x in candidate_scores if abs(x["alpha"] - 1.0) <= 1e-12
                ]
                baseline = min(baseline_candidates, key=lambda x: x["score"])
                relative_improvement = (
                    (baseline["score"] - best["score"])
                    / max(abs(baseline["score"]), 1e-12)
                )
                min_improvement = float(
                    self.parms.get("arrival_calendar_blend_min_relative_improvement", 0.005)
                )
                if best["alpha"] != 1.0 and relative_improvement < min_improvement:
                    best = baseline
                    relative_improvement = 0.0

            result = {
                "version": "simple_rate_calendar_v2_rolling",
                "enabled": True,
                "rate_scale": rate_scale,
                "raw_rate_scale": raw_rate_scale,
                "rate_ratio_candidates": [float(x) for x in rate_ratios],
                "calendar_blend_alpha": float(best["alpha"]),
                "calendar_smoothing": float(best["smoothing"]),
                "baseline_alpha1_score": float(baseline["score"]),
                "selected_score": float(best["score"]),
                "relative_improvement": float(relative_improvement),
                "candidate_scores": candidate_scores,
                "rolling_origins_used": [int(x["origin"]) for x in fold_records],
                "source": "training_internal_rolling_origin_multi_horizon_validation",
            }
            print(
                "[滚动生成校准] "
                f"rate_scale={rate_scale:.4f} (raw={raw_rate_scale:.4f}), "
                f"calendar_alpha={result['calendar_blend_alpha']:.2f}, "
                f"smoothing={result['calendar_smoothing']:.2f}, "
                f"folds={len(fold_records)}, "
                f"relative_improvement={relative_improvement:.2%}"
            )
            return result
        except Exception as error:
            default["reason"] = str(error)
            print(f"[滚动生成校准] 失败，退化为 scale=1/alpha=1: {error}")
            return default

    def _evaluate_long_horizon_stability(
            self,
            train_df,
            predicted_total,
            horizon_hours):
        """
        检查最长 horizon 是否发生“远期强度塌缩”。

        参考强度只由预测起点之前的训练历史估计：
            expected = recent_training_rate * horizon_hours
            stability_ratio = predicted_total / expected

        默认当 stability_ratio < 0.30 时标记为 collapse 警告。
        该指标只做诊断，不淘汰 trial，也不读取 validation/test 的真实总量。
        """
        horizon_hours = int(max(1, horizon_hours))
        threshold = float(
            self.parms.get(
                "arrival_long_horizon_min_stability_ratio",
                0.30,
            )
        )
        threshold = float(np.clip(threshold, 0.0, 1.0))

        reference_days = float(
            self.parms.get(
                "arrival_long_horizon_reference_days",
                28.0,
            )
        )
        reference_hours = max(24, int(round(reference_days * 24.0)))
        reference_hours = min(reference_hours, max(len(train_df), 1))

        recent = pd.to_numeric(
            train_df.tail(reference_hours)["y"],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)

        reference_rate = float(recent.mean()) if len(recent) else 0.0
        expected_cases = float(reference_rate * horizon_hours)
        predicted_total = float(max(predicted_total, 0.0))

        min_expected_cases = float(
            self.parms.get(
                "arrival_long_horizon_min_expected_cases",
                3.0,
            )
        )

        if expected_cases <= max(min_expected_cases, 1e-12):
            return {
                "enabled": False,
                "collapse": False,
                "ratio": 1.0,
                "threshold": threshold,
                "reference_rate": reference_rate,
                "reference_hours": int(reference_hours),
                "expected_cases": expected_cases,
                "predicted_cases": predicted_total,
                "reason": "reference_expected_mass_too_small",
            }

        ratio = float(predicted_total / expected_cases)
        collapse = bool(
            np.isfinite(ratio)
            and ratio < threshold
        )
        if not np.isfinite(ratio):
            collapse = True

        return {
            "enabled": True,
            "collapse": collapse,
            "ratio": ratio,
            "threshold": threshold,
            "reference_rate": reference_rate,
            "reference_hours": int(reference_hours),
            "expected_cases": expected_cases,
            "predicted_cases": predicted_total,
            "reason": "ok" if not collapse else "long_horizon_intensity_collapse",
        }

    def _discover_model(self):
        # 预处理阶段暂不使用最终 cap；cap 在选定日志配置后按 cap_multiplier 计算。
        df_train, _ = self._transform_features(self.log, cap_multiplier=1.0)

        if len(df_train) < 24:
            print("[NeuralProphet] Warning: Data extremely small (<24h).")

        valid_p = 0.2
        if len(df_train) < 720:
            valid_p = 0.15
        if len(df_train) < 200:
            valid_p = 0.10

        multi_horizon_plan = self._get_multi_horizon_validation_plan(
            n_hours=len(df_train),
            valid_p=valid_p,
        )
        rolling_plan = self._get_rolling_origin_validation_plan(
            n_hours=len(df_train),
            validation_plan=multi_horizon_plan,
        )
        reserved_validation_hours = int(multi_horizon_plan["max_horizon_hours"])

        # 搜索空间画像只能看最早 rolling origin 之前的历史，避免未来 fold 信息泄漏。
        earliest_origin = int(rolling_plan["earliest_origin"])
        selection_history = (
            df_train.iloc[:earliest_origin]
            .copy().reset_index(drop=True)
        )
        if selection_history.empty:
            selection_history = df_train.iloc[:max(24, len(df_train) // 2)].copy().reset_index(drop=True)

        print(
            "[Rolling-Origin × Multi-Horizon 验证方案] "
            + ", ".join(
                f"{item['label']}@{item['weight']:.2f}"
                for item in multi_horizon_plan["horizons"]
            )
            + f", origins={rolling_plan['origin_indices']}, "
              f"step={rolling_plan['origin_step_hours']}h, "
              f"max_horizon={reserved_validation_hours}h"
        )

        total_cases = max(1, int(self.log[La.CASE_ID].nunique()))
        full_train_profile = self._build_generation_profile(
            df_history=df_train,
            num_instances=total_cases,
            total_cases=total_cases,
        )
        selection_cases = max(
            1,
            int(round(pd.to_numeric(
                selection_history["y"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0).sum())),
        )
        selection_profile = self._build_generation_profile(
            df_history=selection_history,
            num_instances=selection_cases,
            total_cases=selection_cases,
        )

        print("[训练期 arrival 画像 - 全训练分区，仅记录]")
        print(f"mean_rate={full_train_profile['mean_rate']:.4f}, "
              f"nonzero_ratio={full_train_profile['nonzero_ratio']:.4f}, "
              f"sparsity={full_train_profile['sparsity']:.4f}, "
              f"daily_strength={full_train_profile['daily_strength']:.4f}, "
              f"weekly_strength={full_train_profile['weekly_strength']:.4f}, "
              f"burst_ratio={full_train_profile['burst_ratio']:.4f}")
        print("[模型选择画像 - 仅使用最早 rolling origin 之前历史]")
        print(f"hours={len(selection_history)}, "
              f"mean_rate={selection_profile['mean_rate']:.4f}, "
              f"nonzero_ratio={selection_profile['nonzero_ratio']:.4f}, "
              f"daily_strength={selection_profile['daily_strength']:.4f}, "
              f"weekly_strength={selection_profile['weekly_strength']:.4f}")

        config, selected_config = self._select_training_config(
            selection_profile, len(selection_history)
        )

        # 数据集配置 -> 生成阶段运行参数
        if (
                "arrival_count_sampling" not in self.parms
                and "arrival_count_sampling" in config
        ):
            self.parms["arrival_count_sampling"] = config["arrival_count_sampling"]

        manual_overrides = (
                self.parms.get("arrival_manual_best_params")
                or config.get("manual_best_params")
                or {}
        )

        if not isinstance(manual_overrides, dict):
            manual_overrides = {}
        cap_multiplier = float(
            manual_overrides.get(
                "cap_multiplier",
                self.parms.get(
                    "arrival_cap_multiplier",
                    config.get("cap_multiplier", 1.5),
                ),
            )
        )
        if (not np.isfinite(cap_multiplier)) or cap_multiplier <= 0.0:
            cap_multiplier = 1.5
        self.max_cap = self._calculate_max_cap(df_train, cap_multiplier)

        self.model_metadata["arrival_model_config"] = selected_config
        self.model_metadata["cap_multiplier"] = float(cap_multiplier)
        self.model_metadata["max_cap"] = float(self.max_cap)
        self.model_metadata["train_profile"] = full_train_profile
        self.model_metadata["selection_profile"] = selection_profile
        self.model_metadata["selection_profile_hours"] = int(len(selection_history))

        print(f"\n[NeuralProphet Search] Mode: {selected_config}")
        print(f"[NeuralProphet Search] Desc: {config['description']}")
        print(f"[NeuralProphet Search] Data Size: {len(df_train)} hours")

        # 复制一份 param_grid，避免直接改全局配置
        param_grid = {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in config["param_grid"].items()
        }

        # 数据驱动 lag 候选扩展：依据 selection_profile 决定是否允许 24/168h。
        # 但当配置显式限制 n_lags=[0]（如 MP_SPARSE_ADAPTIVE）时，跳过扩展，
        # 避免向搜索空间注入非预期参数（n_lags=24 + ar_reg=0.1/0.5）。
        config_lags = [int(x) for x in param_grid.get("n_lags", [0])]
        allow_auto_lag_expansion = bool(
            config.get("allow_auto_lag_expansion", True)
        )
        if (not allow_auto_lag_expansion) or all(x == 0 for x in config_lags):
            # MP 等稀疏日志显式固定 n_lags=0，不再自动注入 24/168h AR。
            param_grid["n_lags"] = sorted(set(config_lags or [0]))
            if all(x <= 0 for x in param_grid["n_lags"]):
                param_grid["ar_reg"] = [0.0]
            print(
                "[数据驱动 lag 候选] 跳过自动扩展: "
                f"allow_auto={allow_auto_lag_expansion}, "
                f"configured_lags={param_grid['n_lags']}"
            )
        else:
            lag_candidates = set(config_lags)
            lag_candidates.add(0)
            positive_hours = int(round(
                len(selection_history)
                * float(selection_profile.get("nonzero_ratio", 0.0))
            ))
            if (
                len(selection_history) >= 24 * 14
                and positive_hours >= 30
                and float(selection_profile.get("daily_strength", 0.0)) >= 0.10
            ):
                lag_candidates.add(24)
            if (
                len(selection_history) >= 24 * 56
                and positive_hours >= 80
                and float(selection_profile.get("weekly_strength", 0.0)) >= 0.10
            ):
                lag_candidates.add(168)
            param_grid["n_lags"] = sorted(lag_candidates)
            if any(x > 0 for x in lag_candidates):
                ar_values = param_grid.get("ar_reg", [0.0])
                if not isinstance(ar_values, list):
                    ar_values = [ar_values]
                ar_values = sorted({float(x) for x in ar_values})
                if len(ar_values) == 1 and abs(ar_values[0]) <= 1e-12:
                    ar_values = [0.0, 0.1, 0.5]
                param_grid["ar_reg"] = ar_values
            print(
                "[数据驱动 lag 候选] "
                f"positive_hours={positive_hours}, "
                f"lags={param_grid['n_lags']}, "
                f"source=selection_profile_before_earliest_rolling_origin"
            )

        daily_default = config.get("daily_seasonality", False)
        weekly_default = config.get("weekly_seasonality", False)
        yearly_seasonality = config.get("yearly_seasonality", False)
        adaptive_seasonality = bool(
            config.get("adaptive_seasonality", True)
        )

        # 显式配置 adaptive_seasonality=False 时，必须严格保留配置中的
        # daily/weekly 阶数。旧代码无条件执行自适应解析，导致 MP 中
        # daily_seasonality=4 被高稀疏性惩罚错误关闭。
        if adaptive_seasonality:
            (
                daily_seasonality,
                weekly_seasonality,
                seasonality_reg_scale
            ) = self._resolve_adaptive_seasonality(
                profile=selection_profile,
                daily_default=daily_default,
                weekly_default=weekly_default
            )
            seasonality_mode = "adaptive"
        else:
            daily_seasonality = daily_default
            weekly_seasonality = weekly_default
            seasonality_reg_scale = 1.0
            seasonality_mode = "fixed"

        self.model_metadata["seasonality_mode"] = seasonality_mode
        self.model_metadata["daily_seasonality"] = daily_seasonality
        self.model_metadata["weekly_seasonality"] = weekly_seasonality
        self.model_metadata["yearly_seasonality"] = yearly_seasonality

        # 周期弱时，提高 seasonality_reg，避免拟合出伪节律
        if "seasonality_reg" in param_grid and isinstance(param_grid["seasonality_reg"], list):
            param_grid["seasonality_reg"] = sorted(
                list({
                    round(float(v) * seasonality_reg_scale, 4)
                    for v in param_grid["seasonality_reg"]
                })
            )

        # 仅 AUTO/显式允许时才自动扩展变点；人工日志配置严格保留各自趋势范围。
        allow_auto_changepoint_expansion = bool(
            config.get("allow_auto_changepoint_expansion", True)
        )
        if (
            allow_auto_changepoint_expansion
            and (daily_seasonality is False)
            and (weekly_seasonality is False)
        ):
            if "n_changepoints" in param_grid and isinstance(param_grid["n_changepoints"], list):
                current_ncp = sorted(set(int(x) for x in param_grid["n_changepoints"]))
                if len(selection_history) >= 24 * 30:
                    current_ncp.append(min(max(current_ncp) + 3, 20))
                param_grid["n_changepoints"] = sorted(set(current_ncp))

        print(
            f"[季节性配置] mode={seasonality_mode}, "
            f"daily={daily_seasonality}, "
            f"weekly={weekly_seasonality}, "
            f"yearly={yearly_seasonality}"
        )
        print(f"[有效 param_grid] {param_grid}")

        epochs = config["epochs"]
        batch_size = config["batch_size"]
        n_trials = config.get("n_trials", 20)

        # ============================================================
        # 形状优先的软联合目标
        # ============================================================
        objective_weights = self._get_arrival_objective_weights(config=config)
        self.model_metadata["arrival_objective_version"] = "v5_rolling_timecdf_softcount"
        self.model_metadata["multi_horizon_validation"] = multi_horizon_plan
        self.model_metadata["rolling_origin_validation"] = rolling_plan
        self.model_metadata["arrival_objective"] = {
            "name": "rolling_origin_multi_horizon_timecdf_week_hour_soft_count_concentration_v5",
            "direction": "minimize",
            "weights": objective_weights,
            "horizon_weights": {
                item["label"]: item["weight"]
                for item in multi_horizon_plan["horizons"]
            },
            "time_cdf": "mean_absolute_cdf_distance",
            "week_hour_distance": "total_variation_distance_168_bins",
            "uses_sampling": False,
            "hard_constraints": False,
            "catastrophic_rejection_only": [
                "non_finite_joint_score",
                "empty_prediction",
                "predicted_total_approximately_zero",
            ],
        }
        print(
            "[到达模型软联合目标] "
            f"TimeCDF={objective_weights['time_cdf']:.2f}, "
            f"WeekHour={objective_weights['week_hour']:.2f}, "
            f"CountBias={objective_weights['count']:.2f}, "
            f"Concentration={objective_weights['concentration']:.2f}; "
            "无 CountRatio/CountBias 硬淘汰"
        )

        def objective(trial):
            params = {}
            for key, values in param_grid.items():
                if key == "ar_reg":
                    continue
                if isinstance(values, list):
                    params[key] = trial.suggest_categorical(key, values)
                else:
                    params[key] = values
            if int(params.get("n_lags", 0)) <= 0:
                params["ar_reg"] = 0.0
            else:
                ar_values = param_grid.get("ar_reg", [0.0])
                params["ar_reg"] = (
                    trial.suggest_categorical("ar_reg", ar_values)
                    if isinstance(ar_values, list)
                    else ar_values
                )

            try:
                origins = list(rolling_plan.get("origin_indices", []))
                max_horizon_hours = int(multi_horizon_plan["max_horizon_hours"])
                if not origins:
                    return float("inf")

                training_seed = int(self.parms.get("arrival_training_seed", 42))
                fold_scores = []
                fold_long_horizon = []
                fold_rmse = []
                rolling_details = []

                for fold_idx, origin in enumerate(origins):
                    origin = int(origin)
                    df_train_split = (
                        df_train.iloc[:origin]
                        .copy().reset_index(drop=True)
                    )
                    df_val_split = (
                        df_train.iloc[origin:origin + max_horizon_hours]
                        .copy().reset_index(drop=True)
                    )
                    if len(df_train_split) < 24 or df_val_split.empty:
                        continue

                    # cap 只能由当前 origin 之前的训练折计算。
                    fold_cap = self._calculate_max_cap(
                        df_train_split,
                        cap_multiplier=cap_multiplier,
                    )

                    self._set_training_seed(training_seed + fold_idx)
                    m = NeuralProphet(
                        **params,
                        quantiles=[0.1, 0.9],
                        global_normalization=True,
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        growth="linear",
                        batch_size=batch_size,
                        trainer_config={
                            "enable_checkpointing": False,
                            "accelerator": "cpu",
                            "logger": False,
                        },
                    )
                    metrics = m.fit(
                        df_train_split,
                        progress="none",
                        freq="h",
                        epochs=epochs,
                    )

                    pred_val = self._predict_validation_arrivals(
                        model=m,
                        train_df=df_train_split,
                        validation_df=df_val_split,
                        max_cap=fold_cap,
                    )
                    if pred_val is None or pred_val.empty or "yhat1" not in pred_val.columns:
                        return float("inf")

                    score = self._score_multi_horizon_validation(
                        actual_full=df_val_split,
                        pred_full=pred_val,
                        validation_plan=multi_horizon_plan,
                        weights=objective_weights,
                    )
                    if not np.isfinite(score["joint_score"]):
                        return float("inf")
                    if float(score.get("predicted_total", 0.0)) <= 1e-8:
                        return float("inf")

                    long_horizon = self._evaluate_long_horizon_stability(
                        train_df=df_train_split,
                        predicted_total=score["predicted_total"],
                        horizon_hours=score["validation_hours"],
                    )
                    fold_scores.append(score)
                    fold_long_horizon.append(long_horizon)

                    diagnostic_rmse = None
                    if isinstance(metrics, pd.DataFrame) and "RMSE_val" in metrics.columns:
                        rmse_values = pd.to_numeric(
                            metrics["RMSE_val"], errors="coerce"
                        ).dropna()
                        if not rmse_values.empty:
                            diagnostic_rmse = float(rmse_values.min())
                            fold_rmse.append(diagnostic_rmse)

                    rolling_details.append({
                        "fold": int(fold_idx),
                        "origin_index": int(origin),
                        "train_end": str(pd.to_datetime(df_train_split["ds"].max())),
                        "validation_start": str(pd.to_datetime(df_val_split["ds"].min())),
                        "validation_end": str(pd.to_datetime(df_val_split["ds"].max())),
                        "cap": float(fold_cap),
                        "joint_score": float(score["joint_score"]),
                        "time_cdf_distance": float(score["time_cdf_distance"]),
                        "week_hour_distance": float(score["week_hour_distance"]),
                        "count_bias": float(score["count_bias"]),
                        "concentration_error": float(score["concentration_error"]),
                        "actual_total": float(score["actual_total"]),
                        "predicted_total": float(score["predicted_total"]),
                        "horizon_scores": score["horizon_scores"],
                        "long_horizon_stability_ratio": float(long_horizon["ratio"]),
                        "long_horizon_collapse": bool(long_horizon["collapse"]),
                    })

                if not fold_scores:
                    return float("inf")

                def avg_score(key):
                    return float(np.mean([float(x[key]) for x in fold_scores]))

                aggregate = {
                    "joint_score": avg_score("joint_score"),
                    "time_cdf_distance": avg_score("time_cdf_distance"),
                    "hourly_wape": avg_score("hourly_wape"),
                    "week_hour_distance": avg_score("week_hour_distance"),
                    "count_bias": avg_score("count_bias"),
                    "concentration_error": avg_score("concentration_error"),
                    "actual_concentration": avg_score("actual_concentration"),
                    "predicted_concentration": avg_score("predicted_concentration"),
                    "signed_count_bias": avg_score("signed_count_bias"),
                    "actual_total": avg_score("actual_total"),
                    "predicted_total": avg_score("predicted_total"),
                    "validation_hours": int(max_horizon_hours),
                }

                # 按 horizon 标签汇总所有 rolling origins，保留原 metadata 接口。
                horizon_summary = []
                for h in multi_horizon_plan.get("horizons", []):
                    label = str(h["label"])
                    matching = []
                    for score in fold_scores:
                        matching.extend([
                            item for item in score["horizon_scores"]
                            if str(item["label"]) == label
                        ])
                    if not matching:
                        continue
                    horizon_summary.append({
                        "label": label,
                        "hours": int(h["hours"]),
                        "weight": float(h["weight"]),
                        "joint_score": float(np.mean([x["joint_score"] for x in matching])),
                        "time_cdf_distance": float(np.mean([x["time_cdf_distance"] for x in matching])),
                        "hourly_wape": float(np.mean([x["hourly_wape"] for x in matching])),
                        "week_hour_distance": float(np.mean([x["week_hour_distance"] for x in matching])),
                        "count_bias": float(np.mean([x["count_bias"] for x in matching])),
                        "concentration_error": float(np.mean([x["concentration_error"] for x in matching])),
                        "actual_total": float(np.mean([x["actual_total"] for x in matching])),
                        "predicted_total": float(np.mean([x["predicted_total"] for x in matching])),
                    })

                rate_diag = self._arrival_rate_diagnostics(aggregate)
                long_ratio = float(np.mean([
                    float(x["ratio"]) for x in fold_long_horizon
                ])) if fold_long_horizon else 1.0
                long_expected = float(np.mean([
                    float(x["expected_cases"]) for x in fold_long_horizon
                ])) if fold_long_horizon else 0.0
                long_predicted = float(np.mean([
                    float(x["predicted_cases"]) for x in fold_long_horizon
                ])) if fold_long_horizon else 0.0
                long_reference_rate = float(np.mean([
                    float(x["reference_rate"]) for x in fold_long_horizon
                ])) if fold_long_horizon else 0.0
                long_collapse = any(bool(x["collapse"]) for x in fold_long_horizon)
                diagnostic_rmse = float(np.mean(fold_rmse)) if fold_rmse else None

                trial.set_user_attr("best_epoch", int(epochs))
                trial.set_user_attr("time_cdf_distance", aggregate["time_cdf_distance"])
                trial.set_user_attr("hourly_wape", aggregate["hourly_wape"])
                trial.set_user_attr("week_hour_distance", aggregate["week_hour_distance"])
                trial.set_user_attr("count_bias", aggregate["count_bias"])
                trial.set_user_attr("concentration_error", aggregate["concentration_error"])
                trial.set_user_attr("actual_concentration", aggregate["actual_concentration"])
                trial.set_user_attr("predicted_concentration", aggregate["predicted_concentration"])
                trial.set_user_attr("signed_count_bias", aggregate["signed_count_bias"])
                trial.set_user_attr("actual_total", aggregate["actual_total"])
                trial.set_user_attr("predicted_total", aggregate["predicted_total"])
                trial.set_user_attr("validation_hours", aggregate["validation_hours"])
                trial.set_user_attr("count_ratio", rate_diag["count_ratio"])
                trial.set_user_attr("diagnostic_near_zero", rate_diag["near_zero"])
                trial.set_user_attr("multi_horizon_scores", horizon_summary)
                trial.set_user_attr("rolling_origin_scores", rolling_details)
                trial.set_user_attr("long_horizon_stability_ratio", long_ratio)
                trial.set_user_attr("long_horizon_expected_cases", long_expected)
                trial.set_user_attr("long_horizon_predicted_cases", long_predicted)
                trial.set_user_attr("long_horizon_reference_rate", long_reference_rate)
                trial.set_user_attr("long_horizon_collapse", bool(long_collapse))
                if diagnostic_rmse is not None:
                    trial.set_user_attr("diagnostic_rmse_val", diagnostic_rmse)

                objective_value = float(aggregate["joint_score"])
                warning_flags = []
                if rate_diag["near_zero"]:
                    warning_flags.append("near_zero")
                if long_collapse:
                    warning_flags.append("long_horizon_collapse")
                status = "WARNING" if warning_flags else "VALID"
                param_summary = ", ".join(f"{k}={v}" for k, v in params.items())
                print(
                    f"[Trial {trial.number}] {status} | rolling_folds={len(fold_scores)} | "
                    f"params=[{param_summary}] | objective={objective_value:.6f} | "
                    f"TimeCDF={aggregate['time_cdf_distance']:.6f} | "
                    f"WeekHour={aggregate['week_hour_distance']:.6f} | "
                    f"CountBias={aggregate['count_bias']:.6f} | "
                    f"Concentration={aggregate['concentration_error']:.6f} | "
                    f"CountRatio={rate_diag['count_ratio']:.4f} | "
                    f"LongStability={long_ratio:.4f}"
                    + (f" | warnings={warning_flags}" if warning_flags else "")
                )
                return objective_value

            except Exception as e:
                print(f"[Trial {trial.number}] ❌ 异常中断: {str(e)}")
                import traceback
                traceback.print_exc()
                return float("inf")

        # ============================================================
        # 手动最佳参数旁路：跳过 Optuna 搜索，直接用指定参数训练 1 次
        # 优先级: 命令行 --manual_ia_params > config.manual_best_params
        # 效果: n_trials=30 -> 1, 总训练 34 -> 5
        # ============================================================
        manual_best_params = (
            self.parms.get("arrival_manual_best_params")
            or config.get("manual_best_params")
        )
        if manual_best_params and isinstance(manual_best_params, dict):
            allowed_lags = {int(x) for x in param_grid.get("n_lags", [0])}
            if "n_lags" in manual_best_params:
                manual_lag = int(manual_best_params["n_lags"])
                if manual_lag not in allowed_lags:
                    raise ValueError(
                        f"Manual n_lags={manual_lag} is not allowed for {selected_config}; "
                        f"allowed={sorted(allowed_lags)}"
                    )
            print(f"[手动参数旁路] 跳过 Optuna 搜索，使用指定参数:")
            for k, v in manual_best_params.items():
                print(f"  {k}: {v}")
            for key in list(param_grid.keys()):
                if key in manual_best_params:
                    param_grid[key] = [manual_best_params[key]]
            n_trials = 1

        search_method = str(config.get("search_method", "tpe")).strip().lower()
        if manual_best_params and isinstance(manual_best_params, dict):
            search_method = "grid"

        if search_method == "grid":
            # GridSampler 只放 objective 实际 suggest 的参数；MP n_lags=0 时不 suggest ar_reg。
            grid_space = {}
            for key, values in param_grid.items():
                if key == "ar_reg" and all(
                    int(x) <= 0 for x in param_grid.get("n_lags", [0])
                ):
                    continue
                grid_space[key] = values if isinstance(values, list) else [values]
            grid_size = int(np.prod([len(v) for v in grid_space.values()])) if grid_space else 1
            n_trials = min(int(n_trials), grid_size)
            sampler = optuna.samplers.GridSampler(search_space=grid_space)
            print(
                f"开始 Optuna Grid Search (n_trials={n_trials}/{grid_size})..."
            )
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
            print(f"开始 Optuna TPE 联合目标优化 (n_trials={n_trials})...")

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        valid_trials = [
            t for t in complete_trials
            if t.value is not None and np.isfinite(float(t.value))
        ]
        if not valid_trials:
            raise RuntimeError(
                "No valid NeuralProphet arrival-model trial completed successfully. "
                "Check training errors or whether all predictions are approximately zero."
            )

        selected_trial = min(valid_trials, key=lambda t: float(t.value))
        best_attrs = dict(selected_trial.user_attrs)
        selected_score = float(selected_trial.value)
        near_zero_trials = [
            t for t in valid_trials
            if bool(t.user_attrs.get("diagnostic_near_zero", False))
        ]

        self.model_metadata["best_joint_objective"] = {
            "score": selected_score,
            "time_cdf_distance": best_attrs.get("time_cdf_distance"),
            "hourly_wape_diagnostic": best_attrs.get("hourly_wape"),
            "week_hour_distance": best_attrs.get("week_hour_distance"),
            "count_bias": best_attrs.get("count_bias"),
            "concentration_error": best_attrs.get("concentration_error"),
            "actual_concentration": best_attrs.get("actual_concentration"),
            "predicted_concentration": best_attrs.get("predicted_concentration"),
            "signed_count_bias": best_attrs.get("signed_count_bias"),
            "count_ratio": best_attrs.get("count_ratio"),
            "actual_total": best_attrs.get("actual_total"),
            "predicted_total": best_attrs.get("predicted_total"),
            "diagnostic_rmse_val": best_attrs.get("diagnostic_rmse_val"),
            "multi_horizon_scores": best_attrs.get("multi_horizon_scores"),
            "rolling_origin_scores": best_attrs.get("rolling_origin_scores"),
            "long_horizon_stability_ratio": best_attrs.get("long_horizon_stability_ratio"),
            "long_horizon_expected_cases": best_attrs.get("long_horizon_expected_cases"),
            "long_horizon_predicted_cases": best_attrs.get("long_horizon_predicted_cases"),
            "long_horizon_reference_rate": best_attrs.get("long_horizon_reference_rate"),
            "long_horizon_collapse": best_attrs.get("long_horizon_collapse"),
        }

        near_zero_ratio = len(near_zero_trials) / max(len(valid_trials), 1)
        health_warning = bool(
            near_zero_ratio > 0.60
            or float(best_attrs.get("week_hour_distance", 1.0)) > 0.60
            or bool(best_attrs.get("long_horizon_collapse", False))
        )
        print(
            f"\n[联合目标优化结果] 剪枝={len(pruned_trials)}, "
            f"完成={len(complete_trials)}, 有效={len(valid_trials)}, "
            f"near-zero诊断={len(near_zero_trials)}"
        )
        print(
            f"最终选择 Trial {selected_trial.number}, "
            f"软联合目标={selected_score:.6f}"
        )
        print(
            "[Arrival Model Health] "
            f"near_zero={len(near_zero_trials)}/{len(valid_trials)} "
            f"({near_zero_ratio:.1%}), "
            f"best_week_hour={float(best_attrs.get('week_hour_distance', float('nan'))):.6f}, "
            f"status={'WARNING' if health_warning else 'OK'}"
        )
        print(
            "最优目标分量: "
            f"TimeCDF={best_attrs.get('time_cdf_distance', float('nan')):.6f}, "
            f"WeekHour={best_attrs.get('week_hour_distance', float('nan')):.6f}, "
            f"CountBias={best_attrs.get('count_bias', float('nan')):.6f}, "
            f"Concentration={best_attrs.get('concentration_error', float('nan')):.6f}, "
            f"WAPE(diag)={best_attrs.get('hourly_wape', float('nan')):.6f}, "
            f"CountRatio={best_attrs.get('count_ratio', float('nan')):.4f}"
        )
        print(
            "[最优模型多时间尺度诊断] "
            + "; ".join(
                (
                    f"{item['label']}:"
                    f"J={item['joint_score']:.4f},"
                    f"CDF={item['time_cdf_distance']:.4f},"
                    f"WeekHour={item['week_hour_distance']:.4f},"
                    f"CountRatio={item['predicted_total'] / max(item['actual_total'], 1e-12):.4f}"
                )
                for item in best_attrs.get("multi_horizon_scores", [])
            )
        )
        print(
            "[最优模型长期稳定性-仅诊断] "
            f"ratio={float(best_attrs.get('long_horizon_stability_ratio', float('nan'))):.4f}, "
            f"expected={float(best_attrs.get('long_horizon_expected_cases', float('nan'))):.2f}, "
            f"predicted={float(best_attrs.get('long_horizon_predicted_cases', float('nan'))):.2f}, "
            f"collapse={bool(best_attrs.get('long_horizon_collapse', False))}"
        )
        print(f"Best Params: {selected_trial.params}")
        print("使用软联合目标最佳参数在全量数据上重新训练...")

        best_params = dict(selected_trial.params)
        if int(best_params.get("n_lags", 0)) <= 0:
            best_params["ar_reg"] = 0.0
        self.model_metadata["best_params"] = best_params
        final_epochs = int(
            selected_trial.user_attrs.get("best_epoch", epochs)
        )
        final_epochs = max(10, min(final_epochs, epochs))
        self.model_metadata["best_epoch"] = final_epochs
        print(f"[全量重训] 使用验证最优轮数: {final_epochs}")

        simple_calibration = self._fit_simple_generation_calibration(
            df_train=df_train,
            best_params=best_params,
            final_epochs=final_epochs,
            batch_size=batch_size,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            validation_plan=multi_horizon_plan,
            rolling_plan=rolling_plan,
            objective_weights=objective_weights,
            config=config,
            cap_multiplier=cap_multiplier,
        )
        self.model_metadata["simple_generation_calibration_version"] = (
            "simple_rate_calendar_v2_rolling"
        )
        self.model_metadata["simple_generation_calibration"] = simple_calibration

        self._set_training_seed(int(self.parms.get("arrival_training_seed", 42)))

        final_m = NeuralProphet(
            **best_params,
            quantiles=[0.1, 0.9],
            global_normalization=True,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            growth="linear",
            batch_size=batch_size
        )

        final_m.fit(
            df_train,
            freq="h",
            progress="none",
            epochs=final_epochs
        )

        base_name = self.parms["file"].split(".")[0]
        temp_model_path = os.path.join(self.temp_output, f"{base_name}_nprf.np")
        torch.save(final_m, temp_model_path)

        return {"loss": float(selected_score)}

    def _save_model(self, metadata_file, acc):
        self.model_metadata["loss"] = acc["loss"]
        self.model_metadata["generated_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.model_metadata["max_cap"] = self.max_cap

        base_name = self.parms["file"].split(".")[0]
        dest_path = os.path.join(self.parms["ia_gen_path"], f"{base_name}_nprf.np")
        source_path = os.path.join(self.temp_output, f"{base_name}_nprf.np")

        shutil.copyfile(source_path, dest_path)
        sup.create_json(self.model_metadata, metadata_file)
        shutil.rmtree(self.temp_output, ignore_errors=True)

    # ==========================================================
    # 数据预处理
    # ==========================================================
    @staticmethod
    def _transform_features(df_train, cap_multiplier=1.5):
        df_train = df_train.copy()
        if "task" in df_train.columns:
            df_train = df_train[
                ~df_train["task"].isin(["Start", "End", "start", "end"])
            ].copy()
        if df_train.empty:
            raise ValueError("No real events available for arrival model training.")

        df_train = df_train.groupby(La.CASE_ID).start_timestamp.min().reset_index()
        df_train = df_train.groupby(
            [pd.Grouper(key=La.START_TIME, freq="h")]
        ).size().reset_index(name="count")
        df_train.rename(columns={La.START_TIME: "ds", "count": "y"}, inplace=True)

        # 显式补齐首个 arrival 到最后一个 arrival 之间的无到达小时为 0。
        df_train = df_train.set_index("ds").asfreq("h").fillna(0).reset_index()
        df_train["y"] = df_train["y"].astype(np.float64)

        max_cap = NeuralProphetGenerator._calculate_max_cap(
            df_train,
            cap_multiplier=cap_multiplier,
        )

        if df_train["ds"].dt.tz is not None:
            df_train["ds"] = df_train["ds"].dt.tz_localize(None)

        return df_train, max_cap

    @staticmethod
    def _calculate_max_cap(df_train, cap_multiplier=1.5):
        """根据当前训练折计算预测上限，避免验证尾段信息进入 cap。"""
        multiplier = float(cap_multiplier)
        if (not np.isfinite(multiplier)) or multiplier <= 0.0:
            multiplier = 1.5
        y = pd.to_numeric(df_train["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
        historical_max = float(y.max()) if len(y) else 0.0
        return max(historical_max * multiplier, 1.0)

    @staticmethod
    def _safe_autocorr(x, lag):
        x = np.asarray(x, dtype=float)
        if len(x) <= lag or lag <= 0:
            return 0.0
        x1 = x[:-lag]
        x2 = x[lag:]
        if np.std(x1) < 1e-8 or np.std(x2) < 1e-8:
            return 0.0
        return float(np.corrcoef(x1, x2)[0, 1])

    @staticmethod
    def _safe_clip(x, low, high):
        return max(low, min(high, x))

    def _resolve_adaptive_seasonality(self, profile, daily_default, weekly_default):
        """
        根据训练 arrival 序列的统计画像，自适应决定 daily / weekly seasonality 强度。
        原则：
        1. 稀疏、弱周期、强突发 -> 弱化甚至关闭 seasonality
        2. 密集、稳定、强周期 -> 保留 seasonality
        """
        nonzero_ratio = float(profile.get("nonzero_ratio", 0.0))
        sparsity = float(profile.get("sparsity", 1.0))
        daily_strength = float(profile.get("daily_strength", 0.0))
        weekly_strength = float(profile.get("weekly_strength", 0.0))
        burst_ratio = float(profile.get("burst_ratio", 1.0))

        # 到达越密集，越允许显式 seasonality
        density_score = self._safe_clip((nonzero_ratio - 0.05) / 0.45, 0.0, 1.0)

        # 周期强度分数
        daily_score = self._safe_clip((daily_strength - 0.06) / 0.22, 0.0, 1.0)
        weekly_score = self._safe_clip((weekly_strength - 0.05) / 0.20, 0.0, 1.0)

        # 稀疏/突发会削弱“可学习周期”
        burst_penalty = self._safe_clip((burst_ratio - 1.5) / 2.5, 0.0, 1.0)

        daily_score = daily_score * (0.55 + 0.45 * density_score) * (1.0 - 0.45 * sparsity) * (
                    1.0 - 0.35 * burst_penalty)
        weekly_score = weekly_score * (0.60 + 0.40 * density_score) * (1.0 - 0.30 * sparsity)

        daily_score = self._safe_clip(daily_score, 0.0, 1.0)
        weekly_score = self._safe_clip(weekly_score, 0.0, 1.0)

        def _resolve_one(default_value, score, min_order=2):
            if default_value in [False, None, 0]:
                return False
            if score < 0.20:
                return False
            return int(max(min_order, round(float(default_value) * score)))

        daily_seasonality = _resolve_one(daily_default, daily_score, min_order=2)
        weekly_seasonality = _resolve_one(weekly_default, weekly_score, min_order=2)

        # 周期越弱，seasonality_reg 越应偏大，避免拟合出伪周期
        seasonality_reg_scale = 1.0 + 1.2 * (1.0 - max(daily_score, weekly_score))

        return daily_seasonality, weekly_seasonality, float(seasonality_reg_scale)

    def _build_generation_profile(self, df_history, num_instances, total_cases):
        """
        根据当前历史 arrival 序列构造生成阶段统计画像。
        设计目标：
        1. 不依赖数据集名字，保持通用性
        2. 对极稀疏序列，避免 burst_ratio 被大量 0 掩盖
        3. 同时刻画：
           - 稀疏性
           - 波动性
           - 突发性
           - 日/周周期强度
           - 非零小时上的局部强度
        """
        y = df_history["y"].astype(float).to_numpy()
        hist_hours = max(len(df_history), 24)

        mean_rate = float(np.mean(y)) if len(y) > 0 else 0.0
        std_rate = float(np.std(y)) if len(y) > 0 else 0.0
        cv_rate = std_rate / max(mean_rate, 1e-6)

        nonzero_mask = (y > 0)
        nonzero_ratio = float(np.mean(nonzero_mask)) if len(y) > 0 else 0.0
        sparsity = 1.0 - nonzero_ratio

        positive_y = y[nonzero_mask]
        positive_mean = float(np.mean(positive_y)) if len(positive_y) > 0 else 0.0
        positive_std = float(np.std(positive_y)) if len(positive_y) > 1 else 0.0

        # -------------------------
        # 更稳健的 burst_ratio
        # 原实现用整体 p95，在极稀疏序列上容易因为大量 0 直接变成 0
        # 这里综合整体高分位与非零部分高分位，避免被 0 掩盖
        # -------------------------
        p95_all = float(np.percentile(y, 95)) if len(y) > 0 else 0.0
        p99_all = float(np.percentile(y, 99)) if len(y) > 0 else 0.0

        p80_pos = float(np.percentile(positive_y, 80)) if len(positive_y) > 0 else 0.0
        p95_pos = float(np.percentile(positive_y, 95)) if len(positive_y) > 0 else 0.0

        # 同时考虑整体尾部和非零小时尾部
        burst_ref = max(p99_all, p95_pos, p80_pos)
        burst_ratio = burst_ref / max(mean_rate, 1e-6)

        # 非零小时上的离散程度
        pos_cv = positive_std / max(positive_mean, 1e-6) if len(positive_y) > 1 else 0.0

        # -------------------------
        # 周期性强度
        # 稀疏序列上，直接自相关会很不稳定，因此仍保留安全计算
        # -------------------------
        daily_strength = self._safe_autocorr(y, 24) if hist_hours >= 24 * 3 else 0.0
        weekly_strength = self._safe_autocorr(y, 24 * 7) if hist_hours >= 24 * 14 else 0.0

        # 只保留正相关部分，更符合“可利用周期性”的含义
        daily_strength = float(max(daily_strength, 0.0))
        weekly_strength = float(max(weekly_strength, 0.0))

        case_ratio = float(num_instances) / max(float(total_cases), 1.0)

        # 额外增加一个“零块长度”特征：连续零段平均长度
        zero_run_mean = 0.0
        if len(y) > 0:
            zero_runs = []
            run_len = 0
            for val in y:
                if val <= 0:
                    run_len += 1
                else:
                    if run_len > 0:
                        zero_runs.append(run_len)
                    run_len = 0
            if run_len > 0:
                zero_runs.append(run_len)
            if len(zero_runs) > 0:
                zero_run_mean = float(np.mean(zero_runs))

        profile = {
            "hist_hours": int(hist_hours),

            # 整体强度
            "mean_rate": float(mean_rate),
            "std_rate": float(std_rate),
            "cv_rate": float(cv_rate),

            # 稀疏性
            "nonzero_ratio": float(nonzero_ratio),
            "sparsity": float(sparsity),
            "zero_run_mean": float(zero_run_mean),

            # 非零小时局部统计
            "positive_mean": float(positive_mean),
            "positive_std": float(positive_std),
            "pos_cv": float(pos_cv),

            # 突发性
            "p95_all": float(p95_all),
            "p99_all": float(p99_all),
            "p80_pos": float(p80_pos),
            "p95_pos": float(p95_pos),
            "burst_ratio": float(burst_ratio),

            # 周期性
            "daily_strength": float(daily_strength),
            "weekly_strength": float(weekly_strength),

            # 目标生成比例
            "case_ratio": float(case_ratio),
        }
        return profile
    def _set_generation_seed(self, seed=42):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    # ==========================================================
    # 生成主入口
    # ==========================================================
    def generate(self, num_instances, start_time):
        """
        按时间向前生成案例到达。

        生成阶段统一采用“全局 exact-N + floor + weighted residual”策略：
        1. NeuralProphet 只负责给出每小时连续到达强度；
        2. 连续按块外推，但不再逐小时 stochastic_round，也不在块内生成案例；
        3. 当累计预测强度达到目标案例数时，在强度层面截到恰好需要的质量，
           不再“整块生成 -> 超量 -> 按时间截断案例”；
        4. 对完整预测窗口一次性做全局整数分配，严格保证最终案例数等于
           num_instances；
        5. 若 NeuralProphet 强度不足，则先追加训练日志驱动的星期-小时 fallback
           强度，再与正常预测一起参与同一次全局 exact-N 分配。
        """
        try:
            target_n = int(num_instances)
            if target_n <= 0:
                return pd.DataFrame(columns=[La.CASE_ID, La.TIMESTAMP])

            base_seed = int(self.parms.get("seed", 42))
            seed = base_seed + self._generation_call
            self._generation_call += 1
            self._set_generation_seed(seed)
            rng = np.random.default_rng(seed)

            requested_start = self._standardize_start_time(start_time)
            print(
                f"[时间同步] 标准化起始时间: {requested_start} "
                f"(无时区, generation_seed={seed})"
            )

            m = self._load_torch_model()
            saved_metadata = self._load_saved_metadata()

            # 加载训练阶段得到的两个简化生成参数：rate_scale + calendar blend。
            simple_calibration = saved_metadata.get(
                "simple_generation_calibration", {}
            )
            rate_scale = float(simple_calibration.get("rate_scale", 1.0))
            if not np.isfinite(rate_scale) or rate_scale <= 0.0:
                rate_scale = 1.0
            calendar_blend_alpha = float(
                simple_calibration.get("calendar_blend_alpha", 1.0)
            )
            calendar_blend_alpha = float(np.clip(calendar_blend_alpha, 0.0, 1.0))
            calendar_smoothing = float(np.clip(
                simple_calibration.get("calendar_smoothing", 0.50),
                0.0,
                1.0,
            ))

            cap_multiplier = float(saved_metadata.get("cap_multiplier", 1.5))
            df_history, recomputed_cap = self._transform_features(
                self.log,
                cap_multiplier=cap_multiplier,
            )
            saved_cap = saved_metadata.get("max_cap", recomputed_cap)
            try:
                saved_cap = float(saved_cap)
            except Exception:
                saved_cap = recomputed_cap
            self.max_cap = saved_cap if np.isfinite(saved_cap) and saved_cap > 0 else recomputed_cap
            if df_history["ds"].dt.tz is not None:
                df_history["ds"] = df_history["ds"].dt.tz_localize(None)

            total_cases = self.log[La.CASE_ID].nunique()
            profile = self._build_generation_profile(
                df_history=df_history,
                num_instances=target_n,
                total_cases=total_cases
            )

            print("[生成阶段统计画像]")
            print(f"hist_hours={profile['hist_hours']}, "
                  f"mean_rate={profile['mean_rate']:.4f}, "
                  f"sparsity={profile['sparsity']:.4f}, "
                  f"cv_rate={profile['cv_rate']:.4f}, "
                  f"daily_strength={profile['daily_strength']:.4f}, "
                  f"weekly_strength={profile['weekly_strength']:.4f}, "
                  f"burst_ratio={profile['burst_ratio']:.4f}")

            print("[强制设定] 为了兼容已训练模型，强制设定生成频率为: 1H")
            n_lags = int(getattr(m, "n_lags", 0))

            history_end = pd.to_datetime(df_history["ds"].max()).tz_localize(None) \
                if pd.Timestamp(df_history["ds"].max()).tzinfo is not None \
                else pd.to_datetime(df_history["ds"].max())

            forecast_start = max(
                requested_start.floor("h"),
                history_end + pd.Timedelta(hours=1)
            )
            if forecast_start > requested_start.floor("h"):
                print(
                    "[到达生成警告] 请求起点不晚于训练到达历史末端，"
                    f"实际预测从 {forecast_start} 开始。请检查训练/测试时间边界。"
                )

            block_hours = max(
                24,
                int(self.parms.get("arrival_forecast_block_hours", 24 * 14))
            )
            max_blocks = max(
                1,
                int(self.parms.get("arrival_max_forecast_blocks", 12))
            )
            max_zero_blocks = max(
                1,
                int(self.parms.get("arrival_max_zero_blocks", 2))
            )

            # 这里只决定“连续强度”的扰动方式；最终整数案例数统一由
            # _allocate_exact_n_counts() 做全局 exact-N 分配。
            configured_sampling_method = self.parms.get(
                "arrival_count_sampling",
                None
            )
            if configured_sampling_method is None:
                intensity_sampling_method = self._select_arrival_sampling_method(
                    profile
                )
            else:
                intensity_sampling_method = str(
                    configured_sampling_method
                ).strip().lower()

            # 向后兼容旧参数：stochastic_round / unbiased_round 不再逐小时取整，
            # 统一解释为使用连续均值，然后进入全局 exact-N。
            if intensity_sampling_method in {
                    "stochastic_round", "unbiased_round"}:
                print(
                    "[Exact-N] 检测到旧 arrival_count_sampling="
                    f"{intensity_sampling_method}，已自动映射为 exact_n。"
                )
                intensity_sampling_method = "exact_n"

            recent_rate_days = max(
                1,
                int(self.parms.get("arrival_recent_rate_days", 60))
            )
            recent_hours = min(
                len(df_history),
                recent_rate_days * 24
            )
            recent_rate = float(
                pd.to_numeric(
                    df_history["y"].tail(recent_hours),
                    errors="coerce"
                ).fillna(0.0).mean()
            )
            if not np.isfinite(recent_rate) or recent_rate <= 0.0:
                recent_rate = max(float(profile["mean_rate"]), 1e-6)

            week_hour_scale = self._build_week_hour_scale(
                history=df_history,
                smoothing=calendar_smoothing,
            )
            fallback_calendar_strength = float(np.clip(
                self.parms.get("arrival_fallback_calendar_strength", 1.0),
                0.0,
                1.0,
            ))

            print(
                f"[生成策略] 连续时间块预测: n_lags={n_lags}, "
                f"block_hours={block_hours}, max_blocks={max_blocks}, "
                f"max_zero_blocks={max_zero_blocks}, "
                f"intensity_sampling={intensity_sampling_method}, "
                "allocation=global_exact_n_floor_weighted_residual"
            )
            print(
                "[简化生成校准] "
                f"rate_scale={rate_scale:.4f}, "
                f"calendar_blend_alpha={calendar_blend_alpha:.2f}, "
                f"calendar_smoothing={calendar_smoothing:.2f}, "
                f"source={simple_calibration.get('source', 'legacy_or_missing')}"
            )
            if not simple_calibration:
                print(
                    "[简化生成校准] 旧模型无新元数据，本次安全退化为 "
                    "rate_scale=1, alpha=1；建议 update_ia_gen=True 重训。"
                )
            print(
                "[自适应采样画像] "
                f"sparsity={profile['sparsity']:.4f}, "
                f"mean_rate={profile['mean_rate']:.4f}, "
                f"cv_rate={profile['cv_rate']:.4f}, "
                f"burst_ratio={profile['burst_ratio']:.4f}"
            )
            print(
                f"[fallback 画像] recent_rate_days={recent_rate_days}, "
                f"recent_hours={recent_hours}, "
                f"recent_rate={recent_rate:.6f} cases/hour, "
                f"calendar_strength={fallback_calendar_strength:.2f}"
            )

            intensity_frames = []
            cumulative_mass = 0.0
            cursor = forecast_start
            context_df = df_history[["ds", "y"]].sort_values("ds").copy()
            zero_blocks = 0
            fallback_reason = None
            last_positive_hour = None
            eps = 1e-10

            for block_idx in range(max_blocks):
                block_end = cursor + pd.Timedelta(hours=block_hours - 1)

                if n_lags <= 0:
                    pred_df = self._predict_density_no_lags(
                        m=m,
                        forecast_start=cursor,
                        horizon_end=block_end,
                        freq="1h",
                        max_cap=self.max_cap,
                    )
                else:
                    pred_df, context_df = self._predict_lagged_block(
                        m=m,
                        context_df=context_df,
                        block_start=cursor,
                        block_end=block_end,
                        profile=profile,
                        max_cap=self.max_cap,
                    )

                if pred_df.empty:
                    zero_blocks += 1
                    print(
                        f"[到达生成警告] 第 {block_idx + 1} 个预测块为空: "
                        f"{cursor} -- {block_end}"
                    )
                    block_mass = 0.0
                else:
                    pred_df = self._apply_rate_calendar_blend(
                        pred_df=pred_df,
                        week_hour_scale=week_hour_scale,
                        calendar_blend_alpha=calendar_blend_alpha,
                        rate_scale=rate_scale,
                    )

                    block_intensity = self._sample_hourly_intensities(
                        pred_df=pred_df,
                        rng=rng,
                        method=intensity_sampling_method
                    )
                    block_intensity = self._apply_generation_start_boundary(
                        intensity_df=block_intensity,
                        requested_start=requested_start
                    )

                    block_mass = float(
                        pd.to_numeric(
                            block_intensity.get("gen_intensity", pd.Series(dtype=float)),
                            errors="coerce"
                        ).fillna(0.0).clip(lower=0.0).sum()
                    )

                    if block_mass <= eps:
                        zero_blocks += 1
                    else:
                        remaining_mass = max(
                            float(target_n) - cumulative_mass,
                            0.0
                        )

                        # 不再让最后一个 block 的强度整体越过目标后再生成并截断。
                        # 如果本块会越界，只保留按时间顺序达到 remaining_mass 所需
                        # 的强度；最后一个小时必要时只保留部分强度。
                        if block_mass > remaining_mass + eps:
                            block_intensity = self._trim_intensity_to_mass(
                                intensity_df=block_intensity,
                                target_mass=remaining_mass
                            )
                            block_mass = float(
                                block_intensity["gen_intensity"].sum()
                            )

                        intensity_frames.append(block_intensity)
                        cumulative_mass += block_mass
                        zero_blocks = 0

                        positive = block_intensity[
                            block_intensity["gen_intensity"] > eps
                        ]
                        if not positive.empty:
                            last_positive_hour = pd.to_datetime(
                                positive["ds"].max()
                            )

                print(
                    f"[到达生成] block={block_idx + 1}, "
                    f"累计预测强度={cumulative_mass:.6f}/{target_n}, "
                    f"本块有效强度={block_mass:.6f}, "
                    f"区间={cursor} -- {block_end}"
                )

                if cumulative_mass >= float(target_n) - eps:
                    break

                if zero_blocks >= max_zero_blocks:
                    fallback_reason = (
                        f"连续 {zero_blocks} 个预测块没有有效到达强度"
                    )
                    print(
                        "[到达生成警告] "
                        f"{fallback_reason}，停止 NeuralProphet 外推；"
                        "立即切换到日历感知 fallback 强度。"
                    )
                    break

                cursor = block_end + pd.Timedelta(hours=1)

            # NeuralProphet 预测强度不足时，不再直接生成 missing 个随机案例。
            # 先构造“恰好补足缺失强度质量”的 fallback 小时强度，再和正常预测
            # 一起进入同一次全局 exact-N 分配。
            if cumulative_mass < float(target_n) - eps:
                missing_mass = float(target_n) - cumulative_mass
                if fallback_reason is None:
                    fallback_reason = "达到最大预测范围但累计预测强度仍不足"

                print(
                    f"[到达生成警告] {fallback_reason}，"
                    f"缺少强度质量 {missing_mass:.6f}；"
                    "使用近期到达率与训练日志星期-小时分布构造 fallback 强度。"
                )

                if last_positive_hour is not None:
                    fallback_start = (
                        pd.to_datetime(last_positive_hour)
                        + pd.Timedelta(hours=1)
                    )
                else:
                    fallback_start = max(
                        pd.to_datetime(requested_start),
                        pd.to_datetime(forecast_start)
                    )

                fallback_intensity = self._build_calendar_fallback_intensity(
                    start=fallback_start,
                    required_mass=missing_mass,
                    mean_rate_per_hour=recent_rate,
                    history=df_history,
                    smoothing=calendar_smoothing,
                    strength=fallback_calendar_strength
                )
                intensity_frames.append(fallback_intensity)
                cumulative_mass += float(
                    fallback_intensity["gen_intensity"].sum()
                )

            if not intensity_frames:
                raise RuntimeError(
                    "No valid arrival intensity was produced for exact-N allocation."
                )

            global_intensity = pd.concat(
                intensity_frames,
                ignore_index=True,
                sort=False
            )
            global_intensity = (
                global_intensity
                .sort_values("ds")
                .reset_index(drop=True)
            )

            hourly_counts = self._allocate_exact_n_counts(
                intensity_df=global_intensity,
                target_n=target_n,
                rng=rng,
                profile=profile
            )
            timestamps = self._expand_hourly_counts(
                hourly_counts=hourly_counts,
                rng=rng
            )
            timestamps = sorted(pd.to_datetime(timestamps))

            # exact-N 是硬不变量；这里不再使用 [:num_instances] 静默截断。
            if len(timestamps) != target_n:
                raise RuntimeError(
                    "Global exact-N allocation failed: "
                    f"expected {target_n} cases, got {len(timestamps)}."
                )
            if timestamps and pd.to_datetime(timestamps[0]) < requested_start:
                raise RuntimeError(
                    "Generated arrival timestamp is earlier than requested_start."
                )

            print(
                "[Exact-N 完成] "
                f"target={target_n}, generated={len(timestamps)}, "
                f"global_intensity_mass={global_intensity['gen_intensity'].sum():.6f}"
            )

            # 可选的正式 test-only 诊断，默认关闭。
            # 若显式开启，也只在 timestamps 完成后读取 test，不回写任何模型参数。
            # 注意：人工根据该 test 指标继续调模型仍属于测试集调参，应避免。
            self._diagnose_arrival_only_day_hour_emd(
                generated_timestamps=timestamps
            )

            result = pd.DataFrame({La.TIMESTAMP: timestamps})
            result[La.CASE_ID] = [
                f"Case{i + 1}" for i in range(len(result))
            ]
            self.times = result[[La.CASE_ID, La.TIMESTAMP]].copy()
            return self.times

        except Exception as e:
            print(f"[NeuralProphet 错误] 生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=[La.CASE_ID, La.TIMESTAMP])

    # ==========================================================
    # 生成辅助函数
    # ==========================================================
    @staticmethod
    def _standardize_start_time(start_time):
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        else:
            start_time = pd.to_datetime(start_time)

        if start_time.tzinfo is not None:
            start_time = start_time.tz_localize(None)

        return start_time

    def _load_torch_model(self):
        try:
            m = torch.load(
                self.model_path,
                map_location=torch.device("cpu"),
                weights_only=False
            )
        except TypeError:
            # 兼容尚不支持 weights_only 参数的旧版 PyTorch。
            m = torch.load(
                self.model_path,
                map_location=torch.device("cpu")
            )

        import pytorch_lightning as pl
        m.trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False
        )
        m.model.to("cpu")
        return m

    def _load_saved_metadata(self):
        """读取与当前到达模型绑定的训练元数据。"""
        base_name = self.parms["file"].split(".")[0]
        metadata_file = os.path.join(
            self.parms["ia_gen_path"],
            f"{base_name}_nprf_meta{Fe.JSON}",
        )
        if not os.path.exists(metadata_file):
            return {}
        with open(metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def _safe_predict(self, model, df):
        with open(os.devnull, "w") as f:
            with redirect_stdout(f):
                forecast = model.predict(df)
        return forecast

    @staticmethod
    def _quantile_columns(pred_df):
        """
        兼容不同 NeuralProphet 版本的分位数列命名，例如：
        yhat1 10.0%、yhat1 90.0% 或 yhat1_q0.1。
        """
        candidates = []
        for col in pred_df.columns:
            text = str(col)
            if text == "yhat1" or not text.startswith("yhat1"):
                continue

            percent_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
            q_match = re.search(r"(?:_q|quantile[_ ]?)(0(?:\.[0-9]+)?|1(?:\.0+)?)", text, re.I)

            if percent_match:
                q = float(percent_match.group(1)) / 100.0
            elif q_match:
                q = float(q_match.group(1))
            else:
                continue
            candidates.append((q, col))

        candidates.sort(key=lambda x: x[0])
        if len(candidates) >= 2:
            return candidates[0][1], candidates[-1][1]
        return None, None

    @staticmethod
    def _build_week_hour_scale(history, smoothing=0.50):
        """
        仅使用训练日志构建 168 个星期-小时到达强度系数。

        返回系数的均值为 1。smoothing 将稀疏位置向全局均值收缩，
        防止少数训练案例造成日历画像过拟合。
        """
        hist = history[["ds", "y"]].copy()
        hist["ds"] = pd.to_datetime(hist["ds"])
        if hist["ds"].dt.tz is not None:
            hist["ds"] = hist["ds"].dt.tz_localize(None)
        hist["y"] = pd.to_numeric(
            hist["y"],
            errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        hist["weekday"] = hist["ds"].dt.dayofweek
        hist["hour"] = hist["ds"].dt.hour

        full_index = pd.MultiIndex.from_product(
            [range(7), range(24)],
            names=["weekday", "hour"]
        )
        global_rate = float(hist["y"].mean())
        if not np.isfinite(global_rate) or global_rate <= 0.0:
            global_rate = 1e-6

        slot_rate = (
            hist.groupby(["weekday", "hour"])["y"]
            .mean()
            .reindex(full_index)
            .fillna(global_rate)
            .clip(lower=0.0)
        )
        smoothing = float(np.clip(smoothing, 0.0, 1.0))
        slot_rate = (
            (1.0 - smoothing) * slot_rate
            + smoothing * global_rate
        )
        calendar_mean = float(slot_rate.mean())
        if not np.isfinite(calendar_mean) or calendar_mean <= 0.0:
            return pd.Series(
                1.0,
                index=full_index,
                dtype=float
            )
        return slot_rate / calendar_mean

    def _select_arrival_sampling_method(self, profile):
        """
        根据训练日志 arrival profile 选择“连续强度”扰动方式。

        注意：本函数不再决定逐小时整数计数。无论返回何种方式，最终所有
        小时都统一进入 global exact-N + floor + weighted residual 分配。

        - 稀疏或稳定日志：exact_n
          直接使用校准后的连续均值作为小时权重，不增加逐小时舍入噪声。
        - 密集且明显突发：triangular_calibrated
          允许利用 NeuralProphet 预测区间扰动连续强度，但最终总案例数仍固定。
        """
        sparsity = float(profile.get("sparsity", 1.0))
        mean_rate = float(profile.get("mean_rate", 0.0))
        cv_rate = float(profile.get("cv_rate", 0.0))
        burst_ratio = float(profile.get("burst_ratio", 1.0))
        nonzero_ratio = float(profile.get("nonzero_ratio", 0.0))

        if (
            sparsity > 0.65
            or nonzero_ratio < 0.35
            or mean_rate < 0.4
        ):
            return "exact_n"

        if (
            mean_rate >= 1.0
            and (cv_rate > 0.8 or burst_ratio > 2.0)
        ):
            return "triangular_calibrated"

        return "exact_n"


    def _sample_hourly_intensities(
            self,
            pred_df,
            rng,
            method="exact_n"
    ):
        """
        把 NeuralProphet 输出转换为“连续小时到达强度”，不做逐小时整数化。

        exact_n / stochastic_round / unbiased_round:
            使用校准后的 yhat1 连续均值。后两者仅为旧参数兼容，不再执行
            Bernoulli stochastic rounding。

        triangular / triangular_calibrated:
            在 NeuralProphet [lower, mean, upper] 内采样连续强度，用于保留模型的
            不确定性；最终整数案例数仍由全局 exact-N 分配。

        poisson:
            为兼容旧运行参数，允许 Poisson 仅作为“小时权重扰动”；采样后的
            总数不会直接作为案例数，最终仍重新归一化并 exact-N。
        """
        if pred_df is None or pred_df.empty:
            return pd.DataFrame(
                columns=["ds", "gen_intensity", "lower_second"]
            )

        result = pred_df[["ds", "yhat1"]].copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)

        mean = (
            pd.to_numeric(
                pred_df["yhat1"],
                errors="coerce"
            )
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(float)
        )
        lower = (
            pd.to_numeric(
                pred_df.get("yhat_lower", pred_df["yhat1"] * 0.8),
                errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
        )
        upper = (
            pd.to_numeric(
                pred_df.get("yhat_upper", pred_df["yhat1"] * 1.2),
                errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
        )

        lower = np.maximum(lower, 0.0)
        mean = np.maximum(mean, 0.0)
        upper = np.maximum(upper, mean)

        # 修正极端预测区间，确保 triangular 的 left <= mode <= right。
        lower = np.minimum(np.maximum(lower, 0.0), mean)
        mean = np.maximum(mean, 0.0)
        upper = np.maximum(upper, mean)

        method = str(method).strip().lower()
        if method in {
                "exact_n", "mean", "stochastic_round", "unbiased_round"}:
            intensity = mean.astype(float, copy=True)

        elif method == "poisson":
            intensity = rng.poisson(
                np.maximum(mean, 0.0)
            ).astype(float)
            # 极稀疏窗口可能整块抽成 0；此时退回连续均值，避免错误触发 fallback。
            if float(intensity.sum()) <= 0.0 and float(mean.sum()) > 0.0:
                intensity = mean.astype(float, copy=True)

        elif method in {"triangular", "triangular_calibrated"}:
            intensity = mean.astype(float, copy=True)
            variable = upper > lower + 1e-12
            if np.any(variable):
                intensity[variable] = rng.triangular(
                    lower[variable],
                    mean[variable],
                    upper[variable]
                )
        else:
            raise ValueError(
                "arrival_count_sampling must be one of: "
                "exact_n, triangular_calibrated, triangular, poisson "
                "(stochastic_round/unbiased_round are accepted as legacy aliases)."
            )

        intensity = np.nan_to_num(
            intensity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        intensity = np.clip(intensity, 0.0, None)

        result["gen_intensity"] = intensity
        result["lower_second"] = 0.0
        return result[["ds", "gen_intensity", "lower_second"]].reset_index(drop=True)


    @staticmethod
    def _apply_generation_start_boundary(intensity_df, requested_start):
        """
        处理生成起点位于小时内部的情况。

        旧实现先生成整小时案例，再把 requested_start 之前的时间戳过滤掉，
        会破坏 exact-N。现在直接在强度层按“该小时剩余比例”缩放，并记录
        lower_second，后续小时内采样不会产生早于 requested_start 的时间戳。
        """
        if intensity_df is None or intensity_df.empty:
            return pd.DataFrame(
                columns=["ds", "gen_intensity", "lower_second"]
            )

        result = intensity_df.copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)

        start = pd.to_datetime(requested_start)
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        start_hour = start.floor("h")

        result = result[result["ds"] >= start_hour].copy()
        if "lower_second" not in result.columns:
            result["lower_second"] = 0.0
        result["lower_second"] = pd.to_numeric(
            result["lower_second"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0, upper=3600.0)

        first_mask = result["ds"] == start_hour
        if first_mask.any():
            lower_second = float(
                np.clip((start - start_hour).total_seconds(), 0.0, 3600.0)
            )
            remaining_fraction = max(
                0.0,
                (3600.0 - lower_second) / 3600.0
            )
            result.loc[first_mask, "gen_intensity"] = (
                pd.to_numeric(
                    result.loc[first_mask, "gen_intensity"],
                    errors="coerce"
                ).fillna(0.0).clip(lower=0.0)
                * remaining_fraction
            )
            result.loc[first_mask, "lower_second"] = lower_second

        return result.reset_index(drop=True)


    @staticmethod
    def _trim_intensity_to_mass(intensity_df, target_mass):
        """
        按时间顺序把连续强度裁到 target_mass。

        这是对旧“最后一个 block 整体超量后，再从生成案例中 [:N] 截断”的替代。
        若目标质量落在某一个小时内部，只缩小该小时的强度，不删除已经生成的
        案例，因为此时尚未做任何整数采样。
        """
        target_mass = float(target_mass)
        if target_mass <= 0.0 or intensity_df is None or intensity_df.empty:
            return pd.DataFrame(
                columns=["ds", "gen_intensity", "lower_second"]
            )

        result = intensity_df.copy()
        result["gen_intensity"] = (
            pd.to_numeric(result["gen_intensity"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        result = result.sort_values("ds").reset_index(drop=True)

        total_mass = float(result["gen_intensity"].sum())
        if total_mass <= target_mass + 1e-12:
            return result

        cumulative = result["gen_intensity"].cumsum().to_numpy(dtype=float)
        crossing = int(np.searchsorted(cumulative, target_mass, side="left"))
        crossing = min(max(crossing, 0), len(result) - 1)

        trimmed = result.iloc[:crossing + 1].copy()
        previous_mass = (
            float(cumulative[crossing - 1]) if crossing > 0 else 0.0
        )
        final_mass = max(target_mass - previous_mass, 0.0)
        trimmed.loc[trimmed.index[-1], "gen_intensity"] = final_mass

        return trimmed.reset_index(drop=True)


    @staticmethod
    def _allocate_exact_n_counts(
            intensity_df,
            target_n,
            rng,
            profile=None):
        """
        对完整预测窗口执行“六日志统一”的 global exact-N 整数分配。

        基础部分：
            p_t = lambda_t / sum(lambda)
            e_t = N * p_t
            base_t = floor(e_t)
            R = N - sum(base_t)

        关键改动：残差不再一律使用 replace=False。

        residual_share = R / N 反映本次 exact-N 中有多少案例需要靠残差完成；
        training profile 中的 sparsity / CV / burst_ratio 反映历史到达过程是否
        稀疏且突发。两者共同决定 replacement_fraction：

        - residual_share 很低：绝大多数案例已由 floor 固定，残差采用低方差的
          不放回加权分配；适合 BPI17W/BPI12W 等大规模、较密集日志。
        - residual_share 很高且历史明显突发：允许一部分甚至全部残差进行
          有放回加权抽样，使同一小时可以获得多个案例；避免 MP/P2P 等稀疏
          日志在 floor_sum≈0 时被强行摊成“每小时最多 1 个案例”。
        - 中间区域：自动混合两种分配方式。

        整个协议不读取日志名称，最终始终严格满足 sum(n_t) == N。
        """
        target_n = int(target_n)
        if target_n <= 0:
            return pd.DataFrame(
                columns=["ds", "gen_round", "lower_second"]
            )
        if intensity_df is None or intensity_df.empty:
            raise ValueError("Exact-N allocation received empty intensity data.")

        result = intensity_df.copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)
        result["gen_intensity"] = (
            pd.to_numeric(result["gen_intensity"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        if "lower_second" not in result.columns:
            result["lower_second"] = 0.0
        result["lower_second"] = (
            pd.to_numeric(result["lower_second"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0, upper=3600.0)
        )
        result = result.sort_values("ds").reset_index(drop=True)

        weights = result["gen_intensity"].to_numpy(dtype=float)
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError(
                "Exact-N allocation requires positive finite arrival intensity."
            )

        probabilities = weights / total_weight
        expected = probabilities * float(target_n)
        counts = np.floor(expected).astype(np.int64)
        fractions = expected - counts
        floor_sum = int(counts.sum())
        remainder = int(target_n - floor_sum)

        if remainder < 0:
            raise RuntimeError(
                "Exact-N base allocation exceeded target count unexpectedly."
            )

        residual_share = float(remainder) / max(float(target_n), 1.0)
        floor_ratio = float(floor_sum) / max(float(target_n), 1.0)

        # ----------------------------------------------------------
        # 训练日志驱动的 burst score；没有 profile 时安全退化为 0。
        # 这里只决定“残差方差/可重复小时”的程度，不改变总案例数。
        # ----------------------------------------------------------
        profile = profile if isinstance(profile, dict) else {}
        sparsity = float(np.clip(profile.get("sparsity", 0.0), 0.0, 1.0))
        cv_rate = max(float(profile.get("cv_rate", 0.0)), 0.0)
        burst_ratio = max(float(profile.get("burst_ratio", 1.0)), 0.0)

        cv_score = float(np.clip((cv_rate - 1.0) / 2.0, 0.0, 1.0))
        burst_score_component = float(
            np.clip((burst_ratio - 1.5) / 4.0, 0.0, 1.0)
        )
        historical_burst_score = float(np.clip(
            0.35 * sparsity
            + 0.30 * cv_score
            + 0.35 * burst_score_component,
            0.0,
            1.0
        ))

        # residual_share <= 0.15：残差很少，保持低方差；
        # residual_share >= 0.70：残差主导，再由历史 burst 决定是否接近
        # multinomial-with-replacement。中间区域线性过渡。
        residual_pressure = float(np.clip(
            (residual_share - 0.15) / 0.55,
            0.0,
            1.0
        ))
        replacement_fraction = float(np.clip(
            residual_pressure * (0.35 + 0.65 * historical_burst_score),
            0.0,
            1.0
        ))

        replacement_draws = int(round(remainder * replacement_fraction))
        replacement_draws = min(max(replacement_draws, 0), remainder)
        unique_draws = remainder - replacement_draws

        if remainder > 0:
            eligible = np.flatnonzero(fractions > 1e-12)
            residual_weights = fractions.copy()
            residual_total = float(residual_weights.sum())

            if len(eligible) == 0 or residual_total <= 0.0 or not np.isfinite(residual_total):
                # 纯浮点极端情况：按 expected 最大位置确定性补足。
                order = np.argsort(expected)[::-1]
                for idx in order[:remainder]:
                    counts[int(idx)] += 1
                unique_draws = remainder
                replacement_draws = 0
                replacement_fraction = 0.0
            else:
                # A. 低方差部分：按小数残差不放回抽样。
                # 每个被选小时最多获得 1 个 stable residual case。
                if unique_draws > 0:
                    unique_draws = min(unique_draws, len(eligible))
                    stable_probs = residual_weights[eligible]
                    stable_probs = stable_probs / stable_probs.sum()
                    chosen_unique = rng.choice(
                        eligible,
                        size=unique_draws,
                        replace=False,
                        p=stable_probs
                    )
                    counts[np.asarray(chosen_unique, dtype=int)] += 1

                # B. burst-preserving 部分：有放回抽样。
                # 当 floor_sum≈0 且 residual 主导时，允许一个小时被多次命中。
                actual_replacement_draws = remainder - unique_draws
                if actual_replacement_draws > 0:
                    replacement_probs = residual_weights / residual_total
                    replacement_additions = rng.multinomial(
                        n=actual_replacement_draws,
                        pvals=replacement_probs
                    )
                    counts += replacement_additions.astype(np.int64)
                    replacement_draws = actual_replacement_draws
                else:
                    replacement_draws = 0

        if int(counts.sum()) != target_n:
            raise RuntimeError(
                "Exact-N invariant violated after adaptive residual allocation: "
                f"target={target_n}, allocated={int(counts.sum())}."
            )

        result["gen_round"] = counts.astype(int)
        nonzero_counts = counts[counts > 0]
        max_hour_count = int(nonzero_counts.max()) if len(nonzero_counts) else 0
        multi_case_hours = int(np.sum(counts > 1))

        print(
            "[Exact-N 分配] "
            f"target={target_n}, intensity_mass={total_weight:.6f}, "
            f"floor_sum={floor_sum}, residual={remainder}"
        )
        print(
            "[Adaptive Residual] "
            f"floor_ratio={floor_ratio:.4f}, residual_share={residual_share:.4f}, "
            f"historical_burst_score={historical_burst_score:.4f}, "
            f"replacement_fraction={replacement_fraction:.4f}, "
            f"unique_draws={unique_draws}, replacement_draws={replacement_draws}, "
            f"multi_case_hours={multi_case_hours}, max_hour_count={max_hour_count}"
        )

        return result[
            result["gen_round"] > 0
        ][["ds", "gen_round", "lower_second"]].reset_index(drop=True)


    @staticmethod
    def _normalized_week_hour_emd(reference_timestamps, generated_timestamps):
        """
        仅用于 arrival-only 诊断的 7×24 weekday-hour EMD。

        将 weekday*24+hour 映射到 0..167，计算两个离散分布的一维
        Wasserstein/EMD，并除以 167 归一化到约 [0, 1]。
        该诊断只比较 case arrival，不使用后续 BPMN 活动时间。
        """
        ref = pd.to_datetime(pd.Series(reference_timestamps), errors="coerce").dropna()
        gen = pd.to_datetime(pd.Series(generated_timestamps), errors="coerce").dropna()
        if len(ref) == 0 or len(gen) == 0:
            return None

        def _to_hist(series):
            if getattr(series.dt, "tz", None) is not None:
                series = series.dt.tz_localize(None)
            bins = (series.dt.dayofweek * 24 + series.dt.hour).astype(int)
            hist = np.bincount(bins.to_numpy(), minlength=168).astype(float)
            total = float(hist.sum())
            return hist / total if total > 0 else hist

        p = _to_hist(ref)
        q = _to_hist(gen)
        # 对单位间隔有序离散分布，W1 = sum |CDF_p - CDF_q|。
        emd = float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]).sum())
        return emd / 167.0


    @staticmethod
    def _extract_arrivals_from_event_frame(frame):
        """从已加载的事件 DataFrame 中提取每个 case 的首个真实活动开始时间。"""
        if frame is None:
            return None
        try:
            df = pd.DataFrame(frame).copy()
        except Exception:
            return None
        if df.empty:
            return None

        # 尽量兼容项目内部标准列名以及常见原始列名。
        case_candidates = [
            La.CASE_ID, "caseid", "Case ID", "case:concept:name",
            "case_id", "case"
        ]
        time_candidates = [
            "start_timestamp", La.START_TIME, La.TIMESTAMP,
            "time:timestamp", "timestamp", "end_timestamp"
        ]
        task_candidates = ["task", "Activity", "concept:name", "activity"]

        case_col = next((c for c in case_candidates if c in df.columns), None)
        time_col = next((c for c in time_candidates if c in df.columns), None)
        task_col = next((c for c in task_candidates if c in df.columns), None)
        if case_col is None or time_col is None:
            return None

        if task_col is not None:
            task_text = df[task_col].astype(str).str.strip().str.lower()
            df = df[~task_text.isin({"start", "end"})].copy()
        if df.empty:
            return None

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[case_col, time_col])
        if df.empty:
            return None

        arrivals = df.groupby(case_col, sort=False)[time_col].min()
        return arrivals.dt.tz_convert(None).tolist()


    def _reference_arrivals_from_valdn(self):
        """若 pipeline 把真实参考日志传入 valdn，则无副作用地提取 arrival。"""
        reference = getattr(self, "valdn", None)
        if reference is None:
            return None

        candidates = []
        if isinstance(reference, pd.DataFrame):
            candidates.append(reference)
        if hasattr(reference, "data"):
            try:
                candidates.append(reference.data)
            except Exception:
                pass
        if isinstance(reference, (list, tuple, dict)):
            candidates.append(reference)

        for candidate in candidates:
            arrivals = self._extract_arrivals_from_event_frame(candidate)
            if arrivals:
                return arrivals
        return None


    def _reference_arrivals_from_test_xes(self):
        """
        诊断兜底：读取标准 exp_data/<log>/log_test.xes 的 case arrivals。

        只在 exact-N timestamps 已经全部生成之后调用；解析结果绝不反馈给
        NeuralProphet、校准器或 residual allocation，因此不构成生成阶段调参。
        """
        try:
            import xml.etree.ElementTree as ET

            base_name = str(self.parms.get("file", "")).rsplit(".", 1)[0]
            event_logs_path = self.parms.get("event_logs_path", None)
            if not base_name or not event_logs_path:
                return None, None

            test_path = os.path.join(
                str(event_logs_path),
                "exp_data",
                base_name,
                "log_test.xes"
            )
            if not os.path.exists(test_path):
                return None, test_path

            arrivals = []
            # 逐 trace 解析，避免大型 BPI17W 测试日志一次性加载全部 XML。
            context = ET.iterparse(test_path, events=("end",))
            for _, elem in context:
                if not str(elem.tag).endswith("trace"):
                    continue

                start_like = []
                all_real = []
                for event in list(elem):
                    if not str(event.tag).endswith("event"):
                        continue
                    attrs = {}
                    for child in list(event):
                        tag = str(child.tag)
                        key = child.attrib.get("key")
                        value = child.attrib.get("value")
                        if key is not None:
                            attrs[key] = value

                    activity = str(attrs.get("concept:name", "")).strip()
                    if activity.lower() in {"start", "end"}:
                        continue
                    ts = attrs.get("time:timestamp")
                    if ts is None:
                        continue
                    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
                    if pd.isna(parsed):
                        continue
                    all_real.append(parsed)
                    lifecycle = str(
                        attrs.get("lifecycle:transition", "")
                    ).strip().lower()
                    if lifecycle == "start":
                        start_like.append(parsed)

                candidates = start_like if start_like else all_real
                if candidates:
                    arrivals.append(min(candidates).tz_convert(None))
                elem.clear()

            return arrivals if arrivals else None, test_path
        except Exception as error:
            print(
                "[Arrival-only day_hour_emd] test XES 诊断读取失败: "
                f"{error}"
            )
            return None, None


    def _diagnose_arrival_only_day_hour_emd(self, generated_timestamps):
        """
        可选的正式 test arrival-only 诊断。

        为避免“看着测试集指标继续调模型”的人工测试集泄露，默认关闭。
        只有显式设置 arrival_enable_test_diagnostic=True 时才读取 log_test.xes。
        该值不得参与训练、gamma/lag/超参数选择或自动回写任何配置。
        """
        enabled = bool(
            self.parms.get("arrival_enable_test_diagnostic", False)
        )
        if not enabled:
            print(
                "[Arrival-only day_hour_emd] disabled: "
                "arrival_enable_test_diagnostic=False；"
                "开发/调参阶段请保持关闭，最终一次性测试时再开启。"
            )
            return

        try:
            reference_arrivals, source_path = self._reference_arrivals_from_test_xes()
            if not reference_arrivals:
                print(
                    "[Arrival-only day_hour_emd] unavailable: "
                    f"未找到正式 log_test.xes, path={source_path}"
                )
                return

            real_n = len(reference_arrivals)
            generated_n = len(generated_timestamps)
            if real_n != generated_n:
                print(
                    "[Arrival-only day_hour_emd] invalid comparison: "
                    f"real_cases={real_n}, generated_cases={generated_n}; "
                    "不计算 EMD，避免不同案例规模造成误导。"
                )
                return

            value = self._normalized_week_hour_emd(
                reference_timestamps=reference_arrivals,
                generated_timestamps=generated_timestamps,
            )
            if value is None:
                print(
                    "[Arrival-only day_hour_emd] unavailable: "
                    "真实或模拟 arrival 为空。"
                )
                return

            print(
                "[Arrival-only day_hour_emd][TEST-ONLY] "
                f"value={value:.9f}, real_cases={real_n}, "
                f"generated_cases={generated_n}, source=log_test.xes, "
                "metric=normalized_emd_168_weekhour_bins, "
                f"path={source_path}; DO_NOT_TUNE_ON_THIS_VALUE"
            )
        except Exception as error:
            print(
                "[Arrival-only day_hour_emd] 诊断失败但不影响仿真: "
                f"{error}"
            )


    @staticmethod
    def _expand_hourly_counts(hourly_counts, rng):
        """在每个小时内部均匀生成时间戳，同时保持 exact-N 不变量。"""
        timestamps = []
        for row in hourly_counts.itertuples(index=False):
            n = int(row.gen_round)
            if n <= 0:
                continue

            base = pd.to_datetime(row.ds)
            lower_second = float(
                np.clip(getattr(row, "lower_second", 0.0), 0.0, 3600.0)
            )
            if lower_second >= 3600.0:
                continue

            offsets = rng.uniform(
                lower_second,
                3600.0,
                size=n
            )
            timestamps.extend(
                base + pd.to_timedelta(offsets, unit="s")
            )

        return sorted(pd.to_datetime(timestamps))


    @staticmethod
    def _build_calendar_fallback_intensity(
            start,
            required_mass,
            mean_rate_per_hour,
            history,
            smoothing=0.05,
            strength=1.0):
        """
        构造星期-小时感知的 fallback 连续强度，并在强度层恰好补足 required_mass。

        与旧实现不同，这里不逐小时 stochastic_round/Poisson，也不先生成案例。
        fallback 只负责给出后续小时的连续强度，最后仍由全局 exact-N 统一整数化。
        """
        required_mass = float(required_mass)
        if required_mass <= 0.0:
            return pd.DataFrame(
                columns=["ds", "gen_intensity", "lower_second"]
            )

        mean_rate_per_hour = max(float(mean_rate_per_hour), 1e-6)
        smoothing = float(np.clip(smoothing, 0.0, 1.0))
        strength = float(np.clip(strength, 0.0, 1.0))

        hist = history[["ds", "y"]].copy()
        hist["ds"] = pd.to_datetime(hist["ds"])
        if hist["ds"].dt.tz is not None:
            hist["ds"] = hist["ds"].dt.tz_localize(None)
        hist["y"] = pd.to_numeric(
            hist["y"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        hist["weekday"] = hist["ds"].dt.dayofweek
        hist["hour"] = hist["ds"].dt.hour

        full_index = pd.MultiIndex.from_product(
            [range(7), range(24)],
            names=["weekday", "hour"]
        )
        slot_mean = (
            hist.groupby(["weekday", "hour"])["y"]
            .mean()
            .reindex(full_index)
        )

        global_rate = float(hist["y"].mean())
        if not np.isfinite(global_rate) or global_rate <= 0.0:
            global_rate = mean_rate_per_hour

        slot_mean = slot_mean.fillna(global_rate).clip(lower=0.0)
        slot_intensity = (
            (1.0 - smoothing) * slot_mean
            + smoothing * global_rate
        )

        calendar_mean = float(slot_intensity.mean())
        if not np.isfinite(calendar_mean) or calendar_mean <= 0.0:
            slot_scale = pd.Series(
                1.0,
                index=full_index,
                dtype=float
            )
        else:
            slot_scale = slot_intensity / calendar_mean
        slot_scale = (
            (1.0 - strength)
            + strength * slot_scale
        )

        start = pd.to_datetime(start)
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        cursor = start.floor("h")

        expected_hours = int(np.ceil(required_mass / mean_rate_per_hour))
        max_hours = max(24 * 365 * 2, expected_hours * 20)

        rows = []
        accumulated = 0.0
        eps = 1e-12

        for _ in range(max_hours):
            if accumulated >= required_mass - eps:
                break

            key = (int(cursor.dayofweek), int(cursor.hour))
            hourly_lambda = max(
                mean_rate_per_hour * float(slot_scale.loc[key]),
                0.0
            )

            lower_second = 0.0
            if cursor == start.floor("h"):
                lower_second = float(
                    np.clip((start - cursor).total_seconds(), 0.0, 3600.0)
                )
                remaining_fraction = max(
                    0.0,
                    (3600.0 - lower_second) / 3600.0
                )
                hourly_lambda *= remaining_fraction

            remaining_mass = max(required_mass - accumulated, 0.0)
            used_lambda = min(hourly_lambda, remaining_mass)

            if used_lambda > 0.0:
                rows.append({
                    "ds": cursor,
                    "gen_intensity": float(used_lambda),
                    "lower_second": float(lower_second),
                })
                accumulated += float(used_lambda)

            cursor += pd.Timedelta(hours=1)

        if accumulated < required_mass - 1e-8:
            raise RuntimeError(
                "Calendar-aware fallback intensity did not reach required mass. "
                "Please check arrival_recent_rate_days and training arrivals."
            )

        result = pd.DataFrame(
            rows,
            columns=["ds", "gen_intensity", "lower_second"]
        )
        if not result.empty and accumulated > required_mass + 1e-10:
            result = NeuralProphetGenerator._trim_intensity_to_mass(
                result,
                required_mass
            )

        return result


    @staticmethod
    def _calendar_aware_fallback(
            start,
            count,
            mean_rate_per_hour,
            history,
            rng,
            smoothing=0.05,
            strength=1.0,
            sampling_method="exact_n"):
        """
        兼容旧调用接口。

        当前实现内部同样使用 calendar intensity + global exact-N，参数
        sampling_method 仅为保持旧接口兼容，不再执行逐小时 stochastic_round。
        """
        count = int(count)
        if count <= 0:
            return []

        intensity = NeuralProphetGenerator._build_calendar_fallback_intensity(
            start=start,
            required_mass=float(count),
            mean_rate_per_hour=mean_rate_per_hour,
            history=history,
            smoothing=smoothing,
            strength=strength
        )
        hourly_counts = NeuralProphetGenerator._allocate_exact_n_counts(
            intensity_df=intensity,
            target_n=count,
            rng=rng
        )
        return NeuralProphetGenerator._expand_hourly_counts(
            hourly_counts=hourly_counts,
            rng=rng
        )


    @staticmethod
    def _exponential_fallback(start, count, mean_rate_per_hour, rng):
        """
        旧版兼容方法。generate() 已改用 _calendar_aware_fallback，
        防止指数间隔破坏训练日志的星期-小时到达模式。
        """
        if count <= 0:
            return []
        mean_gap_seconds = 3600.0 / max(float(mean_rate_per_hour), 1e-6)
        gaps = rng.exponential(mean_gap_seconds, size=int(count))
        cumulative = np.cumsum(gaps)
        start = pd.to_datetime(start)
        return [
            start + pd.to_timedelta(float(x), unit="s")
            for x in cumulative
        ]

    def _predict_lagged_block(
            self,
            m,
            context_df,
            block_start,
            block_end,
            profile,
            max_cap=None):
        """
        n_lags>0 的真正逐小时 recursive multi-step forecasting。

        对每个未来小时 t：
          1) 仅追加一个 y=NaN 的未来行并预测 t；
          2) 取 yhat1(t)；
          3) 将 yhat1(t) 作为下一小时的历史 y 写回 context；
          4) 再预测 t+1。

        因而不会再把整块未来 y=0 当成 AR 输入。若 block_start 与历史末端之间
        有时间间隔，会先递归桥接该间隔，但只返回 block_start 之后的预测。
        """
        context_df = context_df[["ds", "y"]].copy()
        context_df["ds"] = pd.to_datetime(context_df["ds"])
        if context_df["ds"].dt.tz is not None:
            context_df["ds"] = context_df["ds"].dt.tz_localize(None)
        context_df["y"] = pd.to_numeric(
            context_df["y"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        context_df = context_df.sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)

        n_lags = max(1, int(getattr(m, "n_lags", 0)))
        lookback_len = max(n_lags * 3, 24 * 10)
        if n_lags >= 168 or profile.get("weekly_strength", 0.0) > 0.25:
            lookback_len = max(lookback_len, 24 * 21)
        context_df = context_df.iloc[-lookback_len:].copy()

        cap = self.max_cap if max_cap is None else float(max_cap)
        if not np.isfinite(cap) or cap <= 0.0:
            cap = float("inf")

        history_end = pd.to_datetime(context_df["ds"].max())
        block_start = pd.to_datetime(block_start)
        block_end = pd.to_datetime(block_end)
        recursive_start = history_end + pd.Timedelta(hours=1)
        if recursive_start > block_end:
            return pd.DataFrame(columns=["ds", "yhat1"]), context_df

        future_dates = pd.date_range(
            start=recursive_start,
            end=block_end,
            freq="1h",
        )
        collected = []

        for ts in future_dates:
            future_row = pd.DataFrame({"ds": [ts], "y": [np.nan]})
            predict_input = pd.concat(
                [context_df[["ds", "y"]], future_row],
                ignore_index=True,
            )
            forecast = self._safe_predict(m, predict_input)
            if forecast is None or len(forecast) == 0 or "yhat1" not in forecast.columns:
                break

            forecast = forecast.copy()
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            row = forecast[forecast["ds"] == ts]
            if row.empty:
                break
            row = row.iloc[-1]

            raw_yhat = pd.to_numeric(pd.Series([row.get("yhat1")]), errors="coerce").iloc[0]
            yhat = 0.0 if pd.isna(raw_yhat) else float(raw_yhat)
            yhat = float(np.clip(yhat, 0.0, cap))

            if ts >= block_start:
                pred_row = {"ds": ts, "yhat1": yhat}
                # 保留 NeuralProphet 的 yhat1 分位数列，供后续 triangular 模式使用。
                for col in forecast.columns:
                    if col == "yhat1" or not str(col).startswith("yhat1"):
                        continue
                    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                    if pd.isna(value):
                        continue
                    pred_row[col] = float(np.clip(float(value), 0.0, cap))
                collected.append(pred_row)

            # 真正的 one-step feedback：下一步只看到已预测出的历史值，不看到未来 0。
            feedback_row = pd.DataFrame({"ds": [ts], "y": [yhat]})
            context_df = pd.concat(
                [context_df[["ds", "y"]], feedback_row],
                ignore_index=True,
            ).iloc[-lookback_len:].copy()

        if not collected:
            return pd.DataFrame(columns=["ds", "yhat1"]), context_df

        pred_df = pd.DataFrame(collected)
        pred_df = pred_df[
            (pred_df["ds"] >= block_start) &
            (pred_df["ds"] <= block_end)
        ].drop_duplicates(subset=["ds"], keep="last").sort_values("ds")
        return pred_df.reset_index(drop=True), context_df

    def _predict_density_no_lags(
            self,
            m,
            forecast_start,
            horizon_end,
            freq,
            max_cap=None):
        """n_lags=0 时直接预测固定跨度内未来每小时强度。"""
        future_dates = pd.date_range(start=forecast_start, end=horizon_end, freq=freq)
        if len(future_dates) == 0:
            return pd.DataFrame(columns=["ds", "yhat1"])

        future_df = pd.DataFrame({"ds": future_dates, "y": None})
        forecast = self._safe_predict(m, future_df)

        keep_cols = [
            c for c in forecast.columns
            if c == "ds" or str(c).startswith("yhat1")
        ]
        pred_df = forecast[keep_cols].copy()
        cap = self.max_cap if max_cap is None else float(max_cap)
        if not np.isfinite(cap) or cap <= 0.0:
            cap = float("inf")
        for col in [c for c in pred_df.columns if str(c).startswith("yhat1")]:
            pred_df[col] = pd.to_numeric(
                pred_df[col], errors="coerce"
            ).fillna(0.0).clip(lower=0.0, upper=cap)

        if pred_df["ds"].dt.tz is not None:
            pred_df["ds"] = pred_df["ds"].dt.tz_localize(None)

        return pred_df

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)
