# -*- coding: utf-8 -*-
"""
训练集内选择的星期－小时到达强度校准器。

该模块只处理案例到达计数，不读取测试日志。训练阶段用按时间留出的
内部验证窗口选择是否启用校准、校准强度和画像平滑系数；生成阶段只
复用训练阶段保存的选择结果。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class WeekHourCalibrator:
    """星期×小时（7×24）到达强度校准的无状态工具类。"""

    VERSION = 1

    @staticmethod
    def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[["ds", "y"]].copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)
        result["y"] = pd.to_numeric(
            result["y"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        return result

    @classmethod
    def build_scale(
            cls,
            history: pd.DataFrame,
            smoothing: float = 0.50) -> pd.Series:
        """
        仅从给定历史构建 168 个星期－小时系数，系数均值为 1。

        ``smoothing=0`` 完全采用历史槽位均值；``smoothing=1`` 完全
        收缩到全局均值，即不引入星期－小时差异。
        """
        hist = cls._clean_frame(history)
        hist["weekday"] = hist["ds"].dt.dayofweek
        hist["hour"] = hist["ds"].dt.hour

        full_index = pd.MultiIndex.from_product(
            [range(7), range(24)],
            names=["weekday", "hour"],
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
            return pd.Series(1.0, index=full_index, dtype=float)
        return slot_rate / calendar_mean

    @staticmethod
    def _quantile_columns(prediction: pd.DataFrame) -> list[str]:
        return [
            column
            for column in prediction.columns
            if str(column).startswith("yhat1")
        ]

    @classmethod
    def apply(
            cls,
            prediction: pd.DataFrame,
            scale: pd.Series,
            strength: float,
            max_cap: float | None = None) -> pd.DataFrame:
        """
        校准预测形状，同时保持当前预测块的点预测总强度基本不变。

        ``strength=0`` 严格返回未校准强度；``strength=1`` 完全采用训练
        画像。点预测和分位数列使用同一倍率，避免分位数失配。
        """
        if prediction is None or prediction.empty:
            return prediction

        result = prediction.copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_localize(None)

        strength = float(np.clip(strength, 0.0, 1.0))
        slot_scale = np.asarray([
            float(scale.get((int(ts.dayofweek), int(ts.hour)), 1.0))
            for ts in result["ds"]
        ], dtype=float)
        effective_scale = (1.0 - strength) + strength * slot_scale

        point = pd.to_numeric(
            result["yhat1"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)

        def normalize(values: np.ndarray) -> np.ndarray:
            if point.sum() > 1e-12:
                denominator = float(
                    np.sum(point * values) / np.sum(point)
                )
            else:
                denominator = float(np.mean(values))
            if not np.isfinite(denominator) or denominator <= 0.0:
                denominator = 1.0
            return values / denominator

        effective_scale = normalize(effective_scale)
        effective_scale = np.clip(effective_scale, 0.35, 2.50)
        effective_scale = normalize(effective_scale)

        upper = np.inf if max_cap is None else float(max_cap)
        for column in cls._quantile_columns(result):
            values = pd.to_numeric(
                result[column], errors="coerce"
            ).fillna(0.0).clip(lower=0.0).to_numpy(float)
            result[column] = np.clip(
                values * effective_scale, 0.0, upper
            )
        return result

    @classmethod
    def _score(
            cls,
            actual: pd.DataFrame,
            prediction: pd.DataFrame,
            distribution_weight: float) -> dict:
        """计算小时 MAE 与归一化星期－小时 EMD 的组合分数。"""
        actual_clean = cls._clean_frame(actual)
        predicted = prediction[["ds", "yhat1"]].copy()
        predicted["ds"] = pd.to_datetime(predicted["ds"])
        if predicted["ds"].dt.tz is not None:
            predicted["ds"] = predicted["ds"].dt.tz_localize(None)
        predicted["yhat1"] = pd.to_numeric(
            predicted["yhat1"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

        aligned = actual_clean.merge(
            predicted, on="ds", how="inner"
        )
        if aligned.empty:
            return {
                "score": float("inf"),
                "normalized_mae": float("inf"),
                "week_hour_emd": float("inf"),
            }

        actual_y = aligned["y"].to_numpy(float)
        predicted_y = aligned["yhat1"].to_numpy(float)
        normalized_mae = float(
            np.mean(np.abs(actual_y - predicted_y))
            / max(float(np.mean(actual_y)), 1e-6)
        )

        aligned["weekday"] = aligned["ds"].dt.dayofweek
        aligned["hour"] = aligned["ds"].dt.hour
        full_index = pd.MultiIndex.from_product(
            [range(7), range(24)],
            names=["weekday", "hour"],
        )
        actual_slots = (
            aligned.groupby(["weekday", "hour"])["y"]
            .sum()
            .reindex(full_index, fill_value=0.0)
            .to_numpy(float)
        )
        predicted_slots = (
            aligned.groupby(["weekday", "hour"])["yhat1"]
            .sum()
            .reindex(full_index, fill_value=0.0)
            .to_numpy(float)
        )
        actual_distribution = actual_slots / max(
            float(actual_slots.sum()), 1e-12
        )
        predicted_distribution = predicted_slots / max(
            float(predicted_slots.sum()), 1e-12
        )

        # 一维离散 EMD，并除以最大槽位距离，使其大致落在 [0, 1]。
        week_hour_emd = float(
            np.abs(
                np.cumsum(actual_distribution - predicted_distribution)
            ).sum() / max(len(full_index) - 1, 1)
        )
        distribution_weight = float(
            np.clip(distribution_weight, 0.0, 1.0)
        )
        score = (
            distribution_weight * week_hour_emd
            + (1.0 - distribution_weight) * normalized_mae
        )
        return {
            "score": float(score),
            "normalized_mae": normalized_mae,
            "week_hour_emd": week_hour_emd,
        }

    @classmethod
    def select_on_validation(
            cls,
            calibration_history: pd.DataFrame,
            validation_actual: pd.DataFrame,
            validation_prediction: pd.DataFrame,
            strength_candidates: Iterable[float],
            smoothing_candidates: Iterable[float],
            max_cap: float | None,
            distribution_weight: float = 0.70,
            min_relative_improvement: float = 0.005,
            mae_worsening_tolerance: float = 0.03) -> dict:
        """
        在训练集内部留出的时间窗口上选择校准参数。

        候选包含 ``strength=0`` 作为“不校准”基线。只有组合分数达到
        最小改善，且归一化 MAE 没有超出允许退化比例时，才启用校准。
        """
        strengths = sorted({
            float(np.clip(value, 0.0, 1.0))
            for value in strength_candidates
        } | {0.0})
        smoothings = sorted({
            float(np.clip(value, 0.0, 1.0))
            for value in smoothing_candidates
        })
        if not smoothings:
            smoothings = [0.50]

        baseline_metrics = cls._score(
            validation_actual,
            validation_prediction,
            distribution_weight,
        )
        results = [{
            "strength": 0.0,
            "smoothing": None,
            **baseline_metrics,
        }]

        for smoothing in smoothings:
            scale = cls.build_scale(
                calibration_history, smoothing=smoothing
            )
            for strength in strengths:
                if strength <= 0.0:
                    continue
                calibrated = cls.apply(
                    validation_prediction,
                    scale,
                    strength,
                    max_cap=max_cap,
                )
                metrics = cls._score(
                    validation_actual,
                    calibrated,
                    distribution_weight,
                )
                results.append({
                    "strength": float(strength),
                    "smoothing": float(smoothing),
                    **metrics,
                })

        finite_results = [
            item for item in results
            if np.isfinite(item["score"])
        ]
        best = min(
            finite_results or results,
            key=lambda item: item["score"],
        )
        baseline_score = float(baseline_metrics["score"])
        relative_improvement = (
            (baseline_score - float(best["score"]))
            / max(abs(baseline_score), 1e-12)
        )
        mae_limit = (
            float(baseline_metrics["normalized_mae"])
            * (1.0 + max(float(mae_worsening_tolerance), 0.0))
        )
        enabled = bool(
            float(best["strength"]) > 0.0
            and relative_improvement >= float(min_relative_improvement)
            and float(best["normalized_mae"]) <= mae_limit
        )

        selected = best if enabled else results[0]
        return {
            "version": cls.VERSION,
            "source": "training_internal_time_validation",
            "enabled": enabled,
            "strength": float(selected["strength"]),
            "smoothing": (
                None
                if selected["smoothing"] is None
                else float(selected["smoothing"])
            ),
            "baseline_score": baseline_score,
            "selected_score": float(selected["score"]),
            "relative_improvement": float(
                relative_improvement if enabled else 0.0
            ),
            "baseline_normalized_mae": float(
                baseline_metrics["normalized_mae"]
            ),
            "selected_normalized_mae": float(
                selected["normalized_mae"]
            ),
            "baseline_week_hour_emd": float(
                baseline_metrics["week_hour_emd"]
            ),
            "selected_week_hour_emd": float(
                selected["week_hour_emd"]
            ),
            "candidates": results,
        }
