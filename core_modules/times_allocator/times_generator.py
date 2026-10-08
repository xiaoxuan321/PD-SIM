# -*- coding: utf-8 -*-

import itertools
import json
import os
import shutil
from datetime import datetime
from operator import itemgetter
from pickle import dump

import numpy as np
import pandas as pd
import readers.log_splitter as ls
import utils.support as sup
from fitter import Fitter
from sklearn.preprocessing import MaxAbsScaler

from core_modules.times_allocator import embedder as emb
from core_modules.times_allocator import intercase_features_calculator as it
from core_modules.times_allocator import times_model_trainer as tmt  # 使用固定参数训练器
from core_modules.times_allocator import times_predictor as tp
from extraction import log_replayer as rpl
from extraction import role_discovery as rl


# ==============================================================================
# 🛠️ 辅助类：自定义对数缩放器 (放入文件顶部或 TimesGenerator 类之前)
# ==============================================================================
class Log1pScaler:
    """
    兼容 sklearn 接口的 Log1p 缩放器
    用于处理长尾分布和 0 值 (Log-Transformation)
    """

    def fit(self, X, y=None):
        return self  # 不需要计算均值方差，直接返回

    def transform(self, X):
        # log1p(x) = log(x + 1)
        return np.log1p(X)

    def inverse_transform(self, X):
        # expm1(x) = exp(x) - 1
        # 增加 max(0) 保护，防止微小负数导致错误
        return np.maximum(0, np.expm1(X))

    # ==============================================================================
    # 🚀 修改后的 _transform_features 方法
    # ==============================================================================


class TimesGenerator:
    """
    时间生成器类，用于评估事件间的到达时间
    This class evaluates the inter-arrival times
    """

    def __init__(self, process_graph, log, parms):
        """构造函数"""
        print(f"[TimesGenerator] 初始化时间生成器，参数: {parms}")
        self.log = log
        self.process_graph = process_graph
        self.parms = parms

        self.one_timestamp = parms['read_options']['one_timestamp']
        self.timeformat = parms['read_options']['timeformat']
        self.model_metadata = dict()

        # === 加载或训练模型 ===
        self._load_model()

    # =============================================================================
    # 生成轨迹
    # =============================================================================
    def _load_model(self) -> None:
        """加载或训练时间预测模型"""
        print("[_load_model] 开始加载或训练时间预测模型")
        model_path = self._define_model_path(self.parms)
        print(f"[_load_model] 模型路径定义完成: {model_path}")

        # 检查模型是否存在
        if isinstance(model_path, tuple):
            self.proc_model_path = model_path[0]
            self.wait_model_path = model_path[1]
            model_exist = (
                    os.path.exists(model_path[0]) and os.path.exists(model_path[1]))
            self.parms['proc_model_path'] = model_path[0]
            self.parms['wait_model_path'] = model_path[1]
            print(f"[_load_model] 双模型路径: 处理模型={model_path[0]}, 等待模型={model_path[1]}, 存在={model_exist}")
        else:
            self.model_path = model_path
            model_exist = os.path.exists(model_path)
            self.parms['model_path'] = model_path
            print(f"[_load_model] 单模型路径: {model_path}, 存在={model_exist}")

        # 如果模型不存在或需要更新，则训练新模型
        if not model_exist or self.parms.get('update_times_gen', False):
            print("[_load_model] 开始训练新模型...")
            times_trainer = self._discover_model()

            # 简化比较逻辑：直接保存新模型
            metadata_file = self._get_metadata_file_path(model_path)
            print(f"[_load_model] 元数据文件路径: {metadata_file}")

            # 可选：与旧模型比较
            save = True
            if model_exist and os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    old_metadata = json.load(f)
                old_loss = old_metadata.get('loss', float('inf'))
                new_loss = times_trainer.best_loss
                print(f"[_load_model] 旧模型损失={old_loss:.4f}, 新模型损失={new_loss:.4f}")

                if old_loss < new_loss:
                    save = False
                    print("[_load_model] 旧模型性能更好，不保存新模型")

            if save:
                print("[_load_model] 保存新模型...")
                self._save_model(metadata_file, times_trainer, model_path)

                # 保存特征缩放器
                name = metadata_file.replace('_meta.json', '')
                dump(self.scaler, open(name + '_scaler.pkl', 'wb'))
                print(f"[_load_model] 特征缩放器已保存: {name}_scaler.pkl")

            # 清理输出文件夹
            if os.path.exists(self.parms['output']):
                shutil.rmtree(self.parms['output'])
                print(f"[_load_model] 临时输出文件夹已清理: {self.parms['output']}")
        else:
            print("[_load_model] 使用现有模型，无需训练")

    def _get_metadata_file_path(self, model_path):
        """获取元数据文件路径"""
        if isinstance(model_path, tuple):
            model = os.path.splitext(os.path.split(model_path[0])[1])[0]
            model = (model.replace('dpiapr', 'diapr') if self.parms.get('all_r_pool', False)
                     else model.replace('dpispr', 'dispr'))
            metadata_file = os.path.join(self.parms['times_gen_path'], model + '_meta.json')
        else:
            model = os.path.splitext(os.path.split(model_path)[1])[0]
            metadata_file = os.path.join(self.parms['times_gen_path'], model + '_meta.json')
        return metadata_file

    def _build_duration_runtime_profile(self):
        """
        从训练日志中提取 case duration 的运行期画像。
        不依赖测试集，不依赖特定数据集名称。
        """
        df = self._get_duration_source_df()
        if df.empty:
            return {}

        start_col = "start_timestamp" if "start_timestamp" in df.columns else "start_time"
        end_col = "end_timestamp" if "end_timestamp" in df.columns else "end_time"

        if start_col not in df.columns or end_col not in df.columns:
            return {}

        if "task" in df.columns:
            df = df[~df["task"].isin(["Start", "End"])].copy()

        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
        df = df.dropna(subset=[start_col, end_col])

        if df.empty:
            return {}

        case_df = df.groupby("caseid").agg(
            case_start=(start_col, "min"),
            case_end=(end_col, "max")
        ).reset_index()

        case_df["duration"] = (case_df["case_end"] - case_df["case_start"]).dt.total_seconds()
        case_df = case_df[case_df["duration"] > 0].copy()

        if case_df.empty:
            return {}

        case_df = case_df.sort_values("case_end").reset_index(drop=True)

        all_arr = case_df["duration"].to_numpy(dtype=float)

        # recent windows
        tail_10 = case_df.tail(max(1, int(len(case_df) * 0.10)))["duration"].to_numpy(dtype=float)
        tail_20 = case_df.tail(max(1, int(len(case_df) * 0.20)))["duration"].to_numpy(dtype=float)
        tail_30 = case_df.tail(max(1, int(len(case_df) * 0.30)))["duration"].to_numpy(dtype=float)

        def pct(arr, q, default=0.0):
            if len(arr) == 0:
                return float(default)
            return float(np.percentile(arr, q))

        p50 = pct(all_arr, 50)
        p80 = pct(all_arr, 80)
        p95 = pct(all_arr, 95)
        p99 = pct(all_arr, 99)

        r10_p50 = pct(tail_10, 50, p50)
        r20_p50 = pct(tail_20, 50, p50)
        r30_p50 = pct(tail_30, 50, p50)

        r10_p80 = pct(tail_10, 80, p80)
        r20_p80 = pct(tail_20, 80, p80)
        r30_p80 = pct(tail_30, 80, p80)

        duration_tail_ratio = p95 / max(p50, 1.0)
        duration_extreme_ratio = p99 / max(p95, 1.0)

        # 趋势漂移：最近时长相对于整体是否变短 / 变长
        recent_global_ratio = r20_p50 / max(p50, 1.0)

        # calendar regularity
        weekday_ratio = float((case_df["case_start"].dt.weekday < 5).mean())
        workhour_ratio = float(
            ((case_df["case_start"].dt.weekday < 5) &
             (case_df["case_start"].dt.hour >= 8) &
             (case_df["case_start"].dt.hour < 18)).mean()
        )
        peak_hour_share = float(case_df["case_start"].dt.hour.value_counts(normalize=True).max())

        calendar_score = 0.45 * workhour_ratio + 0.35 * weekday_ratio + 0.20 * peak_hour_share

        profile = {
            "num_cases": int(len(case_df)),
            "span_days": float((case_df["case_end"].max() - case_df["case_start"].min()).total_seconds() / 86400.0),
            "p50": float(p50),
            "p80": float(p80),
            "p95": float(p95),
            "p99": float(p99),
            "r10_p50": float(r10_p50),
            "r20_p50": float(r20_p50),
            "r30_p50": float(r30_p50),
            "r10_p80": float(r10_p80),
            "r20_p80": float(r20_p80),
            "r30_p80": float(r30_p80),
            "duration_tail_ratio": float(duration_tail_ratio),
            "duration_extreme_ratio": float(duration_extreme_ratio),
            "recent_global_ratio": float(recent_global_ratio),
            "weekday_ratio": float(weekday_ratio),
            "workhour_ratio": float(workhour_ratio),
            "peak_hour_share": float(peak_hour_share),
            "calendar_score": float(calendar_score),
        }

        print("[duration runtime profile]")
        print(profile)
        return profile

    def _get_duration_source_df(self):
        """
        获取用于构造 case duration prior 的训练数据源。
        优先级：
        1) self.log_train（当前对象里若已存在）
        2) 从 self.log.data / self.log 重新按 timeline 方式切出训练部分
        """
        if hasattr(self, "log_train") and self.log_train is not None:
            df = pd.DataFrame(self.log_train).copy()
            src = "self.log_train"
            print(f"[duration source] 使用数据源: {src}, 记录数={len(df)}")
            return df

        if hasattr(self.log, "data"):
            raw_df = pd.DataFrame(self.log.data).copy()
            src = "self.log.data"
        else:
            raw_df = pd.DataFrame(self.log).copy()
            src = "self.log"

        if raw_df.empty:
            print(f"[duration source] 使用数据源: {src}, 但为空")
            return raw_df

        # 尝试重建与训练一致的时间切分
        try:
            records = raw_df.to_dict("records")
            splitter = ls.LogSplitter(records)
            train, _ = splitter.split_log("timeline_trace", 0.8, self.one_timestamp)

            df = pd.DataFrame(train).copy()
            print(f"[duration source] 使用数据源: reconstructed_train_split_from_{src}, 记录数={len(df)}")
            return df
        except Exception as e:
            print(f"[duration source] 重建训练切分失败，退回原始数据源 {src}: {e}")
            print(f"[duration source] 使用数据源: {src}, 记录数={len(raw_df)}")
            return raw_df

    def _resolve_duration_prior_params(self, profile):
        """
        根据 duration runtime profile 自动推断先验构造参数
        目标：
        1) 保持通用
        2) 对 ACR 这类“稀疏 + 短trace + 低并发 + 非强calendar”的数据自动更强压缩
        3) 对长尾/高密度数据不过度压缩
        """
        if not profile:
            return {
                "recent_ratio": 0.15,
                "upper_winsor_q": 0.92,
                "lower_winsor_q": 0.02,
                "shrink_strength": 0.25,
                "duration_scale": 0.82,
                "upper_tail_compress": 0.30,
                "sample_quantile_low": 0.05,
                "sample_quantile_high": 0.95,
            }

        num_cases = int(profile.get("num_cases", 0))
        span_days = float(profile.get("span_days", 0.0))
        tail_ratio = float(profile.get("duration_tail_ratio", 1.0))
        extreme_ratio = float(profile.get("duration_extreme_ratio", 1.0))
        recent_ratio_score = float(profile.get("recent_global_ratio", 1.0))
        calendar_score = float(profile.get("calendar_score", 0.0))

        p50 = float(profile.get("p50", 0.0))
        r10_p50 = float(profile.get("r10_p50", p50))
        r20_p50 = float(profile.get("r20_p50", p50))
        r30_p50 = float(profile.get("r30_p50", p50))

        # -------------------------
        # 画像识别（完全按统计特征）
        # -------------------------
        sparse_short_profile = (
                num_cases < 1200 and
                span_days < 180 and
                tail_ratio < 8.0 and
                calendar_score < 0.80
        )

        recent_shorter_profile = (recent_ratio_score < 0.92)
        heavy_tail_profile = (tail_ratio >= 8.0 or extreme_ratio >= 2.2)

        # -------------------------
        # 默认通用基线
        # -------------------------
        recent_ratio = 0.18
        upper_winsor_q = 0.93
        lower_winsor_q = 0.02
        shrink_strength = 0.18
        duration_scale = 0.88
        upper_tail_compress = 0.25
        sample_quantile_low = 0.03
        sample_quantile_high = 0.97

        # -------------------------
        # 样本少：窗口略大
        # -------------------------
        if num_cases < 120:
            recent_ratio = 0.30
        elif num_cases < 400:
            recent_ratio = 0.20
        else:
            recent_ratio = 0.15

        # 最近明显变短：更看近一点
        if recent_shorter_profile:
            recent_ratio = max(0.05, recent_ratio - 0.05)

        # -------------------------
        # 长尾强：更收上界，但别整体压得过猛
        # -------------------------
        if heavy_tail_profile:
            upper_winsor_q = 0.88
            shrink_strength = 0.22
            duration_scale = 0.84
            upper_tail_compress = 0.40
            sample_quantile_low = 0.05
            sample_quantile_high = 0.92

        # -------------------------
        # ACR 类画像：自动更强压缩
        # -------------------------
        if sparse_short_profile:
            recent_ratio = 0.05 if num_cases >= 300 else 0.08
            upper_winsor_q = 0.85
            lower_winsor_q = 0.03
            shrink_strength = 0.38
            duration_scale = 0.68 if recent_shorter_profile else 0.74
            upper_tail_compress = 0.50
            sample_quantile_low = 0.10
            sample_quantile_high = 0.88

        # 如果最近窗口已经比整体短很多，就再轻微加大压缩
        if recent_ratio_score < 0.85:
            duration_scale *= 0.92
            shrink_strength = min(shrink_strength + 0.06, 0.50)

        params = {
            "recent_ratio": float(np.clip(recent_ratio, 0.05, 0.35)),
            "upper_winsor_q": float(np.clip(upper_winsor_q, 0.80, 0.98)),
            "lower_winsor_q": float(np.clip(lower_winsor_q, 0.00, 0.05)),
            "shrink_strength": float(np.clip(shrink_strength, 0.05, 0.50)),
            "duration_scale": float(np.clip(duration_scale, 0.60, 0.95)),
            "upper_tail_compress": float(np.clip(upper_tail_compress, 0.00, 0.60)),
            "sample_quantile_low": float(np.clip(sample_quantile_low, 0.00, 0.20)),
            "sample_quantile_high": float(np.clip(sample_quantile_high, 0.80, 1.00)),
        }

        print("[duration prior adaptive params]")
        print(params)
        return params

    def _build_case_duration_targets(self, num_targets=None, runtime_profile=None, adaptive_params=None):
        """
        通用版 case duration 先验构造：
        1) 只使用训练日志
        2) 先构造 runtime profile
        3) 再由 adaptive params 自动决定 recent window / winsor / shrink
        4) 输出与当前仿真 case 数一致的 target durations
        """
        df = self._get_duration_source_df()
        if df.empty:
            return [], {}

        start_col = "start_timestamp" if "start_timestamp" in df.columns else "start_time"
        end_col = "end_timestamp" if "end_timestamp" in df.columns else "end_time"

        if start_col not in df.columns or end_col not in df.columns:
            return [], {}

        if "task" in df.columns:
            df = df[~df["task"].isin(["Start", "End"])].copy()

        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
        df = df.dropna(subset=[start_col, end_col])

        if df.empty:
            return [], {}

        case_df = df.groupby("caseid").agg(
            case_start=(start_col, "min"),
            case_end=(end_col, "max")
        ).reset_index()

        case_df["duration"] = (case_df["case_end"] - case_df["case_start"]).dt.total_seconds()
        case_df = case_df[case_df["duration"] > 0].copy()

        if case_df.empty:
            return [], {}

        case_df = case_df.sort_values("case_end").reset_index(drop=True)

        if runtime_profile is None:
            runtime_profile = self._build_duration_runtime_profile()
        if adaptive_params is None:
            adaptive_params = self._resolve_duration_prior_params(runtime_profile)

        recent_ratio = float(adaptive_params["recent_ratio"])
        upper_winsor_q = float(adaptive_params["upper_winsor_q"])
        lower_winsor_q = float(adaptive_params["lower_winsor_q"])
        shrink_strength = float(adaptive_params["shrink_strength"])
        duration_scale = float(adaptive_params["duration_scale"])
        upper_tail_compress = float(adaptive_params["upper_tail_compress"])
        sample_quantile_low = float(adaptive_params["sample_quantile_low"])
        sample_quantile_high = float(adaptive_params["sample_quantile_high"])

        keep_n = max(1, int(round(len(case_df) * recent_ratio)))
        recent_df = case_df.tail(keep_n).copy()

        arr = recent_df["duration"].to_numpy(dtype=float)
        arr = np.sort(arr)

        if len(arr) == 0:
            return [], {}

        # ---------- winsorize ----------
        lower_val = float(np.quantile(arr, lower_winsor_q))
        upper_val = float(np.quantile(arr, upper_winsor_q))
        arr = np.clip(arr, lower_val, upper_val)

        # ---------- target center ----------
        recent_median = float(np.median(arr))
        r10_p50 = float(runtime_profile.get("r10_p50", recent_median))
        r20_p50 = float(runtime_profile.get("r20_p50", recent_median))
        global_p50 = float(runtime_profile.get("p50", recent_median))

        # 更偏向最近窗口，而不是全局
        target_center = min(recent_median, r20_p50, max(r10_p50, 60.0))
        target_center = min(target_center, global_p50)

        # ---------- shrink toward target center ----------
        arr = (1.0 - shrink_strength) * arr + shrink_strength * target_center

        # ---------- compress upper tail ----------
        # 只压 target_center 以上的部分
        arr = np.where(
            arr > target_center,
            target_center + (arr - target_center) * (1.0 - upper_tail_compress),
            arr
        )

        # ---------- global downward scale ----------
        arr = arr * duration_scale

        # ---------- floor / cap ----------
        floor_val = max(60.0, target_center * 0.12)
        cap_val = float(np.quantile(arr, min(0.98, upper_winsor_q + 0.03)))
        arr = np.clip(arr, floor_val, cap_val)

        arr = np.sort(arr)

        profiles = {
            "ALL": {
                "p50": float(np.percentile(arr, 50)),
                "p80": float(np.percentile(arr, 80)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        }

        if num_targets is None:
            return arr.tolist(), profiles

        if num_targets <= 0:
            return [], profiles

        if num_targets == 1:
            return [float(np.median(arr))], profiles

        # 不再取 [0,1] 全分位，避免把长尾稳定带进去
        q_low = min(sample_quantile_low, sample_quantile_high - 1e-6)
        q_high = max(sample_quantile_high, q_low + 1e-6)

        quantiles = np.linspace(q_low, q_high, num_targets)
        sampled = []

        n = len(arr)
        for q in quantiles:
            idx = int(round(q * (n - 1)))
            sampled.append(float(arr[idx]))

        return sampled, profiles
    def generate(self, sequences, iarr):
        """
        Generate event timestamps with the selected time predictor.

        D-SIM-compatible revision:
        - do not build case-duration targets from the training log;
        - do not inject runtime-profile correction parameters;
        - let the processing/waiting models and resource queue determine time.
        """
        print(f"[generate] 开始生成时间预测，模型类型={self.parms['model_type']}")
        model_path = (
            self.model_path
            if self.parms["model_type"] in ["basic", "inter", "inter_nt"]
            else (self.proc_model_path, self.wait_model_path)
        )
        predictor = tp.TimesPredictor(model_path, self.parms, sequences, iarr)
        return predictor.predict(self.parms["model_type"])

    def _debug_case_source(self, tag, data_obj):
        """
        打印某个数据对象的 case 数、记录数、时间范围，方便确认来源。
        """
        try:
            if hasattr(data_obj, 'data'):
                df = pd.DataFrame(data_obj.data).copy()
                source_name = f"{tag} (obj.data)"
            else:
                df = pd.DataFrame(data_obj).copy()
                source_name = f"{tag} (direct)"
        except Exception as e:
            print(f"[debug] {tag} 无法转成DataFrame: {e}")
            return

        if df.empty:
            print(f"[debug] {source_name}: 空")
            return

        case_n = df['caseid'].nunique() if 'caseid' in df.columns else 'N/A'
        rec_n = len(df)

        time_col = None
        if 'start_timestamp' in df.columns:
            time_col = 'start_timestamp'
        elif 'start_time' in df.columns:
            time_col = 'start_time'
        elif 'end_timestamp' in df.columns:
            time_col = 'end_timestamp'
        elif 'end_time' in df.columns:
            time_col = 'end_time'

        if time_col is not None:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            tmin = df[time_col].min()
            tmax = df[time_col].max()
        else:
            tmin, tmax = None, None

        print(f"[debug] {source_name}: case数={case_n}, 记录数={rec_n}, 时间范围=({tmin}, {tmax})")
    @staticmethod
    def _define_model_path(parms):
        """根据参数定义模型路径"""
        path = parms['times_gen_path']
        fname = parms['file'].split('.')[0]
        inter = parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']
        is_dual = parms['model_type'] == 'dual_inter'
        arpool = parms.get('all_r_pool', False)
        next_ac = parms['model_type'] == 'inter_nt'

        # 根据模型类型和参数生成不同的文件名
        if inter:
            if is_dual:
                if arpool:
                    return (os.path.join(path, fname + '_dpiapr.h5'),
                            os.path.join(path, fname + '_dwiapr.h5'))
                else:
                    return (os.path.join(path, fname + '_dpispr.h5'),
                            os.path.join(path, fname + '_dwispr.h5'))
            else:
                if next_ac:
                    if arpool:
                        return os.path.join(path, fname + '_inapr.h5')
                    else:
                        return os.path.join(path, fname + '_inspr.h5')
                else:
                    if arpool:
                        return os.path.join(path, fname + '_iapr.h5')
                    else:
                        return os.path.join(path, fname + '_ispr.h5')
        else:
            return os.path.join(path, fname + '.h5')

    def _save_model(self, metadata_file, times_trainer, model_path):
        """保存模型和元数据"""
        print(f"[_save_model] 保存模型到: {model_path}")
        model_metadata = dict()

        # 记录最佳参数
        model_metadata['loss'] = times_trainer.best_loss
        model_metadata['generated_at'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        model_metadata['ac_index'] = self.ac_index
        model_metadata['usr_index'] = self.usr_index
        model_metadata['log_size'] = len(pd.DataFrame(self.log).caseid.unique())
        model_metadata['sim_metric'] = 'val_loss'

        # 合并超参数
        model_metadata.update(times_trainer.best_parms)

        model_name = metadata_file.replace('_meta.json', '')

        # 处理交互特征模型
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            model_metadata['roles'] = self.roles
            model_metadata['roles_table'] = self.roles_table.to_dict('records')
            model_metadata['inter_mean_states'] = self.mean_states

            # 保存交互特征缩放器
            dump(self.inter_scaler, open(model_name + '_inter_scaler.pkl', 'wb'))
            if self.parms['model_type'] == 'dual_inter':
                dump(self.end_inter_scaler, open(model_name + '_end_inter_scaler.pkl', 'wb'))
            print(f"[_save_model] 交互特征缩放器已保存")

        # 保存模型文件
        if isinstance(model_path, tuple):
            proc_model_file = os.path.split(model_path[0])[1]
            wait_model_file = os.path.split(model_path[1])[1]

            shutil.copyfile(
                os.path.join(times_trainer.best_output, proc_model_file),
                self.proc_model_path
            )
            shutil.copyfile(
                os.path.join(times_trainer.best_output, wait_model_file),
                self.wait_model_path
            )
            print(f"[_save_model] 双模型已复制: {self.proc_model_path}, {self.wait_model_path}")
        else:
            source = os.path.join(times_trainer.best_output, self.parms['file'].split('.')[0] + '.h5')
            shutil.copyfile(source, self.model_path)
            print(f"[_save_model] 单模型已复制: {self.model_path}")

        # 保存元数据
        sup.create_json(model_metadata, metadata_file)
        print(f"[_save_model] 模型元数据已保存: {metadata_file}")

    # =============================================================================
    # 训练模型（使用固定参数，不进行优化）
    # =============================================================================
    def extract_distribution(self, X):
        """拟合最佳概率分布"""
        print(f"[extract_distribution] 拟合{len(X)}个样本的最佳分布")
        f = Fitter(X, distributions=['norm', 'expon', 'uniform', 'lognorm', 'loguniform'])
        f.fit()
        best_dist = list(f.get_best(method='sumsquare_error').keys())[0]
        print(f"[extract_distribution] 最佳拟合分布: {best_dist}")
        return best_dist

    def extract_day_moment(self, start_timestamp):
        """根据时间戳提取时间段（早晨/下午/晚上）"""
        if 0 <= start_timestamp.hour < 12:
            return 'morning'
        elif 12 <= start_timestamp.hour < 17:
            return 'afternoon'
        elif 17 <= start_timestamp.hour < 24:
            return 'night'

    def extract_description_activities(self, log):
        """提取活动描述性统计信息"""
        print("[extract_description_activities] 开始提取活动描述信息")
        log['order'] = log.sort_values(by='start_timestamp', ascending=True).groupby('caseid').cumcount() + 1
        activity_desc = []

        for activity in log['task'].drop_duplicates():
            log_activity = log[log['task'] == activity]
            day_moment = list(set([self.extract_day_moment(x) for x in log_activity['start_timestamp']]))
            rol = list(set([x for x in log_activity['user']]))
            trace_position = np.mean(list(set([x for x in log_activity['order']])))
            distribution = self.extract_distribution([x for x in log_activity['processing_time']])
            mean_proc_time = np.mean(log_activity['processing_time'])
            std_proc_time = np.std(log_activity['processing_time'])
            activity_desc.append(
                [activity, day_moment, rol, trace_position, distribution, mean_proc_time, std_proc_time])

        df_activity_desc = pd.DataFrame(data=activity_desc,
                                        columns=['task_name', 'day_moment', 'rol', 'trace_position', 'dstribution',
                                                 'mean_processing_time', 'std_processing_time'])
        output_path = 'output_files/Activity_description.csv'
        df_activity_desc.to_csv(output_path, sep='|')
        print(f"[extract_description_activities] 活动描述已保存到: {output_path}")

    def _discover_model(self, **_kwargs):
        """训练时间预测模型（使用固定参数，不进行优化）"""
        print("[_discover_model] 开始训练时间预测模型（使用固定参数）")

        # 创建索引
        self.ac_index, self.index_ac = self._indexing(self.log.data, 'task')
        self.usr_index, self.index_usr = self._indexing(self.log.data, 'user')
        print(f"[_discover_model] 活动索引数量: {len(self.ac_index)}, 用户索引数量: {len(self.usr_index)}")

        # 重放日志以计算时间
        self._replay_process()
        print("[_discover_model] 日志重放完成")

        # 添加交互特征
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            print("[_discover_model] 添加交互特征")
            self._add_intercases()

        # 划分训练/验证集
        self._split_timeline(0.8, self.one_timestamp)
        print(f"[_discover_model] 日志划分完成: 训练集={len(self.log_train)}, 验证集={len(self.log_valdn)}")
        self._debug_case_source("self.log", self.log)
        self._debug_case_source("self.log_train", self.log_train)
        self._debug_case_source("self.log_valdn", self.log_valdn)
        # 添加计算时间
        self.log_train = self._add_calculated_times(self.log_train)
        self.log_valdn = self._add_calculated_times(self.log_valdn)
        print("[_discover_model] 时间特征计算完成")

        # 添加活动索引
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log_train['ac_index'] = self.log_train.apply(ac_idx, axis=1)
        self.log_valdn['ac_index'] = self.log_valdn.apply(ac_idx, axis=1)

        # 添加下一个活动索引（特定模型）
        if self.parms['model_type'] in ['inter_nt', 'dual_inter']:
            ac_idx = lambda x: self.ac_index[x['n_task']]
            self.log_train['n_ac_index'] = self.log_train.apply(ac_idx, axis=1)
            self.log_valdn['n_ac_index'] = self.log_valdn.apply(ac_idx, axis=1)
            print("[_discover_model] 下一个活动索引已添加")

        # 创建训练+验证日志的索引
        self.train_val_log = pd.concat([pd.DataFrame(self.log_train), pd.DataFrame(self.log_valdn)])
        self.ac_index_train_val, self.index_ac_train_val = self._indexing(self.train_val_log, 'task')
        self.usr_index_train_val, self.index_usr_train_val = self._indexing(self.train_val_log, 'user')
        print(f"[_discover_model] 训练+验证日志索引: 活动={len(self.ac_index_train_val)}, 用户={len(self.usr_index_train_val)}")

        # 创建嵌入矩阵
        emb_trainer = emb.Embedder(self.parms, self.log, self.ac_index, self.index_ac, self.usr_index, self.index_usr)
        self.ac_weights = emb_trainer.create_embeddings(self.parms['emb_method'])
        print(f"[_discover_model] 嵌入矩阵创建完成, 形状={self.ac_weights.shape if self.ac_weights is not None else 'N/A'}")

        # 提取活动描述信息
        self.extract_description_activities(self.log.copy())

        # 特征缩放
        self._transform_features()
        print("[_discover_model] 特征缩放完成")

        # 设置输出目录
        self.parms['output'] = os.path.join('output_files', sup.folder_id())
        print(f"[_discover_model] 输出目录: {self.parms['output']}")

        # 训练器优先读取 parms['times_config']；未提供时使用 ACTIVE_CONFIG。
        times_trainer = tmt.TimesModelTrainer(
            self.parms,
            self.log_train,
            self.log_valdn,
            self.ac_index,
            emb_trainer.embedding_file_name
        )

        # 执行训练
        times_trainer.execute_trials()

        print(f"[_discover_model] 模型训练完成, 最佳损失={times_trainer.best_loss:.4f}")
        return times_trainer

    # =============================================================================
    # Support modules
    # =============================================================================

    def _replay_process(self) -> None:
        """
        重放日志以计算处理/等待时间
        Process replaying
        """
        print("[_replay_process] 开始日志重放...")
        replayer = rpl.LogReplayer(self.process_graph,
                                   self.log.get_traces(),
                                   self.parms,
                                   msg='reading conformant training traces:')
        print("[_replay_process] 日志重放完成，处理统计信息")
        self.log = replayer.process_stats.rename(columns={'resource': 'user'})
        print("[_replay_process] 填充缺失的用户值为'sys'")
        self.log['user'] = self.log['user'].fillna('sys')
        self.log = self.log.to_dict('records')
        print(f"[_replay_process] 重放后日志记录数: {len(self.log)}")

    @staticmethod
    def _indexing(log, feat):
        """
        创建特征索引映射
        """
        print(f"[_indexing] 开始为特征 '{feat}' 创建索引")
        log = pd.DataFrame(log)

        # 过滤掉'Start'和'End'活动
        if feat == 'task':
            print("[_indexing] 过滤掉'Start'和'End'活动")
            log = log[~log[feat].isin(['Start', 'End'])]
        else:
            print("[_indexing] 填充缺失值为'sys'")
            log[feat] = log[feat].fillna('sys')

        # 获取唯一值并排序
        subsec_set = log[feat].unique().tolist()
        subsec_set = [x for x in subsec_set if x not in ['Start', 'End']]
        print(f"[_indexing] 唯一值数量: {len(subsec_set)}")

        # 创建索引映射
        index = dict()
        for i, value in enumerate(subsec_set):
            index[value] = i + 1
        index['Start'] = 0
        index['End'] = len(index)
        index_inv = {v: k for k, v in index.items()}
        print(f"[_indexing] 索引创建完成，总条目数: {len(index)}")
        return index, index_inv

    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        按时间线划分训练集和验证集

        Parameters
        ----------
        size : float, 验证集比例
        one_ts : bool, 是否只使用单个时间戳
        """
        print(f"[_split_timeline] 开始划分数据集，验证集比例={size}, 使用单时间戳={one_ts}")
        total_events = len(self.log)
        print(f"[_split_timeline] 总事件数: {total_events}")

        # 分割日志数据
        splitter = ls.LogSplitter(self.log)
        print("[_split_timeline] 使用'timeline_trace'方法分割（案例级，不丢弃跨分割案例）")
        train, valdn = splitter.split_log('timeline_trace', size, one_ts)

        # 设置分割结果
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)

        # 过滤掉'Start'和'End'活动
        print("[_split_timeline] 过滤掉'Start'和'End'活动")
        valdn = valdn[~valdn.task.isin(['Start', 'End'])]
        train = train[~train.task.isin(['Start', 'End'])]

        # 排序并重置索引
        self.log_valdn = (valdn.sort_values(key, ascending=True).reset_index(drop=True))
        self.log_train = (train.sort_values(key, ascending=True).reset_index(drop=True))
        print(f"[_split_timeline] 划分完成: 训练集={len(self.log_train)}, 验证集={len(self.log_valdn)}")

    def _add_intercases(self):
        """
        添加案例间特征（WIP计数等）
        """
        print("[_add_intercases] 开始添加案例间特征")
        log = pd.DataFrame(self.log)
        print(f"[_add_intercases] 资源池相似度阈值: {self.parms.get('rp_similarity', 0.8)}")

        # 资源池分析
        print("[_add_intercases] 执行资源池分析...")
        res_analyzer = rl.ResourcePoolAnalyser(
            log,
            sim_threshold=self.parms.get('rp_similarity', 0.8))
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        resource_table.rename(columns={'resource': 'user'}, inplace=True)

        # 创建角色到用户的映射
        self.roles = {role: group.user.to_list() for role, group in resource_table.groupby('role')}
        print(f"[_add_intercases] 发现角色数量: {len(self.roles)}")

        # 合并资源表
        print("[_add_intercases] 合并资源表到日志")
        log = log.merge(resource_table, on='user', how='left')

        # 管理案例间特征
        print("[_add_intercases] 计算案例间特征...")
        inter_mannager = it.IntercaseMannager(log,
                                              self.parms.get('all_r_pool', False),
                                              self.parms['model_type'])
        log, mean_states = inter_mannager.fit_transform()
        self.mean_states = mean_states
        self.log = log

        # 创建角色表
        print("[_add_intercases] 创建角色-活动映射表")
        roles_table = (self.log[['caseid', 'role', 'task']]
                       .groupby(['task', 'role']).count()
                       .sort_values(by=['caseid'])
                       .groupby(level=0)
                       .tail(1)
                       .reset_index())
        self.roles_table = roles_table[['role', 'task']]
        print("[_add_intercases] 案例间特征添加完成")

    def _add_calculated_times(self, log):
        """
        添加计算的时间特征（一天中的时间、星期几等）
        """
        print("[_add_calculated_times] 开始添加计算的时间特征")
        log = pd.DataFrame(log)
        log['daytime'] = 0
        log = log.to_dict('records')

        # 按案例分组
        print("[_add_calculated_times] 按案例分组处理时间特征")
        log = sorted(log, key=lambda x: x['caseid'])
        case_count = 0
        for caseid, group in itertools.groupby(log, key=lambda x: x['caseid']):
            case_count += 1
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))

            # 处理每个事件的时间特征
            for i in range(0, len(events)):
                # 转换为一天中的秒数
                time = events[i]['start_timestamp'].time()
                time_sec = time.second + time.minute * 60 + time.hour * 3600
                events[i]['st_daytime'] = time_sec
                events[i]['st_weekday'] = events[i]['start_timestamp'].weekday()

                # 如果是双交互模型，添加结束时间特征
                if self.parms['model_type'] == 'dual_inter':
                    time = events[i]['end_timestamp'].time()
                    time_sec = time.second + time.minute * 60 + time.hour * 3600
                    events[i]['end_daytime'] = time_sec
                    events[i]['end_weekday'] = events[i]['end_timestamp'].weekday()

        print(f"[_add_calculated_times] 处理完成，案例数: {case_count}, 事件总数: {len(log)}")
        return pd.DataFrame.from_dict(log)

    def _transform_features(self):
        """
        特征缩放和转换 (优化版：Log变换 + 周期性时间编码)
        """
        print("[_transform_features] 开始特征缩放和转换 (通用优化版)")

        # === 1. 处理时间和等待时间：使用 Log1p 变换 ===
        # 优化理由：Log 变换能同时适应 P2P 的零值和 BPI 的长尾分布
        cols = ['processing_time', 'waiting_time']
        print(f"[_transform_features] 对目标变量应用 Log1p 变换: {cols}")

        for frame_name, frame in [
            ("log_train", self.log_train),
            ("log_valdn", self.log_valdn),
        ]:
            values = frame[cols].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{frame_name} 的处理时间或等待时间包含 NaN/Inf"
                )
            if (values < 0).any():
                negative_counts = (frame[cols] < 0).sum().to_dict()
                raise ValueError(
                    f"{frame_name} 包含负时间，不能直接进行 Log1p: "
                    f"{negative_counts}"
                )

        # 使用自定义的 Log1pScaler，这样 _save_model 保存它后，
        # 预测器端的 inverse_transform 会自动执行 expm1，无需修改预测器代码
        self.scaler = Log1pScaler()
        self.scaler.fit(self.log_train[cols])

        self.log_train[cols] = self.scaler.transform(self.log_train[cols])
        self.log_valdn[cols] = self.scaler.transform(self.log_valdn[cols])

        # === 2. 缩放案例间特征（交互特征）===
        # 保持使用 MaxAbsScaler，因为 WIP 是计数特征，线性缩放即可
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            inter_feat = ['st_wip', 'st_tsk_wip']  # ✅ 正确的列名
            print(f"[_transform_features] 缩放案例间特征 (MaxAbs): {inter_feat}")

            self.inter_scaler = MaxAbsScaler()
            self.inter_scaler.fit(self.log_train[inter_feat])
            self.log_train[inter_feat] = self.inter_scaler.transform(self.log_train[inter_feat])
            self.log_valdn[inter_feat] = self.inter_scaler.transform(self.log_valdn[inter_feat])
            cols.extend(inter_feat)

            # === 3. 双交互模型的额外特征 ===
            if self.parms['model_type'] in ['dual_inter']:
                inter_feat = ['end_wip']  # ✅ 正确：只有 'end_wip'
                print(f"[_transform_features] 缩放结束特征 (MaxAbs): {inter_feat}")

                self.end_inter_scaler = MaxAbsScaler()
                self.end_inter_scaler.fit(self.log_train[inter_feat])
                self.log_train[inter_feat] = self.end_inter_scaler.transform(self.log_train[inter_feat])
                self.log_valdn[inter_feat] = self.end_inter_scaler.transform(self.log_valdn[inter_feat])
                cols.extend(inter_feat)

        # === 4. 时间特征优化：周期性编码 (Cyclical Encoding) ===
        # 优化理由：Sin/Cos 能让模型理解 23:59 和 00:01 是相连的，
        # 且能更好地捕捉 P2P 的硬性时间规则（如早9晚5）
        print("[_transform_features] 应用周期性时间编码 (Sin/Cos)")

        seconds_in_day = 24 * 60 * 60
        days_in_week = 7

        # 定义辅助函数以避免代码重复
        def encode_cyclical(df, col_daytime, col_weekday, prefix):
            # 日周期 (Daytime)
            df[f'{prefix}_daytime_sin'] = np.sin(2 * np.pi * df[col_daytime] / seconds_in_day)
            df[f'{prefix}_daytime_cos'] = np.cos(2 * np.pi * df[col_daytime] / seconds_in_day)
            # 周周期 (Weekday)
            df[f'{prefix}_weekday_sin'] = np.sin(2 * np.pi * df[col_weekday] / days_in_week)
            df[f'{prefix}_weekday_cos'] = np.cos(2 * np.pi * df[col_weekday] / days_in_week)

        # 处理开始时间
        encode_cyclical(self.log_train, 'st_daytime', 'st_weekday', 'st')
        encode_cyclical(self.log_valdn, 'st_daytime', 'st_weekday', 'st')

        # 添加新的特征列名
        cols.extend(['caseid', 'ac_index'])
        cols.extend(['st_daytime_sin', 'st_daytime_cos', 'st_weekday_sin', 'st_weekday_cos'])

        # === 5. 双交互模型的额外时间特征 (同样使用周期性编码) ===
        if self.parms['model_type'] in ['dual_inter']:
            print("[_transform_features] 处理双交互模型的时间特征 (Sin/Cos)")
            encode_cyclical(self.log_train, 'end_daytime', 'end_weekday', 'end')
            encode_cyclical(self.log_valdn, 'end_daytime', 'end_weekday', 'end')

            cols.extend(['end_daytime_sin', 'end_daytime_cos', 'end_weekday_sin', 'end_weekday_cos'])

        # === 6. 处理其他模型特定的特征 ===
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            suffixes = (['_st_oc', '_end_oc'] if (self.parms['model_type'] in ['dual_inter']) else ['_st_oc'])

            if self.parms.get('all_r_pool', False):
                print("[_transform_features] 处理所有资源池的特征")
                for suffix in suffixes:
                    suffix_cols = [c_n for c_n in self.log_train.columns if suffix in c_n]
                    cols.extend(suffix_cols)
            else:
                print("[_transform_features] 添加资源池特征")
                cols.extend(['rp' + x for x in suffixes])

        # === 7. 添加下一个活动的索引 ===
        if self.parms['model_type'] in ['inter_nt', 'dual_inter']:
            print("[_transform_features] 添加下一个活动索引")
            cols.extend(['n_ac_index'])

        # === 8. 过滤特征列 ===
        print(f"[_transform_features] 最终选择的特征列数: {len(cols)}")
        # 确保列存在，防止报错
        existing_cols = [c for c in cols if c in self.log_train.columns]
        if len(existing_cols) != len(cols):
            print(f"⚠️ 警告: 部分特征列丢失: {set(cols) - set(existing_cols)}")

        self.log_train = self.log_train[existing_cols]
        self.log_valdn = self.log_valdn[existing_cols]

        # === 9. 填充缺失值 ===
        print("[_transform_features] 填充缺失值为0")
        self.log_train = self.log_train.fillna(0)
        self.log_valdn = self.log_valdn.fillna(0)

        print("[_transform_features] ✅ 特征转换完成")


