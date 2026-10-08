# -*- coding: utf-8 -*-

import os
import subprocess
import copy
import multiprocessing
from multiprocessing import Pool
import itertools
import traceback
import numpy as np
import pandas as pd
import math
import random
import optuna
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

import utils.support as sup
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from utils.support import timeit
import readers.log_splitter as ls
import readers.log_reader as lr
from support_modules.writers import xes_writer as xes
from support_modules.writers import xml_writer as xml
import analyzers.sim_evaluator as sim

import core_modules.sequences_generator.structure_miner as sm
import core_modules.sequences_generator.structure_params_miner as spm
from tqdm import tqdm
import time


# 流程结构优化器类
class StructureOptimizer():
    """
    Hyperparameter-optimizer class
    超参数优化器类,用于优化流程挖掘算法的参数
    """

    # 内部装饰器类
    class Decorators(object):
        """安全执行方法的装饰器集合"""

        @classmethod
        def safe_exec(cls, method):
            """
            安全执行方法的装饰器,捕获异常并返回执行状态
            Parameters:
                method : 要执行的方法
            Returns:
                dict : 包含执行结果和状态(STATUS_OK/STATUS_FAIL)的字典
            """

            def safety_check(*args, **kw):
                # 获取方法执行状态,默认为方法名的大写形式
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}

                # 只有当前状态为OK时才执行方法
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)  # 执行目标方法
                    except Exception as e:
                        print(e)
                        traceback.print_exc()  # 打印异常堆栈
                        response['status'] = STATUS_FAIL  # 更新状态为失败
                return response

            return safety_check

    def __init__(self, settings, log):
        """构造函数,初始化优化器"""
        # 读取输入数据
        self.log = log
        # 将日志按时间线分割为训练集和验证集(80%训练)
        self._split_timeline(0.8, settings['read_options']['one_timestamp'])

        # 创建日志的深拷贝以保留原始数据
        self.org_log = copy.deepcopy(log)
        self.org_log_train = copy.deepcopy(self.log_train)
        self.org_log_valdn = copy.deepcopy(self.log_valdn)

        # 加载设置
        self.settings = settings
        # 创建临时输出目录
        self.temp_output = os.path.join('output_files', sup.folder_id())
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)

        # 创建结果文件名
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))

        # 初始化结果文件
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()

        # 初始化 Optuna Study
        # ASHA 在 Optuna 中通过 SuccessiveHalvingPruner 实现
        self.pruner = SuccessiveHalvingPruner(
            min_resource=1,  # 至少跑 1 次仿真后开始考虑剪枝
            reduction_factor=3  # 每一梯级保留前 1/3 的优秀试验
        )
        from optuna.samplers import RandomSampler

        # self.study = optuna.create_study(
        #      direction="maximize",
        #      sampler=RandomSampler(),
        #      pruner=self.pruner
        #  )
        self.study = optuna.create_study(
            direction="maximize",
            #正确：multivariate 是 TPESampler 的参数
            sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=10),
            pruner=self.pruner
        )

    def execute_trials(self):
        """
        执行ASHA超参数优化并汇总结果。

        关键处理：
        1. self.log_train和self.log_valdn是LogReader对象；
        2. 仅在统计案例数和事件数时读取LogReader.data；
        3. 每个trial开始前重新恢复原始训练集和验证集；
        4. 防止removal策略跨trial累积删除训练轨迹；
        5. Top-K只统计完整完成的trial。
        """

        def log_to_dataframe(log_object):
            """
            将LogReader或DataFrame统一转换成DataFrame。

            注意：
            该函数只用于数据规模统计，不会修改原日志对象。
            """
            if isinstance(log_object, pd.DataFrame):
                return log_object.copy()

            if hasattr(log_object, 'data'):
                if isinstance(log_object.data, pd.DataFrame):
                    return log_object.data.copy()

                return pd.DataFrame(log_object.data)

            raise TypeError(
                f"不支持的日志对象类型：{type(log_object).__name__}。"
                "日志对象既不是DataFrame，也不存在data属性。"
            )

        def get_log_statistics(log_object, log_name):
            """
            统计日志的案例数、事件数和案例ID集合。
            """
            dataframe = log_to_dataframe(log_object)

            if 'caseid' not in dataframe.columns:
                raise KeyError(
                    f"{log_name}中不存在caseid列，"
                    f"当前列为：{list(dataframe.columns)}"
                )

            case_number = dataframe['caseid'].nunique()
            event_number = len(dataframe)
            case_ids = frozenset(dataframe['caseid'].unique())

            return {
                'cases': case_number,
                'events': event_number,
                'case_ids': case_ids
            }

        # ========================================================
        # 1. 获取固定的原始训练集和验证集规模
        # ========================================================
        expected_train_stats = get_log_statistics(
            self.org_log_train,
            '原始内部训练集'
        )

        expected_val_stats = get_log_statistics(
            self.org_log_valdn,
            '原始内部验证集'
        )

        print("")
        print("=" * 80)
        print("结构优化固定数据划分")
        print(
            f"内部训练集：{expected_train_stats['cases']} cases, "
            f"{expected_train_stats['events']} events"
        )
        print(
            f"内部验证集：{expected_val_stats['cases']} cases, "
            f"{expected_val_stats['events']} events"
        )
        print("=" * 80)

        # ========================================================
        # 2. 使用固定原始训练集挖掘资源参数
        # ========================================================
        self.log = copy.deepcopy(self.org_log)
        self.log_train = copy.deepcopy(self.org_log_train)
        self.log_valdn = copy.deepcopy(self.org_log_valdn)

        parameters = spm.StructureParametersMiner.mine_resources(
            self.settings,
            copy.deepcopy(self.org_log_train)
        )

        # mine_resources可能原地修改日志，因此再次恢复
        self.log = copy.deepcopy(self.org_log)
        self.log_train = copy.deepcopy(self.org_log_train)
        self.log_valdn = copy.deepcopy(self.org_log_valdn)

        # ========================================================
        # 3. 定义单个Optuna试验
        # ========================================================
        def objective(trial):
            # ----------------------------------------------------
            # 3.1 每个trial开始前恢复完整的原始日志
            # ----------------------------------------------------
            self.log = copy.deepcopy(self.org_log)
            self.log_train = copy.deepcopy(self.org_log_train)
            self.log_valdn = copy.deepcopy(self.org_log_valdn)

            current_train_stats = get_log_statistics(
                self.log_train,
                f'Trial {trial.number}内部训练集'
            )

            current_val_stats = get_log_statistics(
                self.log_valdn,
                f'Trial {trial.number}内部验证集'
            )

            print("")
            print("=" * 80)
            print(f"[Trial {trial.number}] 数据恢复检查")
            print(
                f"内部训练集：{current_train_stats['cases']} cases, "
                f"{current_train_stats['events']} events"
            )
            print(
                f"内部验证集：{current_val_stats['cases']} cases, "
                f"{current_val_stats['events']} events"
            )
            print("=" * 80)

            # ----------------------------------------------------
            # 3.2 检查案例数、事件数和案例集合
            # ----------------------------------------------------
            if (
                    current_train_stats['cases']
                    != expected_train_stats['cases']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}训练集案例数异常："
                    f"期望{expected_train_stats['cases']}，"
                    f"实际{current_train_stats['cases']}"
                )

            if (
                    current_train_stats['events']
                    != expected_train_stats['events']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}训练集事件数异常："
                    f"期望{expected_train_stats['events']}，"
                    f"实际{current_train_stats['events']}"
                )

            if (
                    current_train_stats['case_ids']
                    != expected_train_stats['case_ids']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}训练集案例集合发生变化"
                )

            if (
                    current_val_stats['cases']
                    != expected_val_stats['cases']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}验证集案例数异常："
                    f"期望{expected_val_stats['cases']}，"
                    f"实际{current_val_stats['cases']}"
                )

            if (
                    current_val_stats['events']
                    != expected_val_stats['events']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}验证集事件数异常："
                    f"期望{expected_val_stats['events']}，"
                    f"实际{current_val_stats['events']}"
                )

            if (
                    current_val_stats['case_ids']
                    != expected_val_stats['case_ids']
            ):
                raise RuntimeError(
                    f"Trial {trial.number}验证集案例集合发生变化"
                )

            # ----------------------------------------------------
            # 3.3 当前trial使用独立的配置副本
            # ----------------------------------------------------
            trial_stg = copy.deepcopy(self.settings)

            trial_stg['alg_manag'] = trial.suggest_categorical(
                'alg_manag',
                self.settings['alg_manag']
            )

            trial_stg['gate_management'] = trial.suggest_categorical(
                'gate_management',
                self.settings['gate_management']
            )

            trial_stg['confidence_threshold'] = (
                trial.suggest_categorical(
                    'confidence_threshold',
                    [10,20,30,40,50]
                )
            )

            trial_stg['laplace_alpha'] = trial.suggest_float(
                'laplace_alpha',
                0.01,
                0.5,
                log=True
            )

            if self.settings['mining_alg'] == 'im':
                trial_stg['im_noise_threshold'] = (
                    trial.suggest_float(
                        'im_noise_threshold',
                        self.settings['im_noise_threshold'][0],
                        self.settings['im_noise_threshold'][1]
                    )
                )

            # ----------------------------------------------------
            # 3.4 创建当前trial的独立输出目录
            # ----------------------------------------------------
            status = STATUS_OK
            exec_times = dict()

            rsp = self._temp_path_redef(
                trial_stg,
                status=status,
                log_time=exec_times
            )

            status = rsp['status']

            if status == STATUS_OK:
                trial_stg = rsp['values']

            trial.set_user_attr(
                'output',
                trial_stg.get('output', 'N/A')
            )

            # ----------------------------------------------------
            # 3.5 挖掘流程结构并提取仿真参数
            # ----------------------------------------------------
            if status == STATUS_OK:
                rsp = self._mine_structure(
                    trial_stg,
                    status=status,
                    log_time=exec_times
                )

                status = rsp['status']

                if status == STATUS_OK:
                    rsp = self._extract_parameters(
                        trial_stg,
                        rsp['values'],
                        copy.deepcopy(parameters),
                        status=status,
                        log_time=exec_times
                    )

                    status = rsp['status']

            # ----------------------------------------------------
            # 3.6 多次仿真并执行ASHA剪枝
            # ----------------------------------------------------
            if status == STATUS_OK:
                repetitions = trial_stg['repetitions']
                current_similarity_sum = 0.0

                try:
                    for step in range(repetitions):
                        single_similarity = (
                            self._simulate_single_step(
                                trial_stg,
                                self.log_valdn,
                                step
                            )
                        )

                        current_similarity_sum += single_similarity

                        intermediate_similarity = (
                                current_similarity_sum / (step + 1)
                        )

                        print(
                            f"[Trial {trial.number}] "
                            f"仿真进度={step + 1}/{repetitions}，"
                            f"本次相似度={single_similarity:.6f}，"
                            f"当前平均相似度="
                            f"{intermediate_similarity:.6f}"
                        )

                        # 使用step + 1，使资源步从1开始
                        trial.report(
                            intermediate_similarity,
                            step=step + 1
                        )

                        if trial.should_prune():
                            self._save_times(
                                exec_times,
                                trial_stg,
                                self.temp_output
                            )

                            print(
                                f"[Trial {trial.number}] "
                                f"在第{step + 1}次仿真后被剪枝，"
                                f"当前平均相似度="
                                f"{intermediate_similarity:.6f}"
                            )

                            raise optuna.exceptions.TrialPruned()

                    final_similarity = (
                            current_similarity_sum / repetitions
                    )

                except optuna.exceptions.TrialPruned:
                    raise

                except Exception as error:
                    print(
                        f"⚠️ Trial {trial.number}仿真失败，"
                        f"相似度记录为0。错误原因：{error}"
                    )
                    traceback.print_exc()
                    final_similarity = 0.0

            else:
                print(
                    f"⚠️ Trial {trial.number}结构挖掘或参数提取失败，"
                    "相似度记录为0"
                )
                final_similarity = 0.0

            # ----------------------------------------------------
            # 3.7 保存完整trial结果
            # ----------------------------------------------------
            self._save_times(
                exec_times,
                trial_stg,
                self.temp_output
            )

            self._define_response(
                trial_stg,
                status,
                [{
                    'sim_val': final_similarity,
                    'metric': 'dl'
                }]
            )

            print(
                f"[Trial {trial.number}] 完整运行结束，"
                f"最终平均相似度={final_similarity:.6f}"
            )

            return final_similarity

        # ========================================================
        # 4. 执行超参数优化
        # ========================================================
        self.study.optimize(
            objective,
            n_trials=self.settings['max_eval']
        )

        # ========================================================
        # 5. 汇总完整试验结果
        # ========================================================
        try:
            print("")
            print("=" * 80)
            print("ASHA优化完成，开始汇总结果")
            print("=" * 80)

            results_dataframe = self.study.trials_dataframe()

            results_dataframe.columns = [
                column.replace(
                    'params_',
                    ''
                ).replace(
                    'user_attrs_',
                    ''
                )
                for column in results_dataframe.columns
            ]

            # 只保留完整运行的trial
            complete_results = results_dataframe[
                (results_dataframe['state'] == 'COMPLETE')
                & (results_dataframe['value'].notna())
                ].copy()

            if complete_results.empty:
                raise RuntimeError(
                    "不存在完整运行成功的trial，无法选择最优参数"
                )

            complete_results['similarity'] = (
                complete_results['value']
            )

            complete_results['loss'] = (
                    1.0 - complete_results['similarity']
            )

            complete_results['status'] = 'ok'

            complete_results = complete_results.sort_values(
                by='similarity',
                ascending=False
            )

            top_k = min(30, len(complete_results))

            print(
                f"\n📊 相似度最高的前{top_k}个完整试验结果"
            )

            columns_to_show = [
                'number',
                'similarity',
                'status',
                'laplace_alpha',
                'confidence_threshold',
                'alg_manag',
                'gate_management'
            ]

            for parameter_name in [
                'im_noise_threshold',
                'epsilon',
                'eta'
            ]:
                if parameter_name in complete_results.columns:
                    columns_to_show.append(parameter_name)

            columns_to_show = [
                column
                for column in columns_to_show
                if column in complete_results.columns
            ]

            print(
                complete_results[
                    columns_to_show
                ].head(top_k).to_string(index=False)
            )

            best_trial = self.study.best_trial

            self.best_parms = copy.deepcopy(
                best_trial.params
            )

            self.best_similarity = float(
                best_trial.value
            )

            self.best_output = best_trial.user_attrs.get(
                'output',
                'N/A'
            )

            print("")
            print("🎯 最佳完整试验结果")
            print(f"   试验编号：{best_trial.number}")
            print(
                f"   相似度："
                f"{self.best_similarity:.6f}"
            )
            print(f"   输出路径：{self.best_output}")
            print(f"   最佳参数：{self.best_parms}")

            complete_number = int(
                (results_dataframe['state'] == 'COMPLETE').sum()
            )

            pruned_number = int(
                (results_dataframe['state'] == 'PRUNED').sum()
            )

            failed_number = int(
                (results_dataframe['state'] == 'FAIL').sum()
            )

            print("")
            print("试验状态统计")
            print(f"   完整试验：{complete_number}")
            print(f"   剪枝试验：{pruned_number}")
            print(f"   失败试验：{failed_number}")
            print("=" * 80)
            print("")

        except Exception as error:
            print(f"❌ 汇总试验结果时发生异常：{error}")
            traceback.print_exc()
            raise

        return self.best_parms

    @staticmethod
    def _fast_count_simulation_events(csv_path):
        """
        流式统计BIMP CSV中的业务活动事件数。

        排除Start/End，与后续read_stats()以及验证日志
        使用完全一致的比较口径。

        不使用pandas，避免超大CSV占用大量内存。
        """
        import csv

        if not os.path.exists(csv_path):
            return 0

        event_count = 0

        with open(
                csv_path,
                'r',
                encoding='utf-8-sig',
                errors='ignore',
                newline=''
        ) as f:

            reader = csv.DictReader(f)

            # 自动寻找活动字段
            possible_task_columns = [
                'task',
                'Activity',
                'activity',
                'Task',
                'element'
            ]

            task_column = None

            for column in possible_task_columns:
                if column in reader.fieldnames:
                    task_column = column
                    break

            if task_column is None:
                raise RuntimeError(
                    f"无法识别BIMP CSV活动列，"
                    f"当前列名: {reader.fieldnames}"
                )

            for row in reader:
                task = str(row.get(task_column, '')).strip()

                if task not in ('Start', 'End'):
                    event_count += 1

        return event_count

    def _simulate_single_step(self, settings, data, rep_index):
        """
        执行单次BIMP仿真。

        如果仿真事件数量明显超过真实验证日志，
        说明当前BPMN很可能存在循环膨胀，
        直接终止当前trial，不再执行DL计算和后续重复。
        """

        args = (settings, rep_index)

        # ============================================================
        # 1. 执行BIMP
        # ============================================================
        self.execute_simulator(args)

        # ============================================================
        # 2. 在真正读取CSV之前进行规模检查
        # ============================================================
        csv_path = os.path.join(
            settings['output'],
            'sim_data',
            settings['file'].split('.')[0]
            + '_'
            + str(rep_index + 1)
            + '.csv'
        )

        sim_rows = self._fast_count_simulation_events(csv_path)
        real_rows = len(data)

        # 通用相对阈值，不针对某一个数据集
        max_ratio = settings.get(
            'max_sim_real_ratio',
            3.0
        )

        ratio = (
            sim_rows / float(real_rows)
            if real_rows > 0
            else float('inf')
        )

        print("")
        print("=" * 70)
        print("[仿真规模预检查]")
        print(f"真实日志事件数: {real_rows}")
        print(f"仿真日志事件数: {sim_rows}")
        print(f"仿真/真实比例: {ratio:.3f}")
        print(f"允许最大比例: {max_ratio:.3f}")
        print("=" * 70)

        # ============================================================
        # 3. 严重膨胀：立即终止当前trial
        # ============================================================
        if ratio > max_ratio:
            print(
                f"⚠️ 当前BPMN产生严重事件膨胀："
                f"{sim_rows} / {real_rows} = {ratio:.2f} 倍。"
            )
            print(
                "⚠️ 当前超参数组合直接判定为无效，"
                "跳过CSV读取、DL计算以及剩余重复仿真。"
            )

            # 可选：立即删除巨大的临时CSV，避免磁盘被塞满
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    print(f"🗑️ 已删除异常大型仿真文件: {csv_path}")
            except Exception as cleanup_error:
                print(
                    f"⚠️ 删除异常仿真文件失败: "
                    f"{cleanup_error}"
                )

            raise RuntimeError(
                f"SIMULATION_SIZE_EXCEEDED: "
                f"sim_rows={sim_rows}, "
                f"real_rows={real_rows}, "
                f"ratio={ratio:.3f}, "
                f"limit={max_ratio:.3f}"
            )

        # ============================================================
        # 4. 规模正常，再真正加载CSV
        # ============================================================
        sim_log = self.read_stats(args)

        # ============================================================
        # 5. 正常计算DL
        # ============================================================
        eval_res = self.evaluate_logs(
            (settings, data, sim_log)
        )

        return eval_res[0]['sim_val']
    # 重定义临时路径
    @timeit(rec_name='PATH_DEF')  # 计时装饰器
    @Decorators.safe_exec  # 安全执行装饰器
    def _temp_path_redef(self, settings, **kwargs) -> None:
        """为当前试验重定义输出路径"""
        # 创建唯一输出目录
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())

        # 如果使用'repair'算法管理,设置对齐文件路径
        if settings['alg_manag'] == 'repair':
            settings['aligninfo'] = os.path.join(settings['output'], 'CaseTypeAlignmentResults.csv')
            settings['aligntype'] = os.path.join(settings['output'], 'AlignmentStatistics.csv')

        # 创建输出目录及子目录
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
            os.makedirs(os.path.join(settings['output'], 'sim_data'))
        # 将训练日志转为XES格式供外部工具使用
        xes.XesWriter(self.log_train, settings)
        return settings

    # 挖掘流程结构
    @timeit(rec_name='MINING_STRUCTURE')
    @Decorators.safe_exec
    def _mine_structure(self, settings, **kwargs) -> None:
        """执行流程结构挖掘"""
        # 初始化结构挖掘器
        structure_miner = sm.StructureMiner(settings, self.log_train)
        # 执行挖掘流水线
        structure_miner.execute_pipeline()

        # 检查是否成功挖掘
        if structure_miner.is_safe:
            return [structure_miner.bpmn, structure_miner.process_graph]  # 返回BPMN模型和流程图
        else:
            raise RuntimeError('Mining Structure error')

    # 提取流程参数
    @timeit(rec_name='EXTRACTING_PARAMS')
    @Decorators.safe_exec
    def _extract_parameters(self, settings, structure, parameters, **kwargs) -> None:
        """提取流程的时间/资源等参数"""
        bpmn, process_graph = structure

        # 初始化参数提取器
        p_extractor = spm.StructureParametersMiner(
            self.log_train, bpmn, process_graph, settings)

        # 计算验证集的实例数
        num_inst = len(self.log_valdn.caseid.unique())
        # 获取验证集的最小开始时间
        start_time = self.log_valdn.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        # 提取参数
        p_extractor.extract_parameters(num_inst, start_time, parameters['resource_pool'])

        # 检查参数提取是否成功
        if p_extractor.is_safe:
            # 合并资源池参数和提取的参数
            parameters = {**parameters, **p_extractor.parameters}

            # 将参数写入BIMP兼容的XML文件
            xml.print_parameters(
                os.path.join(settings['output'], settings['file'].split('.')[0] + '.bpmn'),
                os.path.join(settings['output'], settings['file'].split('.')[0] + '.bpmn'),
                parameters)

            # 预处理验证日志(为仿真评估做准备)
            self.log_valdn.rename(columns={'user': 'resource'}, inplace=True)
            self.log_valdn['source'] = 'log'
            self.log_valdn['run_num'] = 0
            self.log_valdn['role'] = 'SYSTEM'
            self.log_valdn = self.log_valdn[~self.log_valdn.task.isin(['Start', 'End'])]
        else:
            raise RuntimeError('Parameters extraction error')

    # 执行流程仿真和评估
    @timeit(rec_name='SIMULATION_EVAL')
    @Decorators.safe_exec
    def _simulate(self, settings, data, **kwargs) -> list:
        """执行流程仿真并评估结果"""

        # 进度条更新函数 (用于异步操作)
        def pbar_async(p, msg):
            """为异步操作显示进度条"""
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = reps - p._number_left
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        # 设置仿真重复次数
        reps = settings['repetitions']
        # 根据CPU核心数和重复次数确定工作进程数
        cpu_count = multiprocessing.cpu_count()
        w_count = reps if reps <= cpu_count else cpu_count

        # 创建进程池
        pool = Pool(processes=w_count)

        # 1. 执行仿真
        args = [(settings, rep) for rep in range(reps)]
        p = pool.map_async(self.execute_simulator, args)
        pbar_async(p, 'simulating:')  # 显示仿真进度条

        # 2. 读取仿真日志
        p = pool.map_async(self.read_stats, args)
        pbar_async(p, 'reading simulated logs:')  # 显示读取进度条

        # 3. 评估结果
        args = [(settings, data, log) for log in p.get()]

        # 根据日志大小选择评估方式(大日志使用单进程+进度条)
        if len(self.log_valdn.caseid.unique()) > 1000:
            pool.close()
            results = [self.evaluate_logs(arg) for arg in tqdm(args, 'evaluating results:')]
            sim_values = list(itertools.chain(*results))
        else:
            p = pool.map_async(self.evaluate_logs, args)
            pbar_async(p, 'evaluating results:')  # 显示评估进度条
            pool.close()
            sim_values = list(itertools.chain(*p.get()))

        return sim_values

    # 读取仿真统计结果
    @staticmethod
    def read_stats(args):
        """读取单次仿真的结果日志"""

        def read(settings, rep):
            """读取仿真生成的CSV日志"""
            # 准备日志读取设置
            m_settings = dict()
            m_settings['output'] = settings['output']
            m_settings['file'] = settings['file']
            column_names = {'resource': 'user'}  # 列名映射
            m_settings['read_options'] = settings['read_options']
            m_settings['read_options']['timeformat'] = '%Y-%m-%d %H:%M:%S.%f'
            m_settings['read_options']['column_names'] = column_names

            # 读取仿真日志
            temp = lr.LogReader(
                os.path.join(m_settings['output'], 'sim_data',
                             m_settings['file'].split('.')[0] + '_' + str(rep + 1) + '.csv'),
                m_settings['read_options'],
                verbose=False)

            # 数据预处理
            temp = pd.DataFrame(temp.data)
            temp.rename(columns={'user': 'resource'}, inplace=True)
            temp['role'] = temp['resource']
            temp['source'] = 'simulation'
            temp['run_num'] = rep + 1
            temp = temp[~temp.task.isin(['Start', 'End'])]  # 移除开始/结束事件
            return temp

        return read(*args)

    # 评估日志相似度
    @staticmethod
    def evaluate_logs(args):
        """评估实际日志与仿真日志的相似度"""

        def evaluate(settings, data, sim_log):

            """使用编辑距离评估日志相似度"""
            rep = sim_log.iloc[0].run_num  # 获取当前重复序号

            # --- [新增] 中文调试打印：对比列名 ---
            # 只在每次试验的第1个run打印，避免刷屏 (rep 1, 2, 3...)
            # 或者每次都打印也没关系，反正 ASHE 逐步汇报
            print(f"\n---  {rep}) ---")

            # 检查关键列 'task' 是否存在 (因为DL算法通常依赖 task 列)
            if 'task' not in data.columns:
                print("❌ 警告: 真实日志中缺少 'task' 列，这可能导致评估分数异常！")
            if 'task' not in sim_log.columns:
                print("❌ 警告: 仿真日志中缺少 'task' 列，这可能导致评估分数异常！")

            # 打印数据行数，看看是否为空
            print(f"3. 真实日志行数: {len(data)}")
            print(f"4. 仿真日志行数: {len(sim_log)}")
            print("-" * 40)
            # -----------------------------------

            sim_values = list()

            # 初始化相似度评估器
            evaluator = sim.SimilarityEvaluator(
                data,  # 实际日志数据
                sim_log,  # 仿真日志数据
                settings,  # 设置参数
                max_cases=1000)  # 最大评估案例数

            # 使用DL(编辑距离)方法评估相似度
            evaluator.measure_distance('dl')

            # 记录评估结果
            sim_values.append({**{'run_num': rep}, **evaluator.similarity})
            return sim_values

        return evaluate(*args)
    # 调用BIMP执行流程仿真
    @staticmethod
    def execute_simulator(args):
        """调用BIMP工具执行流程仿真（增强版）"""

        def sim_call(settings, rep):
            """通过命令行调用BIMP仿真器"""
            import sys

            # 构建输入输出路径
            bpmn_path = os.path.join(
                settings['output'],
                settings['file'].split('.')[0] + '.bpmn'
            )

            output_csv = os.path.join(
                settings['output'],
                'sim_data',
                settings['file'].split('.')[0] + '_' + str(rep + 1) + '.csv'
            )

            # 确保输出目录存在
            output_dir = os.path.dirname(output_csv)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"  创建输出目录: {output_dir}")

            # 检查BPMN文件是否存在
            if not os.path.exists(bpmn_path):
                error_msg = f"BPMN文件不存在: {bpmn_path}"
                print(f"  ❌ {error_msg}")
                raise FileNotFoundError(error_msg)

            # 检查BPMN文件大小
            bpmn_size = os.path.getsize(bpmn_path)
            if bpmn_size == 0:
                error_msg = f"BPMN文件为空: {bpmn_path}"
                print(f"  ❌ {error_msg}")
                raise ValueError(error_msg)

            print(f"\n  {'=' * 60}")
            print(f"  BIMP仿真 - 重复 {rep + 1}")
            print(f"  {'=' * 60}")
            print(f"  输入BPMN: {bpmn_path} ({bpmn_size} bytes)")
            print(f"  输出CSV: {output_csv}")
            print(f"  BIMP路径: {settings['bimp_path']}")

            # 构建命令行参数
            args = [
                'java', '-jar', settings['bimp_path'],
                bpmn_path,
                '-csv', output_csv
            ]

            print(f"  执行命令: {' '.join(args)}")

            try:
                # 执行BIMP（捕获输出）
                result = subprocess.run(
                    args,
                    check=False,  # 不自动抛出异常，手动检查
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600  # 10分钟超时
                )

                # 打印BIMP输出
                if result.stdout:
                    print(f"  BIMP标准输出:\n{result.stdout}")

                if result.stderr:
                    print(f"  BIMP错误输出:\n{result.stderr}")

                # 检查返回码
                if result.returncode != 0:
                    error_msg = (
                        f"BIMP执行失败 (返回码: {result.returncode})\n"
                        f"标准输出: {result.stdout}\n"
                        f"错误输出: {result.stderr}"
                    )
                    print(f"  ❌ {error_msg}")
                    raise RuntimeError(error_msg)

                # 检查输出文件是否生成
                if not os.path.exists(output_csv):
                    error_msg = (
                        f"BIMP执行成功但未生成输出文件: {output_csv}\n"
                        f"可能原因:\n"
                        f"  1. BPMN模型中没有仿真参数\n"
                        f"  2. BIMP版本不兼容\n"
                        f"  3. 仿真过程中发生错误但未报告\n"
                        f"BIMP输出: {result.stdout}"
                    )
                    print(f"  ❌ {error_msg}")

                    # 尝试列出输出目录内容
                    if os.path.exists(output_dir):
                        files = os.listdir(output_dir)
                        print(f"  输出目录 {output_dir} 内容: {files}")

                    raise FileNotFoundError(error_msg)

                # 检查输出文件大小
                csv_size = os.path.getsize(output_csv)
                if csv_size == 0:
                    error_msg = f"输出CSV文件为空: {output_csv}"
                    print(f"  ⚠️ {error_msg}")
                    raise ValueError(error_msg)

                print(f"  ✅ 仿真成功，生成文件: {output_csv} ({csv_size} bytes)")
                print(f"  {'=' * 60}\n")

            except subprocess.TimeoutExpired:
                error_msg = f"BIMP执行超时（超过300秒）"
                print(f"  ❌ {error_msg}")
                raise TimeoutError(error_msg)

            except Exception as e:
                print(f"  ❌ BIMP执行异常: {type(e).__name__}: {str(e)}")
                raise

        sim_call(*args)

    # 保存各阶段执行时间
    @staticmethod
    def _save_times(times, settings, temp_output):
        """将各阶段执行时间保存到CSV文件"""
        if times:
            # 添加输出路径信息
            times = [{**{'output': settings['output']}, **times}]
            log_file = os.path.join(temp_output, 'execution_times.csv')

            # 创建文件(如果不存在)
            if not os.path.exists(log_file):
                open(log_file, 'w').close()

            # 写入数据(追加或创建新文件)
            if os.path.getsize(log_file) > 0:
                sup.create_csv_file(times, log_file, mode='a')
            else:
                sup.create_csv_file_header(times, log_file)

    # 定义试验响应
    def _define_response(self, settings, status, sim_values, **kwargs) -> None:
        """构建贝叶斯优化的响应字典"""
        response = dict()
        measurements = list()

        # 记录基础参数
        data = {
            'alg_manag': settings['alg_manag'],
            'gate_management': settings['gate_management'],
            'output': settings['output'],
            # --- [新增] 记录选定的参数值 ---
            'confidence_threshold': settings.get('confidence_threshold', 30),
            'laplace_alpha': settings.get('laplace_alpha', 1.0)
        }

        # 记录算法特定参数
        if settings['mining_alg'] in ['sm1', 'sm3']:
            data['epsilon'] = settings['epsilon']
            data['eta'] = settings['eta']
        elif settings['mining_alg'] == 'im':
            # 处理 'im' 算法特定的参数
            data['im_noise_threshold'] = settings['im_noise_threshold']
        else:
            raise ValueError(settings['mining_alg'])

        # 计算平均相似度
        similarity = 0.0
        response['output'] = settings['output']

        # 处理成功状态
        if status == STATUS_OK:
            similarity = np.mean([x['sim_val'] for x in sim_values])  # 计算平均相似度
            loss = (1 - similarity)  # 损失函数(1-相似度)
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL

            # ⭐ 中文打印:每次 trial 的总体结果
            mining_alg = settings.get('mining_alg', 'unknown')
            if mining_alg == 'im':
                im_thr = settings.get('im_noise_threshold', None)
                if isinstance(im_thr, float):
                    thr_str = f"{im_thr:.4f}"
                else:
                    thr_str = str(im_thr)
                print(
                    f"[试验] 算法=im | 噪声阈值(im_noise_threshold)={thr_str} | "
                    f"相似度={similarity:.4f} | 损失值={loss:.4f} | 状态={response['status']}"
                )
            elif mining_alg in ['sm1', 'sm3']:
                eps = settings.get('epsilon', None)
                eta = settings.get('eta', None)
                eps_str = f"{eps:.4f}" if isinstance(eps, float) else str(eps)
                eta_str = f"{eta:.4f}" if isinstance(eta, float) else str(eta)
                print(
                    f"[试验] 算法={mining_alg} | epsilon={eps_str} | eta={eta_str} | "
                    f"相似度={similarity:.4f} | 损失值={loss:.4f} | 状态={response['status']}"
                )
            else:
                print(
                    f"[试验] 算法={mining_alg} | "
                    f"相似度={similarity:.4f} | 损失值={loss:.4f} | 状态={response['status']}"
                )

            # 记录每次仿真的测量结果(逐 run)
            for sim_val in sim_values:
                measurements.append({
                    **{'similarity': sim_val['sim_val'],
                       'sim_metric': sim_val['metric'],
                       'status': response['status']},
                    **data})
        # 处理失败状态
        else:
            response['status'] = status
            measurements.append({
                **{'similarity': 0,
                   'sim_metric': 'dl',
                   'status': response['status']},
                **data})

        # 保存测量结果到CSV
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)

        return response

    # 分割日志为训练集和验证集
    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        按时间线分割事件日志,用于交叉验证
        Parameters:
            size : 验证集比例
            one_ts : 是否使用单一时间戳
        """
        # 初始化日志分割器
        splitter = ls.LogSplitter(self.log.data)

        # 使用 timeline_trace（案例级分割），不丢弃跨分割点的案例
        train, valdn = splitter.split_log('timeline_trace', size, one_ts)

        # 设置时间戳字段
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)

        # 如果日志过大,对训练集采样
        train = self._sample_log(train)

        # 保存分区结果
        self.log_valdn = valdn.sort_values(key, ascending=True).reset_index(drop=True)
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))

        # 日志采样(大日志时减少数据量)

    @staticmethod
    def _sample_log(train):
        """当训练日志过大时进行采样"""

        def sample_size(p_size, c_level, c_interval):
            """
            计算样本大小(基于置信水平和区间)
            Parameters:
                p_size : 总体大小
                c_level : 置信水平(如95)
                c_interval : 置信区间(如3表示±3%)
            """
            # 置信水平对应的Z值
            c_level_constant = {50: .67, 68: .99, 90: 1.64, 95: 1.96, 99: 2.57}
            Z = c_level_constant[c_level]
            p = 0.5  # 最大方差概率
            e = c_interval / 100.0  # 误差幅度

            # 计算初始样本大小
            n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)
            # 根据总体大小调整样本量
            n = n_0 / (1 + ((n_0 - 1) / float(p_size)))
            return int(math.ceil(n))

        # 获取所有案例ID
        cases = list(train.caseid.unique())

        # 如果案例数超过1000,进行采样
        if len(cases) > 1000:
            # 计算95%置信水平、3%置信区间所需的样本量
            sample_sz = sample_size(len(cases), 99, 3)
            # 随机抽样
            scases = random.sample(cases, sample_sz)
            train = train[train.caseid.isin(scases)]

        return train
