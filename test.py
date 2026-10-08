import pm4py
import external_tools.SimilarityEvaluator as sim
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import readers.log_reader as lr
import copy
import support_modules.common as cm

class LogEvaluationPipeline:
    """
    增强版日志评估流水线，整合了日志预处理功能
    """

    def __init__(self):
        self.is_safe = True
        self.sim_values = list()
        self.metrics = ['day_hour_emd', 'log_mae', 'mae']
        self.default_settings = {
            'read_options': {
                'one_timestamp': False,  # 是否只使用一个时间戳
                'timeformat': '%Y-%m-%d %H:%M:%S',
                'column_names': {
                    'CaseID': 'caseid',
                    'Activity': 'task',
                    'Resource': 'user',
                    'StartTime': 'start_timestamp',
                    'EndTime': 'end_timestamp'
                },
                'filter_d_attrib': True
            }
        }

        # 初始化缺失的属性
        self.parms = {'gl': {}}  # 添加这行初始化代码
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载并预处理CSV日志"""
        print(f"加载CSV文件: {file_path}")
        df = pd.read_csv(file_path)
        df = df.rename(columns=self.default_settings['read_options']['column_names'])
        df = self._prepare_data(df, is_xes=False)
        return df

    def _prepare_data(self, df: pd.DataFrame, is_xes: bool) -> pd.DataFrame:
        """增强版数据预处理，整合时间戳处理逻辑"""
        df = df.copy()
        # CSV日志处理
        time_cols = {
            'timestamp': 'start_timestamp',
            'start_time': 'start_timestamp',
            'end_time': 'end_timestamp'
        }
        for old, new in time_cols.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        # 如果没有end_timestamp，使用start_timestamp作为end_timestamp
        if 'end_timestamp' not in df.columns and 'start_timestamp' in df.columns:
            df['end_timestamp'] = df['start_timestamp']

        # 确保必须的时间戳列存在
        required_cols = ['start_timestamp', 'end_timestamp'] if not self.default_settings['read_options'][
            'one_timestamp'] else ['end_timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺失必要时间戳列: {col}")

        # 时间戳格式转换
        for col in ['start_timestamp', 'end_timestamp']:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(
                    df[col],
                    format=self.default_settings['read_options']['timeformat'],
                    utc=True,
                    errors='coerce'
                )


        # 添加Start/End事件
        return df

    def _match_start_complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """匹配XES日志中的start和complete事件，创建完整的时间范围"""
        processed = []

        for caseid, group in df.groupby('caseid'):
            trace = group.to_dict('records')
            temp_trace = []

            for i in range(0, len(trace) - 1):
                if trace[i]['event_type'] == 'start':
                    c_task_name = trace[i]['task']
                    remaining = trace[i + 1:]

                    # 查找匹配的complete事件
                    complete_event = next(
                        (event for event in remaining
                         if (event['task'] == c_task_name and event['event_type'] == 'complete')),
                        None
                    )

                    if complete_event:
                        temp_trace.append({
                            'caseid': caseid,
                            'task': trace[i]['task'],
                            'user': trace[i]['user'],
                            'start_timestamp': trace[i]['timestamp'],
                            'end_timestamp': complete_event['timestamp']
                        })
                    else:
                        # 如果没有找到complete事件，使用start时间作为end时间
                        temp_trace.append({
                            'caseid': caseid,
                            'task': trace[i]['task'],
                            'user': trace[i]['user'],
                            'start_timestamp': trace[i]['timestamp'],
                            'end_timestamp': trace[i]['timestamp']
                        })

            processed.extend(temp_trace)

        return pd.DataFrame(processed)
    def evaluate_logs(self,
                      real_log,
                      simulated_log,
                      rep_num: int = 0) -> List[Dict]:
        """评估日志相似度"""
        print(f"\n===== 评估第{rep_num}次 =====")

        try:
            # 1. 加载和预处理日志
            if isinstance(real_log, (str, Path)):
                real_log = self._load_and_preprocess(real_log, is_real_log=True)
                if real_log is None:
                    raise ValueError("真实日志加载失败")

            if isinstance(simulated_log, (str, Path)):
                simulated_log = self._load_and_preprocess(simulated_log, is_real_log=False)
                if simulated_log is None:
                    raise ValueError("模拟日志加载失败")
            log = copy.deepcopy(real_log)

            # 过滤掉开始/结束任务
            original_size = len(log)
            log = log[~log.task.isin(['Start', 'End'])]
            filtered_size = len(log)
            print(f"过滤后真实日志记录数: {filtered_size} (移除了 {original_size - filtered_size} 条Start/End记录)")

            log['source'] = 'log'
            log.rename(columns={'user': 'resource'}, inplace=True)
            log['caseid'] = log['caseid'].astype(str)
            log['caseid'] = 'Case' + log['caseid']

            # 检查日志中是否包含必要列
            required_columns = ['caseid', 'task', 'start_timestamp', 'end_timestamp']
            for col in required_columns:
                if col not in log.columns:
                    print(f"[错误] 真实日志缺少必要列: {col}")
                    return []
                if col not in simulated_log.columns:
                    print(f"[错误] 模拟日志缺少必要列: {col}")
                    return []
            # 2. 初始化评估器
            evaluator = sim.SimilarityEvaluator(
                log_data=real_log,
                simulation_data=simulated_log,
                settings=self.default_settings,
                max_cases=1000,
                dtype='log'
            )

            # 3. 执行评估
            results = []
            for metric in self.metrics:
                try:
                    evaluator.measure_distance(metric)
                    results.append({
                        'run_num': rep_num,
                        'metric': metric,
                        'sim_val': evaluator.similarity['sim_val']
                    })
                    print(f"{metric}: {evaluator.similarity['sim_val']:.4f}")
                except Exception as e:
                    print(f"[WARN] {metric}评估失败: {str(e)}")

            self.sim_values.extend(results)
            return results

        except Exception as e:
            print(f"[ERROR] 评估失败: {str(e)}")
            traceback.print_exc()
            return []

    def run_pipeline(self):
        """完整工作流"""
        print("=" * 50 + "\n日志评估启动\n" + "=" * 50)
        real_log_path = "data/input/cvs_pharmacy.xes"
        simulated_log = "data/input/gen_cvs_pharmacy_1.csv"
        sim_log = self._load_csv(simulated_log)
        column_names = {'Case ID': 'caseid',
                        'Activity': 'task',
                        'lifecycle:transition': 'event_type',
                        'Resource': 'user'}
        self.log_test = pd.DataFrame()
        self.log_train = pd.DataFrame()
        self.parms['gl']['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': False,
            'filter_d_attrib': True}

        # Event log reading
        self.log = lr.LogReader(real_log_path, self.parms['gl']['read_options'])
        # 检查读取的日志是否为空
        if not hasattr(self.log, 'data') or not self.log.data or len(self.log.data) == 0:
            print("[严重错误] 读取的事件日志数据为空！")
            return False

        print(f"原始日志记录数: {len(self.log.data)}")

        # 正确统计案例数（处理字典列表结构）
        caseids = set(event['caseid'] for event in self.log.data)
        print(f"原始日志案例数: {len(caseids)}")

        # Time splitting 80-20
        self._split_timeline(0.8, False)

        try:
            self.evaluate_logs(
                real_log=self.log_test,
                simulated_log=sim_log,
            )
            self._export_results()
        except Exception as e:
            print(f"流程错误: {str(e)}")
        finally:
            print("\n" + "=" * 50 + "\n评估结束\n" + "=" * 50)
    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        按时间分割事件日志数据框以执行分割验证。
        """
        print("\n===== 分割时间线 =====")
        print(f"分割比例: {size * 100}% 训练 / {(1 - size) * 100}% 测试")

        key = 'end_timestamp' if one_ts else 'start_timestamp'

        # 检查日志数据是否有效
        if not hasattr(self.log, 'data') or not self.log.data:
            print("[错误] 日志数据为空或无效，无法分割")
            return

        print(f"分割前原始日志记录数: {len(self.log.data)}")

        # 正确统计案例数（处理字典列表结构）
        try:
            # 确保使用正确的方法统计案例数
            caseids = set(event['caseid'] for event in self.log.data)
            print(f"分割前原始日志案例数: {len(caseids)}")
        except KeyError:
            print("[警告] 日志数据中没有找到'caseid'字段，尝试使用'case_id'")
            try:
                caseids = set(event['case_id'] for event in self.log.data)
                print(f"分割前原始日志案例数: {len(caseids)}")
            except KeyError:
                print("[错误] 日志数据中找不到有效的案例ID字段")
                return

        try:
            # Split log data
            train, test = cm.split_log(self.log, one_ts, size)
        except Exception as e:
            print(f"分割日志时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # 检查分割结果 - 处理可能的DataFrame或字典列表格式
        try:
            # 处理测试集统计
            if isinstance(test, pd.DataFrame):
                test_records = len(test)
                test_cases = test['caseid'].nunique()
            else:  # 如果是字典列表
                test_records = len(test)
                test_cases = len(set(event['caseid'] for event in test))

            # 处理训练集统计
            if isinstance(train, pd.DataFrame):
                train_records = len(train)
                train_cases = train['caseid'].nunique()
            else:  # 如果是字典列表
                train_records = len(train)
                train_cases = len(set(event['caseid'] for event in train))

            print(f"训练集记录数: {train_records}")
            print(f"训练集案例数: {train_cases}")
            print(f"测试集记录数: {test_records}")
            print(f"测试集案例数: {test_cases}")
        except Exception as e:
            print(f"统计分割结果时出错: {str(e)}")
            return

        if test_records == 0 or test_cases == 0:
            print("[错误] 测试集为空或没有案例!")
            return

        if train_records == 0 or train_cases == 0:
            print("[错误] 训练集为空或没有案例!")
            return

        # 确保test和train都是DataFrame格式
        if not isinstance(test, pd.DataFrame):
            test = pd.DataFrame(test)
        if not isinstance(train, pd.DataFrame):
            train = pd.DataFrame(train)

        # 排序并设置索引
        try:
            self.log_test = (test.sort_values(key, ascending=True).reset_index(drop=True))
            print('测试日志中的实例数: {}'.format(self.log_test['caseid'].nunique()))

            self.log_train = copy.deepcopy(self.log)
            # 确保设置的是DataFrame格式的数据
            train_df = train.sort_values(key, ascending=True).reset_index(drop=True)
            self.log_train.set_data(train_df.to_dict('records'))
            print('训练日志中的实例数: {}'.format(train_df['caseid'].nunique()))
        except KeyError as e:
            print(f"[错误] 日志中缺少必要的列: {str(e)}")
        except Exception as e:
            print(f"处理分割结果时出错: {str(e)}")
    def _export_results(self):
        """结果导出"""
        Path("data/results").mkdir(parents=True, exist_ok=True)
        if not self.sim_values:
            print("警告: 无有效评估结果")
            return

        df = pd.DataFrame(self.sim_values)
        df.to_csv("data/results/evaluation_results.csv", index=False)
        print("\n评估结果汇总:")
        print(df.groupby('metric')['sim_val'].describe())


if __name__ == "__main__":
    LogEvaluationPipeline().run_pipeline()