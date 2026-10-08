# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import load_model
from core_modules.times_allocator import entities as en
import warnings

from datetime import timedelta
import uuid
from tqdm import tqdm

from pickle import load
from enum import Enum


class InstanceState(Enum):
    WAITING = 1
    INEXECUTION = 2
    COMPLETE = 3


class DualIntercasesPredictor():
    """
    双阶段（处理和等待）预测器，用于事件驱动仿真
    """

    def __init__(self, model_path, parms):
        """构造函数"""
        print("初始化 DualIntercasesPredictor...")
        self.execution_state = None
        self.queue = None
        self.ac_dict = None
        self.rl_dict = None
        self.sequences = None
        self.end_inter_scaler = None
        self.inter_scaler = None
        self.scaler = None

        # 加载两个模型
        self.g1, self.first_session, self.proc_model_path = self._load_models(model_path[0])
        self.g2, self.second_session, self.wait_model_path = self._load_models(model_path[1])
        self.parms = parms

        # 模型输入特征维度
        self.n_feat_proc = (self.proc_model_path.get_layer('features').output_shape[0][2])
        self.n_feat_wait = (self.wait_model_path.get_layer('features').output_shape[0][2])

    def _load_models(self, path):
        """加载Keras模型到独立图和Session"""
        print(f"加载模型: {path}")

        graph = tf.Graph()

        with graph.as_default():
            session = tf.compat.v1.Session()

            with session.as_default():
                try:
                    print(f"   [方法1] 尝试不编译加载...")
                    model = load_model(path, compile=False)
                    return graph, session, model

                except Exception as e1:

                    try:
                        print(f"   [方法2] 尝试自定义对象加载...")

                        from tensorflow.keras.optimizers import Adam, SGD, RMSprop

                        class LegacyAdam(Adam):
                            def __init__(self, *args, weight_decay=None, **kwargs):
                                if weight_decay is not None:
                                    print(f"      ⚠️  忽略废弃参数: weight_decay={weight_decay}")
                                kwargs.pop('weight_decay', None)
                                super().__init__(*args, **kwargs)

                        class LegacySGD(SGD):
                            def __init__(self, *args, weight_decay=None, **kwargs):
                                kwargs.pop('weight_decay', None)
                                super().__init__(*args, **kwargs)

                        class LegacyRMSprop(RMSprop):
                            def __init__(self, *args, weight_decay=None, **kwargs):
                                kwargs.pop('weight_decay', None)
                                super().__init__(*args, **kwargs)

                        custom_objects = {
                            'Adam': LegacyAdam,
                            'SGD': LegacySGD,
                            'RMSprop': LegacyRMSprop,
                        }

                        model = load_model(path, custom_objects=custom_objects, compile=False)
                        print(f"   ✅ 模型加载成功（自定义对象模式）")
                        return graph, session, model

                    except Exception as e2:
                        print(f"   ❌ 方法2失败: {str(e2)[:100]}")

                        try:
                            print(f"   [方法3] 尝试重建模型架构...")

                            import h5py
                            import json
                            from tensorflow.keras.models import model_from_json

                            with h5py.File(path, 'r') as f:
                                if 'model_config' in f.attrs:
                                    model_config = f.attrs['model_config']

                                    if isinstance(model_config, bytes):
                                        model_config = model_config.decode('utf-8')

                                    config_dict = json.loads(model_config)
                                    model = model_from_json(json.dumps(config_dict))
                                    model.load_weights(path)

                                    print(f"   ✅ 模型架构重建成功（仅权重）")
                                    return graph, session, model
                                else:
                                    raise ValueError("模型文件中缺少 model_config")

                        except Exception as e3:
                            print(f"   ❌ 方法3失败: {str(e3)[:100]}")

                            print(f"   [方法4] 尝试强制加载（忽略所有错误）...")

                            try:
                                import warnings
                                warnings.filterwarnings('ignore')

                                tf.compat.v1.disable_eager_execution()

                                model = load_model(path, compile=False)
                                print(f"   ✅ 强制加载成功")
                                return graph, session, model

                            except Exception as e4:
                                print(f"   ❌ 所有方法均失败")
                                print(f"\n" + "=" * 60)
                                print(f"错误汇总:")
                                print(f"  方法1（不编译）: {str(e1)[:80]}")
                                print(f"  方法2（自定义对象）: {str(e2)[:80]}")
                                print(f"  方法3（重建架构）: {str(e3)[:80]}")
                                print(f"  方法4（强制加载）: {str(e4)[:80]}")
                                print(f"=" * 60)

                                session.close()

                                raise RuntimeError(
                                    f"\n❌ 无法加载模型: {path}\n\n"
                                    f"可能的原因:\n"
                                    f"  1. TensorFlow 版本不兼容\n"
                                    f"  2. 模型使用了已废弃的优化器参数\n\n"
                                    f"解决方案:\n"
                                    f"  方案A（推荐）：重新保存模型\n"
                                    f"    在原训练环境运行:\n"
                                    f"    >>> model = load_model('{path}')\n"
                                    f"    >>> model.save('{path}', include_optimizer=False)\n\n"
                                    f"  方案B：降级 TensorFlow\n"
                                    f"    pip install tensorflow==2.10.0\n\n"
                                    f"  方案C：使用 SavedModel 格式\n"
                                    f"    model.save('model_dir/')  # 不用 .h5 后缀\n"
                                )

    def predict(self, sequences, iarr):
        """主预测入口：生成事件驱动仿真结果"""
        print("\n" + "=" * 80)
        print("🚀 开始 PREDICT 流程")
        print("=" * 80)

        # 🔍 调试：检查输入数据
        print(f"\n📊 输入数据检查:")
        print(f"   - sequences 类型: {type(sequences)}")
        print(f"   - sequences 形状: {sequences.shape if hasattr(sequences, 'shape') else 'N/A'}")
        print(f"   - sequences 长度: {len(sequences) if hasattr(sequences, '__len__') else 'N/A'}")
        if isinstance(sequences, pd.DataFrame):
            print(f"   - sequences 列: {sequences.columns.tolist()}")
            print(f"   - sequences 唯一案例数: {sequences['caseid'].nunique() if 'caseid' in sequences.columns else 'N/A'}")
            print(f"   - sequences 前5行:")
            print(sequences.head())

        print(f"\n   - iarr 类型: {type(iarr)}")
        print(f"   - iarr 形状: {iarr.shape if hasattr(iarr, 'shape') else 'N/A'}")
        print(f"   - iarr 长度: {len(iarr) if hasattr(iarr, '__len__') else 'N/A'}")
        if isinstance(iarr, pd.DataFrame):
            print(f"   - iarr 列: {iarr.columns.tolist()}")
            print(f"   - iarr 前5行:")
            print(iarr.head())

        tmodel = '_diapr' if self.parms['all_r_pool'] else '_dispr'
        metadata_file = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0] + tmodel + '_meta.json')

        # 加载元数据
        print(f"\n📁 加载元数据文件: {metadata_file}")
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                self.ac_index = data['ac_index']
                self.index_ac = {v: k for k, v in self.ac_index.items()}
                n_size = data['n_size']
                rl_task = pd.DataFrame(data['roles_table'])
                rl_table = pd.DataFrame([{'role_name': item[0],
                                          'size': len(item[1]),
                                          'role_index': num}
                                         for num, item in enumerate(data['roles'].items())])
                pr_act_initial = int(
                    round(data['inter_mean_states']['wip']))
                init_states = data['inter_mean_states']['tasks']
            print("   ✅ 元数据加载完成")
            print(f"   - ac_index 活动数: {len(self.ac_index)}")
            print(f"   - ac_index 内容: {self.ac_index}")
            print(f"   - rl_table 角色数: {len(rl_table)}")
            print(f"   - 初始在制品数: {pr_act_initial}")

        # 加载标准化器
        s_path = os.path.join(self.parms['times_gen_path'], self.parms['file'].split('.')[0] + tmodel + '_scaler.pkl')
        i_s_path = os.path.join(self.parms['times_gen_path'],
                                self.parms['file'].split('.')[0] + tmodel + '_inter_scaler.pkl')
        e_s_path = os.path.join(self.parms['times_gen_path'],
                                self.parms['file'].split('.')[0] + tmodel + '_end_inter_scaler.pkl')

        print(f"\n🔧 加载标准化器...")
        self.scaler = load(open(s_path, 'rb'))
        self.inter_scaler = load(open(i_s_path, 'rb'))
        self.end_inter_scaler = load(open(e_s_path, 'rb'))
        print("   ✅ 标准化器加载完成")

        # 🔍 第1步：过滤超短案例
        print(f"\n" + "=" * 80)
        print("📋 步骤1: 过滤超短案例")
        print("=" * 80)
        extra_short, sequences = self._filter_extra_short_cases(sequences)

        # 🔍 第2步：分离短案例和长案例的到达时间
        print(f"\n" + "=" * 80)
        print("📋 步骤2: 分离短案例和长案例到达时间")
        print("=" * 80)
        short_iarr, long_iarr = self._filter_short_iarr(extra_short, iarr)

        # 🔍 第3步：编码序列
        print(f"\n" + "=" * 80)
        print("📋 步骤3: 编码序列")
        print("=" * 80)
        self.sequences, num_elements = self._encode_secuences(sequences, self.ac_index, rl_task, rl_table)

        # 🔍 第4步：初始化角色、活动、队列、执行状态
        print(f"\n" + "=" * 80)
        print("📋 步骤4: 初始化系统组件")
        print("=" * 80)

        print(f"\n🔧 初始化角色字典...")
        self.rl_dict = self._initialize_roles(rl_table, check_avail=self.parms['reschedule'])

        print(f"\n🔧 初始化活动字典...")
        self.ac_dict = self._initialize_activities(self.ac_index, init_states)

        print(f"\n🔧 初始化事件队列...")
        self.queue = self._initialize_queue(long_iarr)

        print(f"\n🔧 初始化执行状态...")
        self.execution_state = self._initialize_exec_state(self.sequences)

        # 🔍 数据一致性检查
        print(f"\n" + "=" * 80)
        print("🔍 数据一致性检查")
        print("=" * 80)

        queue_caseids = set([event['caseid'] for event in self.queue.get_all().queue])
        exec_caseids = set(self.execution_state.keys())

        print(f"   - 队列中的案例数: {len(queue_caseids)}")
        print(f"   - 执行状态中的案例数: {len(exec_caseids)}")
        print(f"   - 队列前5个案例ID: {list(queue_caseids)[:5]}")
        print(f"   - 执行状态前5个案例ID: {list(exec_caseids)[:5]}")

        missing_in_exec = queue_caseids - exec_caseids
        missing_in_queue = exec_caseids - queue_caseids

        if missing_in_exec:
            print(f"\n   ❌ 错误: {len(missing_in_exec)} 个案例在队列中但不在执行状态中")
            print(f"   缺失案例示例: {list(missing_in_exec)[:10]}")

            # 🔍 深度调试：找出为什么这些案例缺失
            print(f"\n   🔍 深度调试缺失案例:")
            for missing_cid in list(missing_in_exec)[:3]:
                print(f"\n      案例 {missing_cid}:")
                # 检查是否在原始 sequences 中
                if isinstance(sequences, pd.DataFrame):
                    in_sequences = missing_cid in sequences['caseid'].values
                    print(f"         - 是否在过滤后的 sequences 中: {in_sequences}")
                    if in_sequences:
                        case_data = sequences[sequences['caseid'] == missing_cid]
                        print(f"         - 该案例数据:")
                        print(case_data)

            raise ValueError("❌ 数据不一致：队列和执行状态案例ID不匹配")

        if missing_in_queue:
            print(f"\n   ⚠️  警告: {len(missing_in_queue)} 个案例在执行状态中但不在队列中")
            print(f"   多余案例示例: {list(missing_in_queue)[:10]}")

        print(f"\n   ✅ 数据一致性检查通过")

        print(f"\n" + "=" * 80)
        print("🎬 开始仿真生成事件日志")
        print("=" * 80)
        return self._generate(pr_act_initial, n_size, num_elements)

    @staticmethod
    def _filter_short_iarr(extra_short, iarr):
        """筛选短案例和长案例的到达时间字典"""
        print(f"\n🔍 _filter_short_iarr 调试:")
        print(f"   - extra_short 类型: {type(extra_short)}")
        print(f"   - extra_short 长度: {len(extra_short) if hasattr(extra_short, '__len__') else 'N/A'}")
        if isinstance(extra_short, pd.DataFrame):
            print(f"   - extra_short 唯一案例数: {extra_short['caseid'].nunique()}")
            print(f"   - extra_short 前5个案例: {extra_short['caseid'].unique()[:5].tolist()}")

        print(f"   - iarr 类型: {type(iarr)}")
        print(f"   - iarr 长度: {len(iarr) if hasattr(iarr, '__len__') else 'N/A'}")
        if isinstance(iarr, pd.DataFrame):
            print(f"   - iarr 唯一案例数: {iarr['caseid'].nunique()}")
            print(f"   - iarr 前5个案例: {iarr['caseid'].unique()[:5].tolist()}")

        long_iarr = {x['caseid']: x['timestamp'] for x in iarr.to_dict('records') if
                     x['caseid'] not in extra_short['caseid'].unique()}
        short_iarr = {x['caseid']: x['timestamp'] for x in iarr.to_dict('records') if
                      x['caseid'] in extra_short['caseid'].unique()}

        print(f"\n   📊 筛选结果:")
        print(f"   - 短案例数量: {len(short_iarr)}")
        print(f"   - 长案例数量: {len(long_iarr)}")
        print(f"   - 短案例前5个ID: {list(short_iarr.keys())[:5]}")
        print(f"   - 长案例前5个ID: {list(long_iarr.keys())[:5]}")

        return short_iarr, long_iarr

    @staticmethod
    def _filter_extra_short_cases(sequences):
        """过滤掉只有 Start/End 的超短案例"""
        print(f"\n🔍 _filter_extra_short_cases 调试:")
        print(f"   - 输入 sequences 类型: {type(sequences)}")
        print(f"   - 输入 sequences 形状: {sequences.shape if hasattr(sequences, 'shape') else 'N/A'}")
        if isinstance(sequences, pd.DataFrame):
            print(f"   - 输入唯一案例数: {sequences['caseid'].nunique()}")
            print(f"   - 输入唯一活动: {sequences['task'].unique().tolist() if 'task' in sequences.columns else 'N/A'}")

        filtered_sequences = sequences[~sequences.task.isin(['Start', 'End'])]['caseid'].unique()
        extra_short_cases = set(sequences['caseid'].unique()) - set(filtered_sequences)
        extra_short = sequences[sequences['caseid'].isin(extra_short_cases)]
        sequences = sequences[sequences['caseid'].isin(filtered_sequences)]
        sequences = sequences[~sequences.task.isin(['Start', 'End'])]

        print(f"\n   📊 过滤结果:")
        print(f"   - 超短案例数: {len(extra_short_cases)}")
        print(f"   - 超短案例ID示例: {list(extra_short_cases)[:5]}")
        print(f"   - 过滤后案例数: {sequences['caseid'].nunique() if isinstance(sequences, pd.DataFrame) else 'N/A'}")
        print(f"   - 过滤后总行数: {len(sequences) if hasattr(sequences, '__len__') else 'N/A'}")

        return extra_short, sequences

    def _generate(self, pr_act_initial, n_size, num_elements):
        """执行事件驱动仿真，生成完整事件日志"""
        print(f"\n🎬 进入 _generate 方法")
        print(f"   - pr_act_initial: {pr_act_initial}")
        print(f"   - n_size: {n_size}")
        print(f"   - num_elements: {num_elements}")

        event_log = list()

        # 辅助函数：创建事件记录
        def create_record(cid, ac_rl, res, ts):
            """创建标准化事件记录"""
            return {
                'caseid': cid,
                'task': self.index_ac[ac_rl[0]],
                'resource': res,
                'role': self.rl_dict[ac_rl[1]].get_name(),
                'end_timestamp': ts
            }

        # 初始化数据结构
        open_events = dict()
        active_instances = dict()
        pr_wip = pr_act_initial

        # 🔍 调试计数器
        processed_count = 0
        skipped_count = 0
        event_type_counter = {
            'create_instance': 0,
            'create_activity': 0,
            'complete_activity': 0,
            'complete_instance': 0
        }

        # 进度条初始化
        pbar = tqdm(total=num_elements, desc='事件生成进度')

        print(f"\n🎮 开始事件循环:")
        print(f"   - 初始队列大小: {self.queue.get_all().qsize()}")
        print(f"   - 初始在制品数: {pr_wip}")

        loop_count = 0
        max_loops = num_elements * 10  # 防止无限循环

        while not self.queue.get_all().empty():
            loop_count += 1

            # 🔍 防止无限循环
            if loop_count > max_loops:
                print(f"\n❌ 检测到可能的无限循环！")
                print(f"   - 已循环次数: {loop_count}")
                print(f"   - 队列剩余: {self.queue.get_all().qsize()}")
                print(f"   - 已处理活动: {processed_count}")
                print(f"   - 跳过案例数: {skipped_count}")
                break

            # 🔍 每100次循环打印一次状态
            if loop_count % 100 == 0:
                print(f"\n📊 循环状态 (第 {loop_count} 次):")
                print(f"   - 队列剩余: {self.queue.get_all().qsize()}")
                print(f"   - 已处理: {processed_count}")
                print(f"   - 跳过: {skipped_count}")
                print(f"   - 事件类型统计: {event_type_counter}")

            element = self.queue.get_remove_first()
            cid = element['caseid']

            # 🔍 详细日志（前10个事件）
            if loop_count <= 10:
                print(f"\n🔄 处理事件 #{loop_count}:")
                print(f"   - 案例ID: {cid}")
                print(f"   - 动作: {element['action']}")
                print(f"   - 时间戳: {element.get('timestamp', 'N/A')}")

            # 检查案例状态是否存在
            if cid not in self.execution_state:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"\n⚠️  跳过未初始化案例: {cid}")
                    print(f"   - 动作: {element['action']}")
                    print(f"   - 执行状态包含的案例数: {len(self.execution_state)}")
                    print(f"   - 执行状态前5个案例: {list(self.execution_state.keys())[:5]}")
                elif skipped_count == 6:
                    print(f"\n... (后续跳过信息将被省略)")
                continue

            # 统计事件类型
            event_type_counter[element['action']] += 1

            # 1. 创建案例实例事件
            if element['action'] == 'create_instance':
                processed_count += 1

                if processed_count <= 3:
                    print(f"\n✅ 创建案例实例: {cid}")

                transition = self.execution_state[cid]['transitions'].pop(0)
                self.execution_state[cid]['state'] = InstanceState.INEXECUTION

                self.queue.add({
                    'timestamp': element['timestamp'],
                    'action': 'create_activity',
                    'caseid': cid,
                    'transition': transition
                })

                pr_wip += 1

                if processed_count <= 3:
                    print(f"   - 全局在制品数: {pr_wip} (+1)")
                    print(f"   - 第一个活动: {transition}")

                active_instances[cid] = en.ProcessInstance(
                    cid, n_size, (self.n_feat_proc, self.n_feat_wait),
                    dual=True, n_act=True
                )

            # 2. 创建活动事件
            elif element['action'] == 'create_activity':
                transition = element['transition']

                ac_wip = self.ac_dict[transition[0]].get_active_instances()

                if self.parms['all_r_pool']:
                    rp_oc = [self.rl_dict[x].get_occupancy() for x in range(0, len(self.rl_dict))]
                else:
                    rp_oc = [self.rl_dict[transition[1]].get_occupancy()]

                wip = self.inter_scaler.transform(
                    np.array([pr_wip, ac_wip]).reshape(-1, 2))[0]

                active_instances[cid].update_proc_ngram(
                    transition[0], element['timestamp'], wip, rp_oc)

                act_ngram, feat_ngram = active_instances[cid].get_proc_ngram()

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    with self.g1.as_default():
                        with self.first_session.as_default():
                            preds = self.proc_model_path.predict(
                                {'ac_input': np.array([act_ngram]),
                                 'features': feat_ngram})

                preds[preds < 0] = 0.000001
                proc_t = preds[0]
                active_instances[cid].update_proc(proc_t)

                ipred = self.scaler.inverse_transform(
                    np.concatenate((preds, np.array([[0.000]])), axis=1))
                iproc_t = ipred[0][0]

                original_time = element['timestamp']
                release_time = original_time + timedelta(seconds=int(round(iproc_t)))

                res = self.rl_dict[transition[1]].assign_resource(release_time)

                if res is not None:

                    ev_id = 'event_' + str(uuid.uuid4())

                    open_events[ev_id] = {
                        'pr_instances': pr_wip,
                        'tsk_start_inst': ac_wip,
                        'res_id': res,
                        'start_timestamp': element['timestamp']
                    }

                    if not self.parms['all_r_pool']:
                        open_events[ev_id]['rp_start_oc'] = rp_oc[0]

                    self.ac_dict[transition[0]].add_act()

                    element['timestamp'] = release_time
                    element['action'] = 'complete_activity'
                    element['ev_id'] = ev_id

                else:
                    release_time = self.rl_dict[transition[1]].get_next_release()
                    element['timestamp'] = release_time + timedelta(microseconds=1)

                self.queue.add(element)

            # 3. 完成活动事件
            elif element['action'] == 'complete_activity':
                transition = element['transition']

                event = open_events[element['ev_id']]

                self.rl_dict[transition[1]].release_resource(event['res_id'])

                self.ac_dict[transition[0]].remove_act()

                complete_event = {
                    **create_record(cid, transition, event['res_id'], element['timestamp']),
                    **event
                }

                event_log.append(complete_event)

                complete_event.pop('res_id', None)
                open_events.pop(element['ev_id'], None)

                try:
                    next_act = self.execution_state[cid]['transitions'].pop(0)
                    element['transition'] = next_act

                    if self.parms['all_r_pool']:
                        rp_oc = [self.rl_dict[x].get_occupancy() for x in range(0, len(self.rl_dict))]
                    else:
                        rp_oc = [self.rl_dict[transition[1]].get_occupancy()]

                    wip = self.end_inter_scaler.transform(
                        np.array([pr_wip]).reshape(-1, 1))[0]

                    active_instances[cid].update_wait_ngram(
                        next_act[0], element['timestamp'], wip, rp_oc)

                    n_act_ngram, feat_ngram = active_instances[cid].get_wait_ngram()

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        with self.g2.as_default():
                            with self.second_session.as_default():
                                preds = self.wait_model_path.predict({
                                    'ac_input': np.array([n_act_ngram]),
                                    'features': feat_ngram
                                })

                    preds[preds < 0] = 0.000001
                    wait_t = preds[0]
                    active_instances[cid].update_wait(wait_t)

                    ipred = self.scaler.inverse_transform(
                        np.concatenate((np.array([[0.000]]), preds), axis=1))
                    iwait_t = ipred[0][1]

                    element['timestamp'] += timedelta(seconds=int(iwait_t))
                    element['action'] = 'create_activity'

                except IndexError:
                    # 没有后续活动，完成案例
                    element['action'] = 'complete_instance'

                # 更新进度条
                pbar.update(1)

                # 将事件重新加入队列
                self.queue.add(element)

            # 4. 完成案例事件
            elif element['action'] == 'complete_instance':

                # 更新案例状态为"已完成"
                self.execution_state[cid]['state'] = InstanceState.COMPLETE

                # 减少全局在制品计数
                pr_wip -= 1

                # 移除案例实例
                active_instances.pop(cid, None)

            # 5. 未知事件类型
            else:
                raise ValueError(f"未知事件类型: {element['action']}")

        # 关闭进度条
        pbar.close()

        # 🔍 最终统计
        print(f"\n" + "=" * 80)
        print(f"🏁 仿真完成统计:")
        print(f"=" * 80)
        print(f"   - 总循环次数: {loop_count}")
        print(f"   - 已处理案例: {processed_count}")
        print(f"   - 跳过案例: {skipped_count}")
        print(f"   - 生成事件数: {len(event_log)}")
        print(f"\n   📊 事件类型统计:")
        for event_type, count in event_type_counter.items():
            print(f"      - {event_type}: {count}")
        print(f"\n   - 最终在制品数: {pr_wip}")
        print(f"   - 活跃实例数: {len(active_instances)}")
        print(f"   - 未完成事件数: {len(open_events)}")
        print("=" * 80 + "\n")

        return event_log

    @staticmethod
    def _initialize_activities(ac_index, init_states):
        """初始化活动字典"""
        print(f"\n🔧 _initialize_activities 调试:")
        print(f"   - ac_index 长度: {len(ac_index)}")
        print(f"   - ac_index 内容: {ac_index}")
        print(f"   - init_states 类型: {type(init_states)}")
        print(f"   - init_states 内容: {init_states}")

        activities_dict = dict()
        for key, value in ac_index.items():
            if key not in ['Start', 'End']:
                initial_count = int(round(init_states.get(key, 0)))
                activities_dict[value] = en.ActivityCounter(
                    key, index=value, initial=initial_count)
                print(f"      - 添加活动: {key} (索引={value}, 初始={initial_count})")

        print(f"   ✅ 活动字典创建完成，共 {len(activities_dict)} 个活动")
        return activities_dict

    @staticmethod
    def _initialize_roles(roles, check_avail):
        """初始化角色字典"""
        print(f"\n🔧 _initialize_roles 调试:")
        print(f"   - roles 类型: {type(roles)}")
        print(f"   - roles 长度: {len(roles) if hasattr(roles, '__len__') else 'N/A'}")
        if isinstance(roles, pd.DataFrame):
            print(f"   - roles 列: {roles.columns.tolist()}")
            print(f"   - roles 内容:")
            print(roles)

        rl_dict = dict()
        for role in roles.to_dict('records'):
            rl_dict[role['role_index']] = en.Role(
                role['role_name'], role['size'], index=role['role_index'], check_avail=check_avail)
            print(f"      - 添加角色: {role['role_name']} (索引={role['role_index']}, 大小={role['size']})")

        print(f"   ✅ 角色字典创建完成，共 {len(rl_dict)} 个角色")
        return rl_dict

    @staticmethod
    def _initialize_queue(iarr):
        """初始化事件队列"""
        print(f"\n🔧 _initialize_queue 调试:")
        print(f"   - iarr 类型: {type(iarr)}")
        print(f"   - iarr 长度: {len(iarr)}")
        print(f"   - iarr 前5个案例:")
        for i, (k, v) in enumerate(list(iarr.items())[:5]):
            print(f"      {i + 1}. 案例ID={k}, 时间戳={v}")

        queue = en.Queue()
        for k, v in iarr.items():
            queue.add({'timestamp': v,
                       'action': 'create_instance',
                       'caseid': k})

        print(f"   ✅ 队列创建完成，共 {queue.get_all().qsize()} 个事件")
        return queue

    @staticmethod
    def _initialize_exec_state(sequences):
        """
        初始化执行状态

        参数:
            sequences: dict, {caseid: [(ac_idx, role_idx), ...]}
        """
        print(f"\n🔧 _initialize_exec_state 调试:")
        print(f"   - sequences 类型: {type(sequences)}")
        print(f"   - sequences 长度: {len(sequences)}")
        print(f"   - sequences 前5个案例ID: {list(sequences.keys())[:5]}")

        # 🔍 详细检查前3个案例
        for i, (caseid, transitions) in enumerate(list(sequences.items())[:3]):
            print(f"\n      案例 #{i + 1}:")
            print(f"         - 案例ID: {caseid}")
            print(f"         - transitions 类型: {type(transitions)}")
            print(f"         - transitions 长度: {len(transitions) if hasattr(transitions, '__len__') else 'N/A'}")
            print(f"         - transitions 内容: {transitions}")

        execution_state = dict()

        skipped_cases = []
        for caseid, transitions in sequences.items():
            # 验证 transitions 不为空
            if not transitions:
                print(f"      ⚠️  警告: 案例 {caseid} 的 transitions 为空，跳过")
                skipped_cases.append(caseid)
                continue

            # 🔍 验证 transitions 格式
            if not isinstance(transitions, list):
                print(f"      ⚠️  警告: 案例 {caseid} 的 transitions 不是列表: {type(transitions)}")
                skipped_cases.append(caseid)
                continue

            execution_state[caseid] = {
                'state': InstanceState.WAITING,
                'transitions': transitions.copy()  # 使用副本
            }

        print(f"\n   📊 执行状态初始化结果:")
        print(f"      - 成功初始化: {len(execution_state)} 个案例")
        print(f"      - 跳过: {len(skipped_cases)} 个案例")
        if skipped_cases:
            print(f"      - 跳过的案例ID: {skipped_cases[:10]}")

        print(f"   ✅ 执行状态字典创建完成")
        return execution_state

    @staticmethod
    def _encode_secuences(sequences, ac_idx, rl_task, rl_table):
        """编码序列"""
        print(f"\n🔧 _encode_secuences 调试:")
        print(f"   - sequences 类型: {type(sequences)}")
        print(f"   - sequences 形状: {sequences.shape if hasattr(sequences, 'shape') else 'N/A'}")
        if isinstance(sequences, pd.DataFrame):
            print(f"   - sequences 列: {sequences.columns.tolist()}")
            print(f"   - sequences 唯一案例数: {sequences['caseid'].nunique()}")
            print(f"   - sequences 前5行:")
            print(sequences.head())

        print(f"\n   - ac_idx 长度: {len(ac_idx)}")
        print(f"   - ac_idx 内容: {ac_idx}")

        print(f"\n   - rl_task 类型: {type(rl_task)}")
        if isinstance(rl_task, pd.DataFrame):
            print(f"   - rl_task 形状: {rl_task.shape}")
            print(f"   - rl_task 列: {rl_task.columns.tolist()}")
            print(f"   - rl_task 内容:")
            print(rl_task)

        print(f"\n   - rl_table 类型: {type(rl_table)}")
        if isinstance(rl_table, pd.DataFrame):
            print(f"   - rl_table 形状: {rl_table.shape}")
            print(f"   - rl_table 列: {rl_table.columns.tolist()}")
            print(f"   - rl_table 内容:")
            print(rl_table)

        seq = sequences.copy()

        # Determine biggest resource pool as default
        def_role = rl_table[rl_table['size'] == rl_table['size'].max()].iloc[0]['role_name']
        print(f"\n   - 默认角色: {def_role}")

        # Assign roles to activities
        print(f"\n   🔄 分配活动索引...")
        seq['ac_index'] = seq.apply(
            lambda x: ac_idx[x.task], axis=1)
        print(f"      ✅ 活动索引分配完成")
        print(f"      - 示例数据:")
        print(seq[['caseid', 'task', 'ac_index']].head())

        print(f"\n   🔄 合并角色任务...")
        seq = seq.merge(rl_task, how='left', on='task')
        print(f"      ✅ 角色任务合并完成")
        print(f"      - 示例数据:")
        print(seq[['caseid', 'task', 'role']].head())

        seq.fillna(value={'role': def_role}, inplace=True)

        print(f"\n   🔄 合并角色表...")
        seq = seq.merge(rl_table, how='left', left_on='role', right_on='role_name')
        print(f"      ✅ 角色表合并完成")
        print(f"      - 示例数据:")
        print(seq[['caseid', 'task', 'role', 'role_index']].head())

        ac_rl = lambda x: (x.ac_index, x.role_index)
        seq['ac_rl'] = seq.apply(ac_rl, axis=1)

        print(f"\n   🔄 按案例分组...")
        num_elements = 0
        encoded_seq = dict()

        # 🔍 调试分组过程
        grouped = seq.sort_values('pos_trace').groupby('caseid')
        print(f"      - 分组数量: {len(grouped)}")

        for i, (key, group) in enumerate(grouped):
            if i < 5:  # 打印前5个案例
                print(f"\n      案例 #{i + 1}:")
                print(f"         - 案例ID: {key}")
                print(f"         - 组大小: {len(group)}")
                print(f"         - ac_rl 列表: {group.ac_rl.to_list()}")

            encoded_seq[key] = group.ac_rl.to_list()
            num_elements += len(encoded_seq[key])

        print(f"\n   📊 编码结果:")
        print(f"      - 编码案例数: {len(encoded_seq)}")
        print(f"      - 总活动数: {num_elements}")
        print(f"      - 前5个案例ID: {list(encoded_seq.keys())[:5]}")

        # 🔍 验证编码结果
        if not encoded_seq:
            print(f"\n   ❌ 错误: encoded_seq 为空！")
            print(f"      - 原始序列数: {len(sequences)}")
            print(f"      - 分组后序列数: {len(encoded_seq)}")
            raise ValueError("编码序列失败：结果为空")

        print(f"   ✅ 序列编码完成")

        return encoded_seq, num_elements


