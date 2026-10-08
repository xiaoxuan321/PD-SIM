# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:45:36 2020

@author: Manuel Camargo
"""
import copy
import itertools
import json
import os
import shutil
import subprocess
from abc import ABCMeta, abstractmethod
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from xml.dom import minidom

import pandas as pd
import utils.support as sup
from utils.support import safe_exec

from core_modules.sequences_generator import structure_optimizer as so
from support_modules.common import FileExtensions as Fe
from support_modules.common import LogAttributes as La
from support_modules.common import SequencesGenerativeMethods as SqM


class SeqGeneratorFabric:

    @classmethod
    def get_generator(cls, method):
        if method == SqM.PROCESS_MODEL:
            return StochasticProcessModelGenerator
        elif method == SqM.TEST:
            return OriginalSequencesGenerator
        else:
            raise ValueError('Nonexistent sequences generator')


class SeqGenerator(metaclass=ABCMeta):
    """
    Generator base class
    """

    def __init__(self, parameters, log_train):
        """constructor"""
        self.parameters = parameters
        self.log_train = log_train
        self.is_safe = True
        self.model_metadata = dict()
        self.gen_seqs = None

    @abstractmethod
    def generate(self, num_inst, start_time):
        pass

    @abstractmethod
    def clean_time_stamps(self):
        pass

    @staticmethod
    def sort_log(log):
        log = sorted(log.to_dict('records'), key=lambda x: x[La.CASE_ID])
        for key, group in itertools.groupby(log, key=lambda x: x[La.CASE_ID]):
            events = list(group)
            events = sorted(events, key=itemgetter(La.START_TIME))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_records(log)
        log.sort_values(by=La.START_TIME, ascending=True, inplace=True)
        return log


class StochasticProcessModelGenerator(SeqGenerator):
    def discovery_model(self):
        self._verify_model()

    def generate(self, log, start_time):
        print("\n" + "=" * 80)
        print("🔍 [DEBUG] StochasticProcessModelGenerator.generate() 开始")
        print("=" * 80)

        num_inst = len(log.caseid.unique())
        print(f"📊 [DEBUG] 实例数量: {num_inst}")
        print(f"⏰ [DEBUG] 开始时间: {start_time}")

        # verify if model exists
        print("\n🔎 [DEBUG] 步骤1: 验证模型...")
        self._verify_model()
        print(f"✅ [DEBUG] 模型路径: {self.model}")
        print(f"📁 [DEBUG] 模型文件存在: {os.path.exists(self.model)}")

        # update model parameters
        print("\n🔧 [DEBUG] 步骤2: 修改仿真模型参数...")
        self._modify_simulation_model(self.model, num_inst, start_time)
        print("✅ [DEBUG] 模型参数修改完成")

        # Create temp path
        print("\n📁 [DEBUG] 步骤3: 创建临时路径...")
        temp_path = self._temp_path_creation()
        print(f"✅ [DEBUG] 临时路径: {temp_path}")
        print(f"📁 [DEBUG] 临时路径存在: {os.path.exists(temp_path)}")

        # generate instances
        print("\n🚀 [DEBUG] 步骤4: 执行仿真器...")
        sim_log = self._execute_simulator(self.parameters['bimp_path'], temp_path, self.model)

        # ✅ 关键调试信息
        print("\n" + "=" * 80)
        print("🔍 [DEBUG] 仿真器执行完成，检查输出...")
        print("=" * 80)
        print(f"📄 [DEBUG] sim_log 类型: {type(sim_log)}")
        print(f"📄 [DEBUG] sim_log 值: {sim_log}")

        # 检查文件是否存在
        if isinstance(sim_log, (str, Path)):
            sim_log_str = str(sim_log)
            print(f"\n🔍 [DEBUG] 检查仿真日志文件:")
            print(f"  - 文件路径: {sim_log_str}")
            print(f"  - 文件存在: {os.path.exists(sim_log_str)}")

            if os.path.exists(sim_log_str):
                file_size = os.path.getsize(sim_log_str)
                print(f"  - 文件大小: {file_size} 字节")
                if file_size == 0:
                    print("  ⚠️  警告: 文件大小为0!")
            else:
                print("  ❌ 错误: 文件不存在!")

                # 列出临时目录内容
                print(f"\n📂 [DEBUG] 临时目录内容 ({temp_path}):")
                if os.path.exists(temp_path):
                    files = os.listdir(temp_path)
                    if files:
                        for f in files:
                            full_path = os.path.join(temp_path, f)
                            size = os.path.getsize(full_path) if os.path.isfile(full_path) else 'DIR'
                            print(f"  - {f} ({size})")
                    else:
                        print("  (空目录)")
                else:
                    print("  ❌ 临时目录也不存在!")

        # order by Case ID and add position in the trace
        print("\n📋 [DEBUG] 步骤5: 重命名仿真日志...")
        sim_log = self._rename_sim_log(sim_log)
        print("✅ [DEBUG] 日志重命名完成")
        print(f"📊 [DEBUG] 仿真日志形状: {sim_log.shape}")

        # save traces
        self.gen_seqs = sim_log

        # remove simulated log
        print("\n🧹 [DEBUG] 步骤6: 清理临时文件...")
        shutil.rmtree(temp_path)
        print("✅ [DEBUG] 临时文件清理完成")

        print("\n" + "=" * 80)
        print("✅ [DEBUG] StochasticProcessModelGenerator.generate() 完成")
        print("=" * 80 + "\n")

    def _rename_sim_log(self, sim_log):
        print(f"\n🔍 [DEBUG] _rename_sim_log() 开始")
        print(f"  - 输入 sim_log: {sim_log}")
        print(f"  - 输入类型: {type(sim_log)}")

        # ✅ 添加文件存在性检查
        if isinstance(sim_log, (str, Path)):
            sim_log_str = str(sim_log)
            if not os.path.exists(sim_log_str):
                print(f"\n❌ [ERROR] 文件不存在: {sim_log_str}")

                # 尝试找到可能的文件
                dir_path = os.path.dirname(sim_log_str)
                base_name = os.path.basename(sim_log_str)
                print(f"\n🔍 [DEBUG] 尝试在目录中查找类似文件:")
                print(f"  - 目录: {dir_path}")
                print(f"  - 预期文件名: {base_name}")

                if os.path.exists(dir_path):
                    print(f"\n📂 [DEBUG] 目录内容:")
                    for f in os.listdir(dir_path):
                        full_path = os.path.join(dir_path, f)
                        size = os.path.getsize(full_path) if os.path.isfile(full_path) else 'DIR'
                        print(f"  - {f} ({size})")

                raise FileNotFoundError(
                    f"仿真日志文件不存在: {sim_log_str}\n"
                    f"请检查仿真生成步骤是否成功执行"
                )

        print(f"📖 [DEBUG] 正在读取CSV文件...")
        sim_log = pd.read_csv(sim_log)
        print(f"✅ [DEBUG] CSV读取成功，形状: {sim_log.shape}")

        print(f"🔄 [DEBUG] 排序日志...")
        sim_log = self.sort_log(sim_log)

        print(f"🏷️  [DEBUG] 重命名 Case ID...")
        sim_log[La.CASE_ID] = sim_log[La.CASE_ID] + 1
        sim_log[La.CASE_ID] = (
                'Case' + sim_log[La.CASE_ID].astype('int64').astype('string')
        )

        print(f"✅ [DEBUG] _rename_sim_log() 完成")
        return sim_log

    def clean_time_stamps(self):
        self.gen_seqs.drop(columns=[La.START_TIME, La.END_TIME], inplace=True)

    def _verify_model(self) -> None:
        model_path = os.path.join(self.parameters['bpmn_models'], self.parameters['file'].split('.')[0] + Fe.BPMN)
        if os.path.exists(model_path) and self.parameters['update_gen']:
            self.is_safe = self._discover_model(True, is_safe=self.is_safe)
        elif not os.path.exists(model_path):
            self.is_safe = self._discover_model(False, is_safe=self.is_safe)
        self.model = model_path

    @safe_exec
    def _discover_model(self, compare, **_kwargs):
        # 初始化 ASHA 优化器
        structure_optimizer = so.StructureOptimizer(self.parameters, copy.deepcopy(self.log_train))

        # 执行 ASHA 试验
        structure_optimizer.execute_trials()

        # 从优化器属性中获取 Optuna 汇总后的结果
        struc_model = structure_optimizer.best_output  # 对应 trial.user_attrs['output']
        best_parameters = structure_optimizer.best_parms
        best_similarity = structure_optimizer.best_similarity

        metadata_file = os.path.join(self.parameters['bpmn_models'],
                                     f"{self.parameters['file'].split('.')[0]}_meta{Fe.JSON}")

        save = True
        if compare:
            save = self._loading_parameters_from_existing_model(best_similarity, metadata_file, save)

        if save and struc_model:  # 确保路径有效
            self._extract_model_metadata(best_parameters, best_similarity)
            self._copy_best_model(struc_model)
            sup.create_json(self.model_metadata, metadata_file)
            print(f"✅ 成功保存最优模型与元数据。相似度: {best_similarity:.4f}")

        # 清理 ASHA 产生的临时文件夹
        shutil.rmtree(structure_optimizer.temp_output)

    def _copy_best_model(self, struc_model):
        file_name = f"{self.parameters['file'].split('.')[0]}{Fe.BPMN}"
        destination = os.path.join(self.parameters['bpmn_models'], file_name)
        source = os.path.join(struc_model, file_name)
        shutil.copyfile(source, destination)

    def _extract_model_metadata(self, best_parameters, best_similarity):
        # Optuna 直接返回选中的值，不再需要通过索引查找
        self.model_metadata['alg_manag'] = best_parameters['alg_manag']
        self.model_metadata['gate_management'] = best_parameters['gate_management']

        # 保存 ASHA 特有的混合策略参数
        self.model_metadata['confidence_threshold'] = best_parameters.get('confidence_threshold')
        self.model_metadata['laplace_alpha'] = best_parameters.get('laplace_alpha')

        if self.parameters['mining_alg'] in ['sm1', 'sm3']:
            self.model_metadata['epsilon'] = best_parameters['epsilon']
            self.model_metadata['eta'] = best_parameters['eta']
        elif self.parameters['mining_alg'] == 'im':
            self.model_metadata['im_noise_threshold'] = best_parameters['im_noise_threshold']

        self.model_metadata['similarity'] = best_similarity
        self.model_metadata['generated_at'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    @staticmethod
    def _loading_parameters_from_existing_model(best_similarity, metadata_file, save):
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                data = {k: v for k, v in data.items()}
                print(data['similarity'])
            if data['similarity'] > best_similarity:
                save = False
                print('dont save')
        return save

    @staticmethod
    def _temp_path_creation() -> Path:
        # Paths redefinition
        temp_path = os.path.join('output_files', sup.folder_id())
        # Output folder creation
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
            print(f"📁 [DEBUG] 创建临时目录: {temp_path}")
        return Path(temp_path)

    @staticmethod
    def _modify_simulation_model(model, num_inst, start_time):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        print(f"  - 模型文件: {model}")
        print(f"  - 实例数: {num_inst}")
        print(f"  - 开始时间: {start_time}")

        my_doc = minidom.parse(model)
        items = my_doc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        with open(model, 'wb') as f:
            f.write(my_doc.toxml().encode('utf-8'))
        f.close()

    @staticmethod
    def _execute_simulator(bimp_path, temp_path, model):
        print(f"  - BIMP路径: {bimp_path}")
        print(f"  - 临时路径: {temp_path}")
        print(f"  - 模型文件: {model}")

        # ✅ 构建输出文件路径（注意：不要加 .csv 后缀！）
        sim_log = os.path.join(temp_path, sup.file_id('SIM_'))
        print(f"  - 仿真日志路径（无后缀）: {sim_log}")

        # ✅ BIMP会自动添加.csv后缀
        expected_output = sim_log
        print(f"  - 预期输出文件: {expected_output}")

        args = ['java', '-jar', bimp_path, model, '-csv', sim_log]
        print(f"\n🚀 [DEBUG] 执行命令: {' '.join(args)}")

        try:
            result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ [DEBUG] BIMP执行成功")
            print(f"📄 [DEBUG] STDOUT:\n{result.stdout.decode('utf-8', errors='ignore')}")
            if result.stderr:
                print(f"⚠️  [DEBUG] STDERR:\n{result.stderr.decode('utf-8', errors='ignore')}")
        except subprocess.CalledProcessError as e:
            print(f"❌ [ERROR] BIMP执行失败")
            print(f"  - 返回码: {e.returncode}")
            print(f"  - STDOUT: {e.stdout.decode('utf-8', errors='ignore')}")
            print(f"  - STDERR: {e.stderr.decode('utf-8', errors='ignore')}")
            raise

        # ✅ 检查BIMP实际生成的文件
        print(f"\n🔍 [DEBUG] 检查BIMP输出:")
        print(f"  - 预期文件: {expected_output}")
        print(f"  - 文件存在: {os.path.exists(expected_output)}")

        if os.path.exists(expected_output):
            print(f"  - 文件大小: {os.path.getsize(expected_output)} 字节")
            # ✅ 返回带.csv后缀的路径
            return expected_output
        else:
            # 如果预期文件不存在，列出目录看看生成了什么
            print(f"\n❌ [ERROR] 预期文件不存在!")
            print(f"📂 [DEBUG] 临时目录内容:")
            for f in os.listdir(temp_path):
                full_path = os.path.join(temp_path, f)
                size = os.path.getsize(full_path) if os.path.isfile(full_path) else 'DIR'
                print(f"  - {f} ({size})")

            # ✅ 仍然返回预期路径，让后续错误处理捕获
            return expected_output


class OriginalSequencesGenerator(SeqGenerator):

    def generate(self, log, start_time):
        print("\n🔍 [DEBUG] OriginalSequencesGenerator.generate() 开始")

        sequences = log.copy(deep=True)
        sequences = sequences[[La.CASE_ID, La.ACTIVITY, La.RESOURCE, La.START_TIME]]
        replacements = {case_name: f'Case{idx + 1}' for idx, case_name in enumerate(sequences[La.CASE_ID].unique())}
        sequences.replace({La.CASE_ID: replacements}, inplace=True)
        sequences = self.sort_log(sequences)
        self.gen_seqs = (sequences.rename(columns={La.RESOURCE: 'resource'}).sort_values([La.CASE_ID, 'pos_trace']))

        print(f"✅ [DEBUG] OriginalSequencesGenerator.generate() 完成，形状: {self.gen_seqs.shape}")

    def clean_time_stamps(self):
        self.gen_seqs.drop(columns=La.START_TIME)
