import copy
import os
import numpy as np
# 导入你的优化器类所在的模块
from structure_optimizer import StructureOptimizer
import core_modules.sequences_generator.structure_params_miner as spm
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

# 1. 还原 0.79 分的那组“神仙参数”
target_params = {
    "alg_manag": "removal",
    "confidence_threshold": 2,
    "gate_management": "discovery",
    "im_noise_threshold": 0.2815073545925609,  # 精确到小数点后16位
    "laplace_alpha": 0.022248713059876433  # 精确到小数点后16位
}

# 2. 基础配置 (请确认这里与之前完全一致！)
settings = {
    'file': 'ConsultaDataMining201618.xes',
    'bimp_path': 'E:\DPSIM-1\DP-SIM123\external_tools\\bimp\qbp-simulator-engine.jar',  # ⚠️请确认这是你的真实jar路径
    'output': 'output_files/verify_test',
    'repetitions': 10,
    'max_eval': 1,
    'mining_alg': 'im',

    'read_options': {
        'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
        'one_timestamp': False,
        'filter_d_attrib': True,
        # ⭐⭐⭐ [新增] 补全缺失的列名映射 ⭐⭐⭐
        # 这是 LogReader 必须的参数。对于 XES 通常不需要复杂映射，
        # 但必须给一个空字典或默认映射以防止 KeyError。
        'column_names': {
            'Case ID': 'caseid',
            'Activity': 'task',
            'lifecycle:transition': 'event_type',
            'Resource': 'user',
            'Timestamp': 'end_timestamp'
        }
    }
}

# 模拟加载日志 (请使用你的读取器)
import readers.log_reader as lr

log = lr.LogReader(os.path.join('E:\DPSIM-1\DP-SIM123\input_files\event_logs', settings['file']), settings['read_options'])

# 初始化优化器
optimizer = StructureOptimizer(settings, log)


# 强制注入参数复现
def run_verification():
    print("🚀 开始终极复现验证...")
    import os

    # --- 强制路径修正 ---
    # 获取当前脚本的绝对路径：E:\DPSIM-1\DP-SIM123\core_modules\sequences_generator\verify_reproducibility.py
    current_script = os.path.abspath(__file__)
    # 向上退三级，到达根目录：E:\DPSIM-1\DP-SIM123
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script)))

    # 重新拼接挖掘工具和仿真工具的绝对路径
    im_jar = os.path.join(base_path, 'external_tools', 'inductiveminer', 'im-process-tree-bpmn.jar')
    bimp_jar = os.path.join(base_path, 'external_tools', 'bimp', 'qbp-simulator-engine.jar')

    # 注入到设置中
    settings['inductive_miner_path'] = im_jar
    settings['bimp_path'] = bimp_jar
    # -------------------

    print(f"✅ 修正后的挖掘工具路径: {im_jar}")
    if not os.path.exists(im_jar):
        print(f"❌ 警告: 路径依然无效，请手动确认文件位置!")
    # 确保基础输出目录存在
    if not os.path.exists('output_files'):
        os.makedirs('output_files')

    # 1. 准备数据
    optimizer.log_train = copy.deepcopy(optimizer.org_log_train)
    optimizer.log_valdn = copy.deepcopy(optimizer.org_log_valdn)

    trial_stg = copy.deepcopy(settings)
    trial_stg.update(target_params)

    # 2. 路径重定义（关键：获取代码生成的实际随机路径）
    status = STATUS_OK
    exec_times = {}
    rsp = optimizer._temp_path_redef(trial_stg, status=status, log_time=exec_times)

    if rsp['status'] != STATUS_OK:
        print(f"❌ 路径定义失败，请检查 output_files 写入权限")
        return

    trial_stg = rsp['values']
    actual_output = trial_stg['output']
    print(f"📂 实际执行输出目录: {actual_output}")

    # 3. 结构挖掘
    print("正在挖掘流程结构...")
    rsp = optimizer._mine_structure(trial_stg, status=status, log_time=exec_times)
    if rsp['status'] != STATUS_OK:
        print("❌ 挖掘失败，请检查 IM 算法配置")
        return
    structure = rsp['values']

    # 4. 参数提取与 BIMP 准备
    res_params = spm.StructureParametersMiner.mine_resources(settings, optimizer.log_train)
    rsp = optimizer._extract_parameters(trial_stg, structure, res_params, status=status, log_time=exec_times)

    # 5. 仿真执行（检查 BIMP 输出文件）
    print("正在仿真...")
    sim_rsp = optimizer._simulate(trial_stg, optimizer.log_valdn, status=status, log_time=exec_times)

    # 检查 sim_data 是否真的生成了
    sim_data_path = os.path.join(actual_output, 'sim_data')
    if not os.path.exists(sim_data_path):
        print(f"❌ 严重错误：找不到目录 {sim_data_path}")
        print("请检查 settings['bimp_path'] 是否指向正确的 .jar 文件")
        return

    if sim_rsp['status'] == STATUS_OK and len(sim_rsp['values']) > 0:
        sim_values = sim_rsp['values']
        scores = [x['sim_val'] for x in sim_values]
        avg_score = np.mean(scores)
        print(f"\n✅ 复现结果: {avg_score:.6f}")
    else:
        print(f"❌ 仿真未产生数据。请检查：\n1. Java 是否在环境变量中\n2. BIMP 路径: {trial_stg['bimp_path']}")


if __name__ == "__main__":
    run_verification()