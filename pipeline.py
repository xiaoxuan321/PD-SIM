# -*- coding: utf-8 -*-

import os
import sys

# ▶ 在任何数值/科学库导入之前限制并行线程
for _k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json

import click
import yaml

from support_modules.common import EmbeddingMethods as Em, OUTPUT_FILES
from support_modules.common import InterArrivalGenerativeMethods as IaG
from support_modules.common import SequencesGenerativeMethods as SqG
from support_modules.common import W2VecConcatMethod as Cm


@click.command()
@click.option('--file', default=None, required=True, type=str)
@click.option('--update_gen/--no-update_gen',
              default=False, required=False, type=bool)
@click.option('--update_ia_gen/--no-update_ia_gen',
              default=False, required=False, type=bool)
@click.option('--update_mpdf_gen/--no-update_mpdf_gen',
              default=False, required=False, type=bool)
@click.option('--update_times_gen/--no-update_times_gen',
              default=False, required=False, type=bool)
@click.option('--save_models/--no-save_models',
              default=True, required=False, type=bool)
@click.option('--evaluate/--no-evaluate',
              default=True, required=False, type=bool)
@click.option('--mining_alg', default='im', required=False)
@click.option('--exp_reps', default=5, required=False, type=int)
@click.option('--s_gen_repetitions', default=5, required=False, type=int)
@click.option('--s_gen_max_eval', default=50, required=False, type=int)
@click.option(
    '--seq_gen_method',
    default=SqG.PROCESS_MODEL,
    required=False,
    type=click.Choice(SqG().get_methods()))
@click.option(
    '--ia_gen_method',
    default=IaG.PROPHET,
    required=False,
    type=click.Choice(IaG().get_methods()))
@click.option(
    '--emb_method',
    default=Em.DOT_PROD,
    required=False,
    type=click.Choice(Em().get_types()))
@click.option(
    '--concat_method',
    default=Cm.SINGLE_SENTENCE,
    required=False,
    type=click.Choice(Cm().get_methods()))
@click.option('--include_times', default=False, required=False, type=bool)
@click.option('--manual_ia_params', default=None, required=False, type=str,
              help='JSON string or file path; skips Optuna search')
def main(file, update_gen, update_ia_gen, update_mpdf_gen,
         update_times_gen, save_models, evaluate, mining_alg,
         exp_reps, s_gen_repetitions, s_gen_max_eval,
         seq_gen_method, ia_gen_method,
         emb_method, concat_method, include_times,
         manual_ia_params):
    """顶层入口。"""
    import deep_simulator as ds

    params = dict()

    # ========================================================
    # 全局参数hA
    # ========================================================
    params['gl'] = dict()
    params['gl']['file'] = file
    params['gl']['update_gen'] = update_gen
    params['gl']['update_ia_gen'] = update_ia_gen
    params['gl']['update_mpdf_gen'] = update_mpdf_gen
    params['gl']['update_times_gen'] = update_times_gen
    params['gl']['save_models'] = save_models
    params['gl']['evaluate'] = evaluate
    params['gl']['mining_alg'] = mining_alg

    params = read_properties(params)

    params['gl']['sim_metric'] = 'tsd'  # 主优化指标
    params['gl']['add_metrics'] = [
        'day_hour_emd',
        'log_mae',
        'dl',
        'mae'
    ]
    params['gl']['exp_reps'] = exp_reps

    # ========================================================
    # 序列生成器/控制流优化参数
    # ========================================================
    params['s_gen'] = dict()
    params['s_gen']['gen_method'] = seq_gen_method
    params['s_gen']['repetitions'] = s_gen_repetitions
    params['s_gen']['max_eval'] = s_gen_max_eval
    params['s_gen']['concurrency'] = [0.0, 1.0]
    params['s_gen']['epsilon'] = [0.0, 1.0]
    params['s_gen']['eta'] = [0.0, 1.0]

    # IM 噪声阈值是 StructureOptimizer 的搜索区间，因此保留在这里。
    params['s_gen']['im_noise_threshold'] = [0.05, 0.15]
    # params['s_gen']['alg_manag'] = ['removal']
    params['s_gen']['alg_manag'] = ['replacement', 'removal']
    params['s_gen']['noise_threshold'] = [0.14, 0.19]
    params['s_gen']['gate_management'] = ['discovery']

    # ===================== [新增配置开始] =====================
    # LogReplayer 采用顺序重放。顺序模式便于保证序列流计数可复现，
    # 并避免并行写入共享 process_graph 导致计数冲突。
    params['s_gen']['replay_mode'] = 'seq'

    # 输出日志重放过程和计数摘要，便于核对模型与日志。
    params['s_gen']['replay_verbose'] = True

    # 并行模式的备用工作进程数。replay_mode='seq' 时不会用到，
    # 但保留该值可避免以后切换模式时 Windows 创建过多进程。
    params['s_gen']['replay_max_workers'] = (
        2 if sys.platform.startswith('win') else None
    )

    # False：某些序列流暂时无法与重放路径完全对应时，
    # 记录诊断信息而不立即中断整个优化过程。
    params['s_gen']['strict_gateway_counting'] = False

    # False：允许支持度为 0 的网关出口经过拉普拉斯平滑获得小概率；
    # True 会将这类情况视为错误，可能使稀疏日志的优化大量失败。
    params['s_gen']['strict_gateway_support'] = True

    # 输出每个网关出口的序列流计数、总支持度和最终概率。
    params['s_gen']['print_gateway_diagnostics'] = True
    # ===================== [新增配置结束] =====================

    # ========================================================
    # 到达间隔生成器参数
    # ========================================================
    params['i_gen'] = dict()
    params['i_gen']['batch_size'] = 32
    params['i_gen']['epochs'] = 300
    params['i_gen']['gen_method'] = ia_gen_method

    # --manual_ia_params: JSON 字符串或文件路径，跳过 Optuna 搜索
    if manual_ia_params:
        import os as _os
        if _os.path.isfile(manual_ia_params):
            with open(manual_ia_params, 'r', encoding='utf-8') as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(manual_ia_params)
        params['gl']['arrival_manual_best_params'] = parsed
        print(f"[pipeline] 手动 IA 参数已加载, 跳过 Optuna: {parsed}")

    # ========================================================
    # 时间预测器参数
    # ========================================================
    params['t_gen'] = dict()
    params['t_gen']['emb_method'] = emb_method
    params['t_gen']['concat_method'] = concat_method
    params['t_gen']['include_times'] = include_times
    params['t_gen']['model_type'] = 'dual_inter'
    params['t_gen']['all_r_pool'] = True
    params['t_gen']['reschedule'] = False
    params['t_gen']['rp_similarity'] = 0.8

    _ensure_locations(params)

    simulator = ds.DeepSimulator(params)
    simulator.execute_pipeline()


def read_properties(params):
    """读取 properties.yml 中的全局属性和路径。"""
    properties_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'properties.yml'
    )
    with open(properties_path, 'r', encoding='utf-8') as f:
        properties = yaml.load(f, Loader=yaml.FullLoader)

    if properties is None:
        raise ValueError('Properties is empty')

    paths = {
        key: os.path.join(*path.split('\\'))
        for key, path in properties.pop('paths').items()
    }
    params['gl'] = {**params['gl'], **properties, **paths}
    return params


def _ensure_locations(params):
    """创建程序运行所需的目录。"""
    required_folders = [
        'event_logs_path',
        'bpmn_models',
        'embedded_path',
        'ia_gen_path',
        'times_gen_path'
    ]

    for folder in required_folders:
        location = params['gl'][folder]
        if not os.path.exists(location):
            os.makedirs(location)

    if not os.path.exists(OUTPUT_FILES):
        os.makedirs(OUTPUT_FILES)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
