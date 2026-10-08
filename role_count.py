import pandas as pd
import pm4py
import matplotlib.pyplot as plt
from extraction import role_discovery as rl  # 引用您提供的模块

# ==========================================
# 1. 配置与数据加载 (支持 XES)
# ==========================================
LOG_PATH = 'E:\DPSIM-1\DP-SIM123\input_files\event_logs\Production.xes'  # 请修改为您的 XES 文件路径
SIMILARITY_RANGE = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def analyze_xes_mp_roles(path):
    print(f"正在加载 XES 日志: {path} ...")
    try:
        # 读取 XES 并转换为 DataFrame
        log = pm4py.read_xes(path)
        df = pm4py.convert_to_dataframe(log)

        # 列名映射：将 XES 标准属性映射到您的代码要求的字段
        # concept:name -> task, org:resource -> user
        mapping = {
            'concept:name': 'task',
            'org:resource': 'user',
            'case:concept:name': 'caseid'
        }
        df.rename(columns=mapping, inplace=True)

        # 预处理：填充缺失资源 (参考您源码中的 'sys' 填充逻辑)
        df['user'] = df['user'].fillna('sys')

    except Exception as e:
        print(f"错误：读取 XES 失败 - {e}")
        return

    print(f"--- MP (XES) 日志初步统计 ---")
    print(f"总事件数: {len(df)}")
    print(f"唯一活动数: {df['task'].nunique()}")
    print(f"总资源数: {df['user'].nunique()}\n")

    results = []

    # ==========================================
    # 2. 核心分析循环
    # ==========================================
    print(f"{'阈值 (rp_similarity)':<20} | {'角色数量':<10} | {'平均资源/角色'}")
    print("-" * 55)

    for sim in SIMILARITY_RANGE:
        # 调用资源池分析器
        res_analyzer = rl.ResourcePoolAnalyser(df, sim_threshold=sim)

        # 获取结果表
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)

        # 统计
        role_counts = resource_table.groupby('role')['resource'].count()
        num_roles = len(role_counts)
        avg_res = role_counts.mean()

        results.append({
            'similarity': sim,
            'num_roles': num_roles,
            'role_dist': role_counts.to_dict()
        })

        print(f"{sim:<20.2f} | {num_roles:<10} | {avg_res:.2f}")

    return results


# ==========================================
# 3. 运行分析
# ==========================================
if __name__ == "__main__":
    analysis_results = analyze_xes_mp_roles(LOG_PATH)

    if analysis_results:
        # 可视化趋势
        sims = [r['similarity'] for r in analysis_results]
        roles = [r['num_roles'] for r in analysis_results]

        plt.figure(figsize=(10, 6))
        plt.plot(sims, roles, marker='s', color='teal', linewidth=2)
        plt.axhline(y=1, color='r', linestyle=':', label='Single Pool')
        plt.title('Role Discovery Sensitivity Analysis (MP XES Log)')
        plt.xlabel('rp_similarity Threshold')
        plt.ylabel('Detected Roles Count')
        plt.grid(True, alpha=0.3)
        plt.show()