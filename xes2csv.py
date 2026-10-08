import pm4py
import pandas as pd


def xes_to_csv(xes_path, csv_path):
    """
    将XES格式的事件日志转换为CSV

    Args:
        xes_path: str, XES文件路径
        csv_path: str, 输出CSV文件路径
    """
    # 读取XES文件
    log = pm4py.read_xes(xes_path)

    # 转换为DataFrame
    df = pm4py.convert_to_dataframe(log)

    # 保存为CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ 转换完成: {csv_path}")
    print(f"📊 事件数: {len(df)}")
    print(f"📋 列名: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())

    return df


# 使用示例
df = xes_to_csv('E:\DPSIM\DP-SIM123-main\input_files\event_logs\Production.xes', 'E:\DPSIM\DP-SIM123-main\output_files\csvfile\output.csv')
