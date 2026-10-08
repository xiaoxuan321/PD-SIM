# DP-SIM: **一种**结合流程挖掘与深度学习的业务流程仿真模型挖掘方法

A Business Process Simulation Model Discovery by Process Mining and Deep Learning

本工具支持以下任务：

- **训练生成式模型**：以事件日志为输入进行训练
- **生成完整事件日志**：基于已训练的生成式模型输出新日志
- **相似性评估**：评估原始日志与生成日志之间的相似度

## 快速开始（Getting Started）

### 先决条件（Prerequisites）

- 需在系统中安装 **Anaconda/Conda**
- 使用仓库提供的 `environment.yml` 创建运行环境

> 提示：通常可使用 `conda env create -f environment.yml` 创建环境，并通过 `conda activate <环境名>` 激活。

## 运行脚本（Running the script）

创建并激活环境后，可在终端中运行工具。至少需要指定输入事件日志文件名，并按需添加参数。

### 必需参数

- `--file`（必填）：XES 格式的事件日志文件。该文件需**预先放置**到 `input_files/event_logs` 目录下。

* ### 可选参数

  - `--update_gen / --no-update_gen`（可选，默认 `False`）
     是否更新此前发现的**序列生成模型**。若设置为 `--update_gen`，将更新模型。
  - `--update_ia_gen / --no-update_ia_gen`（可选，默认 `False`）
     是否更新此前已经生成的**案例开始时间生成模型**。若设置为 `--update_ia_gen`，将更新案例开始时间生成模型。
  - `--update_times_gen / --no-update_times_gen`（可选，默认 `False`）
     是否更新此前已经生成的**活动处理时间和等待时间生成模型**。若设置为 `--update_times_gen`，将更新模型。
  - `--save_models / --no-save_models`（可选，默认 `True`）
     是否保存训练得到的模型。
  - `--evaluate / --no-evaluate`（可选，默认 `True`）
     是否对最终仿真模型的准确性进行评估。

* ## 运行示例（Examples）

  **基础用法：**

  ```
  python .\pipeline.py --file Production.xes
  ```

  **更新“活动处理时间和等待时间生成模型”的用法：**

  ```
   python .\pipeline.py --file Production.xes --update_times_gen --t_gen_epochs 20 --t_gen_max_eval 3
  ```

  