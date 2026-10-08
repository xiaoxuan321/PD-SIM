# -*- coding: utf-8 -*-
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd


class DirectSamplerEngine:
    """
    基于 BPMN token 语义的无时间戳活动序列采样器。
    目标：
    1) 替代 BIMP 在第一阶段的“无时间戳活动序列生成”
    2) 尽量保持与现有 gen_seqs 接口兼容
    3) 通过训练日志学习并行线性化偏好、局部转移偏好与长度先验
    """

    def __init__(
        self,
        process_graph,
        gateway_sequences=None,
        train_log=None,
        seed=42,
        max_steps_factor=6,
        parallel_policy="learned",
        temperature=0.8,
        repeat_quantile=0.95,
        w_bigram=1.8,
        w_precedence=1.2,
        w_prior=0.6,
        w_repeat_penalty=1.6,
        w_exit_bias=1.0,
    ):
        """
        参数说明：
        - process_graph: 由 BPMN 转成的有向图
        - gateway_sequences: StructureParametersMiner 提取出的 parameters['sequences']
        - train_log: 训练日志（DataFrame 或 LogReader），用于学习并行线性化偏好
        - parallel_policy: 并行任务线性化策略，建议 learned
        """
        self.g = process_graph
        self.gateway_sequences = gateway_sequences or []
        self.seed = seed
        self.rng = random.Random(seed)

        self.parallel_policy = parallel_policy
        self.temperature = temperature

        self.max_steps_factor = max_steps_factor
        self.repeat_quantile = repeat_quantile

        self.w_bigram = w_bigram
        self.w_precedence = w_precedence
        self.w_prior = w_prior
        self.w_repeat_penalty = w_repeat_penalty
        self.w_exit_bias = w_exit_bias

        # ---------- 图结构缓存 ----------
        self.node_data = dict(self.g.nodes(data=True))
        self.start_nodes = [n for n in self.g.nodes if self._is_start(n)]
        self.end_nodes = [n for n in self.g.nodes if self._is_end(n)]
        self.task_nodes = [n for n in self.g.nodes if self._is_task(n)]

        if not self.start_nodes:
            raise ValueError("process_graph 中未找到 start 节点")
        if not self.end_nodes:
            raise ValueError("process_graph 中未找到 end 节点")

        # ---------- 训练统计 ----------
        self.trace_len_dist = [8]
        self.task_freq = Counter()
        self.bigram = defaultdict(Counter)
        self.precedence = defaultdict(Counter)
        self.task_repeat_cap = defaultdict(lambda: 5)

        if train_log is not None:
            self._fit_statistics(train_log)

        self.max_steps = max(
            20,
            int((sum(self.trace_len_dist) / max(len(self.trace_len_dist), 1)) * self.max_steps_factor)
        )

        # ---------- 距离到 end 的启发信息 ----------
        self.dist_to_end = self._compute_dist_to_end()

        # ---------- 网关概率映射 ----------
        self.gateway_prob_map = self._build_gateway_prob_map(self.gateway_sequences)

    # =========================================================
    # 公共接口
    # =========================================================
    def sample_dataframe(self, num_cases, start_time=None, case_prefix="Case"):
        """
        生成与现有 phase1 接口兼容的 DataFrame：
        caseid, task, resource, start_timestamp, end_timestamp, pos_trace, trace_len
        注意：这里的时间戳是“伪时间戳”，仅用于兼容现有流程，
        后续 clean_time_stamps() 会删除。
        """
        start_dt = self._parse_start_time(start_time)

        rows = []
        for i in range(num_cases):
            target_len = self.rng.choice(self.trace_len_dist) if self.trace_len_dist else 8
            trace = self._sample_one_case(target_len=target_len)
            if not trace:
                continue

            caseid = f"{case_prefix}{i + 1}"
            case_start = start_dt + timedelta(seconds=i * 3)

            for pos, task_name in enumerate(trace, start=1):
                # 伪时间戳：只为了兼容现有接口
                ts = case_start + timedelta(seconds=pos)
                rows.append({
                    "caseid": caseid,
                    "task": task_name,
                    "resource": "SYSTEM",
                    "start_timestamp": ts,
                    "end_timestamp": ts + timedelta(milliseconds=1),
                    "pos_trace": pos,
                    "trace_len": len(trace),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["caseid", "pos_trace"]).reset_index(drop=True)
        return df

    # =========================================================
    # 训练统计学习
    # =========================================================
    def _fit_statistics(self, train_log):
        """
        从训练日志学习：
        1) trace 长度分布
        2) 任务先验频率
        3) 相邻 bigram 偏好
        4) 任务对先后偏好（用于并行线性化）
        5) 每个任务的重复次数上界
        """
        if hasattr(train_log, "data"):
            df = pd.DataFrame(train_log.data)
        else:
            df = train_log.copy()

        if df.empty:
            return

        # 统一字段名
        if "user" in df.columns and "resource" not in df.columns:
            df["resource"] = df["user"]

        # 去掉 Start / End
        if "task" in df.columns:
            df = df[~df["task"].isin(["Start", "End"])]

        # 排序
        sort_key = "start_timestamp" if "start_timestamp" in df.columns else None
        if sort_key is not None:
            df = df.sort_values(["caseid", sort_key]).reset_index(drop=True)
        else:
            df = df.sort_values(["caseid"]).reset_index(drop=True)

        grouped = list(df.groupby("caseid"))
        if not grouped:
            return

        self.trace_len_dist = []
        task_repeat_counter = defaultdict(list)

        for _, grp in grouped:
            seq = grp["task"].tolist()
            if not seq:
                continue

            self.trace_len_dist.append(len(seq))
            self.task_freq.update(seq)

            # bigram
            for a, b in zip(seq[:-1], seq[1:]):
                self.bigram[a][b] += 1

            # precedence（a 在 b 前面的次数）
            uniq_positions = defaultdict(list)
            for idx, t in enumerate(seq):
                uniq_positions[t].append(idx)

            tasks = list(uniq_positions.keys())
            for i, a in enumerate(tasks):
                for b in tasks[i + 1:]:
                    if min(uniq_positions[a]) < min(uniq_positions[b]):
                        self.precedence[a][b] += 1
                    else:
                        self.precedence[b][a] += 1

            # 统计每个任务在单条 trace 中的重复次数
            per_trace_count = Counter(seq)
            for t, c in per_trace_count.items():
                task_repeat_counter[t].append(c)

        # 每个任务的重复上界：用经验分位
        for t, arr in task_repeat_counter.items():
            arr = sorted(arr)
            if not arr:
                self.task_repeat_cap[t] = 5
                continue
            q_idx = int((len(arr) - 1) * self.repeat_quantile)
            self.task_repeat_cap[t] = max(1, arr[q_idx] + 1)

        # 平滑，避免 0 概率
        total_task = sum(self.task_freq.values())
        if total_task == 0:
            self.task_freq = Counter()

    # =========================================================
    # 单 case 采样
    # =========================================================
    def _sample_one_case(self, target_len=8):
        """
        采样一条 case 的活动序列。
        token 只在 BPMN 图上流动；
        当多个 task 同时可执行时，进行“学习驱动的线性化”。
        """
        # 初始化：从所有 start 节点后继开始
        tokens = []
        for s in self.start_nodes:
            for succ in self.g.successors(s):
                tokens.append(succ)

        join_buffer = defaultdict(int)
        trace = []
        visit_task = Counter()
        last_task = None
        step = 0

        # 先做一次闭包推进，尽量把非 task 节点展开
        tokens = self._closure(tokens, join_buffer, trace_len=len(trace), target_len=target_len, visit_task=visit_task)

        while tokens and step < self.max_steps:
            step += 1

            enabled_tasks = [n for n in tokens if self._is_task(n)]

            # 所有 token 都不在 task 上时，再推进一次
            if not enabled_tasks:
                tokens = self._closure(tokens, join_buffer, trace_len=len(trace), target_len=target_len, visit_task=visit_task)
                enabled_tasks = [n for n in tokens if self._is_task(n)]
                if not enabled_tasks:
                    break

            chosen_task = self._choose_enabled_task(
                enabled_tasks=enabled_tasks,
                trace=trace,
                last_task=last_task,
                visit_task=visit_task,
                target_len=target_len,
            )

            if chosen_task is None:
                break

            task_name = self._node_name(chosen_task)
            trace.append(task_name)
            visit_task[task_name] += 1
            last_task = task_name

            # 消耗被执行的 task token
            removed = False
            new_tokens = []
            for t in tokens:
                if (not removed) and t == chosen_task:
                    removed = True
                    continue
                new_tokens.append(t)
            tokens = new_tokens

            # 任务执行后，将后继压入 tokens
            for succ in self.g.successors(chosen_task):
                tokens.append(succ)

            # 再做闭包推进
            tokens = self._closure(tokens, join_buffer, trace_len=len(trace), target_len=target_len, visit_task=visit_task)

            # 长度保护：超过目标很多时，尽量结束
            if len(trace) >= max(target_len * 2, target_len + 10):
                break

        return trace

    # =========================================================
    # BPMN token 闭包推进
    # =========================================================
    def _closure(self, tokens, join_buffer, trace_len, target_len, visit_task):
        """
        将 token 从 start/gateway 等非 task 节点尽可能推进，
        返回“当前已启用 task token 列表 + 尚待 join 的 token”。
        """
        agenda = list(tokens)
        result_tokens = []

        while agenda:
            n = agenda.pop(0)

            # 已经是 task，保留，等待调度执行
            if self._is_task(n):
                result_tokens.append(n)
                continue

            # end：直接吞掉
            if self._is_end(n):
                continue

            # start 或普通事件：直接过
            if self._is_start(n) or self._is_passthrough_node(n):
                for succ in self.g.successors(n):
                    agenda.append(succ)
                continue

            # 网关处理
            if self._is_gateway(n):
                indeg = self.g.in_degree(n)
                outdeg = self.g.out_degree(n)

                # AND-join
                if self._is_parallel_gateway(n) and indeg > 1 and outdeg == 1:
                    join_buffer[n] += 1
                    if join_buffer[n] >= indeg:
                        join_buffer[n] -= indeg
                        for succ in self.g.successors(n):
                            agenda.append(succ)
                    continue

                # AND-split
                if self._is_parallel_gateway(n) and outdeg > 1:
                    for succ in self.g.successors(n):
                        agenda.append(succ)
                    continue

                # XOR-join：直接过
                if (not self._is_parallel_gateway(n)) and indeg > 1 and outdeg == 1:
                    for succ in self.g.successors(n):
                        agenda.append(succ)
                    continue

                # XOR-split：按概率采样
                if (not self._is_parallel_gateway(n)) and outdeg > 1:
                    succ = self._choose_gateway_successor(
                        gateway=n,
                        trace_len=trace_len,
                        target_len=target_len,
                        visit_task=visit_task,
                    )
                    if succ is not None:
                        agenda.append(succ)
                    continue

                # 其他 gateway：默认透传
                for succ in self.g.successors(n):
                    agenda.append(succ)
                continue

            # 其他未知节点：默认透传
            for succ in self.g.successors(n):
                agenda.append(succ)

        return result_tokens

    # =========================================================
    # 启用任务选择（并行线性化）
    # =========================================================
    def _choose_enabled_task(self, enabled_tasks, trace, last_task, visit_task, target_len):
        """
        当多个 task 同时可执行时，使用训练日志偏好进行线性化：
        - bigram 偏好
        - precedence 偏好
        - 任务先验
        - 重复惩罚
        - 接近目标长度时的退出偏置
        """
        if not enabled_tasks:
            return None
        if len(enabled_tasks) == 1:
            return enabled_tasks[0]

        names = [self._node_name(n) for n in enabled_tasks]
        scores = []

        total_task_freq = sum(self.task_freq.values()) + 1e-9

        for n in enabled_tasks:
            tn = self._node_name(n)
            score = 0.0

            # 1) bigram: 上一个任务到当前任务的局部偏好
            if last_task is not None:
                local = self.bigram[last_task][tn]
                denom = sum(self.bigram[last_task].values()) + len(self.bigram[last_task]) + 1e-9
                score += self.w_bigram * ((local + 1.0) / denom)

            # 2) precedence: 在当前并行任务集合中谁更可能先出现
            prec_sum = 0.0
            prec_cnt = 0
            for other in names:
                if other == tn:
                    continue
                ab = self.precedence[tn][other]
                ba = self.precedence[other][tn]
                if ab + ba > 0:
                    prec_sum += (ab + 1.0) / (ab + ba + 2.0)
                    prec_cnt += 1
            if prec_cnt > 0:
                score += self.w_precedence * (prec_sum / prec_cnt)

            # 3) 任务先验频率
            score += self.w_prior * ((self.task_freq[tn] + 1.0) / (total_task_freq + len(self.task_freq)))

            # 4) 重复惩罚
            repeat_cap = self.task_repeat_cap[tn]
            repeat_ratio = visit_task[tn] / max(repeat_cap, 1)
            score -= self.w_repeat_penalty * repeat_ratio

            # 5) 接近目标长度时，偏向更快走向结束的任务
            if len(trace) >= target_len:
                d = self.dist_to_end.get(n, 8)
                score += self.w_exit_bias * (1.0 / (1.0 + d))

            scores.append(score)

        return self._softmax_sample(enabled_tasks, scores, temperature=self.temperature)

    # =========================================================
    # XOR 分支选择
    # =========================================================
    def _choose_gateway_successor(self, gateway, trace_len, target_len, visit_task):
        succs = list(self.g.successors(gateway))
        if not succs:
            return None
        if len(succs) == 1:
            return succs[0]

        weights = []
        prob_map = self.gateway_prob_map.get(gateway, {})

        for succ in succs:
            base = prob_map.get(succ, 1.0 / len(succs))

            # 接近目标长度后，优先靠近结束节点的分支
            if trace_len >= target_len:
                d = self.dist_to_end.get(succ, 8)
                exit_bias = 1.0 / (1.0 + d)
            else:
                exit_bias = 1.0

            # 对明显回环分支加轻微惩罚：看最邻近任务是否已重复很多次
            near_task = self._nearest_task_name(succ)
            repeat_penalty = 1.0
            if near_task is not None:
                repeat_cap = self.task_repeat_cap[near_task]
                penalty_ratio = visit_task[near_task] / max(repeat_cap, 1)
                repeat_penalty = 1.0 / (1.0 + 0.8 * penalty_ratio)

            w = max(1e-9, base * exit_bias * repeat_penalty)
            weights.append(w)

        return self._weighted_choice(succs, weights)

    # =========================================================
    # 网关概率映射
    # =========================================================
    def _build_gateway_prob_map(self, sequences):
        """
        将 StructureParametersMiner.parameters['sequences'] 映射为：
        {gateway_node: {successor_node: prob}}
        由于 process_graph 的边属性命名可能不同，这里做了若干兼容尝试。
        """
        gp = defaultdict(dict)
        if not sequences:
            return gp

        # 建立 edge id -> successor 的索引
        edge_index = {}
        for u, v, data in self.g.edges(data=True):
            # 尝试若干常见字段
            for k in ["id", "elementid", "arc_id", "path_id", "name"]:
                if k in data:
                    edge_index[(u, data[k])] = v

        for seq in sequences:
            gw = seq.get("gatewayid")
            if gw is None:
                continue

            p = seq.get("prob") or seq.get("probability") or seq.get("value") or seq.get("percentage")
            if p is None:
                p = seq.get("probability_value", None)
            if p is None:
                p = 0.0

            # 优先用 out_path_id / elementid 映射到 successor
            succ = None
            out_path_id = seq.get("out_path_id")
            elementid = seq.get("elementid")

            # 1) out_path_id 直接就是后继节点
            if out_path_id in self.g.nodes:
                succ = out_path_id

            # 2) edge id 映射
            if succ is None and (gw, out_path_id) in edge_index:
                succ = edge_index[(gw, out_path_id)]
            if succ is None and (gw, elementid) in edge_index:
                succ = edge_index[(gw, elementid)]

            # 3) 如果只有一个 edge，退化匹配
            if succ is None:
                succs = list(self.g.successors(gw))
                if len(succs) == 1:
                    succ = succs[0]

            if succ is not None:
                gp[gw][succ] = float(p)

        # 概率归一化
        for gw, mp in gp.items():
            s = sum(mp.values())
            if s <= 0:
                succs = list(self.g.successors(gw))
                if succs:
                    val = 1.0 / len(succs)
                    gp[gw] = {x: val for x in succs}
            else:
                gp[gw] = {k: v / s for k, v in mp.items()}

        return gp

    # =========================================================
    # 图辅助函数
    # =========================================================
    def _compute_dist_to_end(self):
        dist = {}
        rev = self.g.reverse(copy=True)
        for n in self.g.nodes:
            best = 999
            for e in self.end_nodes:
                try:
                    d = nx.shortest_path_length(self.g, n, e)
                    best = min(best, d)
                except Exception:
                    pass
            dist[n] = best if best != 999 else 20
        return dist

    def _nearest_task_name(self, node, max_depth=6):
        """
        从某个节点往后找最近 task 的名字，用于 loop 惩罚启发。
        """
        q = [(node, 0)]
        seen = set()
        while q:
            n, depth = q.pop(0)
            if n in seen or depth > max_depth:
                continue
            seen.add(n)
            if self._is_task(n):
                return self._node_name(n)
            for succ in self.g.successors(n):
                q.append((succ, depth + 1))
        return None

    def _node_type(self, n):
        return str(self.node_data.get(n, {}).get("type", "")).lower()

    def _node_name(self, n):
        return self.node_data.get(n, {}).get("name", str(n))

    def _is_start(self, n):
        tp = self._node_type(n)
        return tp == "start" or "start" in tp

    def _is_end(self, n):
        tp = self._node_type(n)
        return tp == "end" or "end" in tp

    def _is_task(self, n):
        tp = self._node_type(n)
        return tp == "task" or "task" in tp

    def _is_gateway(self, n):
        tp = self._node_type(n)
        return tp.startswith("gate") or "gateway" in tp

    def _is_parallel_gateway(self, n):
        tp = self._node_type(n)
        # 兼容你现有代码中的 gate3 表示
        return tp == "gate3" or "parallel" in tp or "and_gateway" in tp

    def _is_passthrough_node(self, n):
        """
        对于某些中间事件/连接节点，默认透传。
        """
        tp = self._node_type(n)
        if self._is_task(n) or self._is_gateway(n) or self._is_start(n) or self._is_end(n):
            return False
        return True

    # =========================================================
    # 采样工具
    # =========================================================
    def _softmax_sample(self, items, scores, temperature=1.0):
        if not items:
            return None
        if len(items) == 1:
            return items[0]

        mx = max(scores)
        exps = [math.exp((s - mx) / max(temperature, 1e-6)) for s in scores]
        s = sum(exps)
        probs = [x / s for x in exps]
        return self._weighted_choice(items, probs)

    def _weighted_choice(self, items, weights):
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(items)
        r = self.rng.random() * total
        c = 0.0
        for item, w in zip(items, weights):
            c += w
            if r <= c:
                return item
        return items[-1]

    def _parse_start_time(self, start_time):
        if start_time is None:
            return datetime(2024, 1, 1, 0, 0, 0)
        if isinstance(start_time, datetime):
            return start_time
        if isinstance(start_time, str):
            # 兼容你当前代码里的格式："%Y-%m-%dT%H:%M:%S.%f+00:00"
            try:
                return datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f+00:00")
            except Exception:
                pass
            try:
                return datetime.fromisoformat(start_time.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
        return datetime(2024, 1, 1, 0, 0, 0)