# -*- coding: utf-8 -*-
"""Execution-based Gateway Probability Estimator.

Branch probability strategy:
1. Extract every outgoing branch of each XOR split gateway.
2. Recursively obtain all downstream tasks belonging to each branch.
3. Sum downstream task execution counts for each branch:
       n_i = sum(executions(t)), t in branch_i
4. If N == 0 or N < confidence_threshold:
       p_i = 1 / K
5. Otherwise:
       p_i = (n_i + alpha) / (N + K * alpha)

This restores the original DSIM-style branch aggregation,
while retaining confidence-threshold fallback and Laplace smoothing.
"""

import numpy as np
import pandas as pd


class GatewaysEvaluator:

    def __init__(self, process_graph, method,
                 confidence_threshold=5, laplace_alpha=0.05):
        self.process_graph = process_graph
        self.method = method
        self.confidence_threshold = int(confidence_threshold)
        self.laplace_alpha = float(laplace_alpha)
        self.probabilities = []
        self.gateway_diagnostics = []
        self.unsupported_gateways = []
        self.define_probabilities()

    def _normalize(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        if x.sum() <= 0:
            return np.ones(len(x)) / len(x)
        return x / x.sum()

    def _find_tasks(self, node, visited=None):
        """Recursively find all downstream task nodes reachable from node."""
        if visited is None:
            visited = set()
        if node in visited:
            return []
        visited.add(node)

        node_type = self.process_graph.nodes[node].get("type")
        if node_type in ["task", "start", "end"]:
            return [node]

        result = []
        for nxt in self.process_graph.successors(node):
            result.extend(self._find_tasks(nxt, visited.copy()))
        return result

    def analize_gateway_structure(self):
        """Produce one row per gateway -> outgoing branch -> downstream task."""
        records = []

        for node in self.process_graph.nodes:
            if self.process_graph.nodes[node].get("type") != "gate":
                continue

            successors = list(self.process_graph.successors(node))
            if len(successors) <= 1:
                continue

            gate_bpmn_id = self.process_graph.nodes[node].get("id")

            for successor in successors:
                tasks = self._find_tasks(successor)
                edge_data = (self.process_graph.get_edge_data(node, successor)
                             or {})
                flow_id = edge_data.get("sf_id")
                target_bpmn_id = edge_data.get("target_bpmn_id")

                if not flow_id or not target_bpmn_id:
                    print("[Gateway Warning] Missing edge attributes: "
                          "gateway={}, target={}, sf_id={}, target_bpmn_id={}"
                          .format(gate_bpmn_id, successor,
                                  flow_id, target_bpmn_id))
                    continue

                if tasks:
                    for task in tasks:
                        records.append({
                            "gate": gate_bpmn_id,
                            "t_path": target_bpmn_id,
                            "target_node": successor,
                            "target_task": task,
                            "sf_id": flow_id,
                        })
                else:
                    self.unsupported_gateways.append({
                        "gate": gate_bpmn_id,
                        "t_path": target_bpmn_id,
                        "sf_id": flow_id,
                        "reason": "no downstream task found"
                    })
                    records.append({
                        "gate": gate_bpmn_id,
                        "t_path": target_bpmn_id,
                        "target_node": successor,
                        "target_task": None,
                        "sf_id": flow_id,
                    })

        if not records:
            gate_nodes = [
                (n, self.process_graph.nodes[n].get("type"),
                 self.process_graph.out_degree(n))
                for n in self.process_graph.nodes
                if "gate" in str(self.process_graph.nodes[n].get("type", ""))
            ]
            if gate_nodes:
                print("[Gateway structure] No multi-branch gateway splits found. "
                      "Gate-type nodes: {}".format(gate_nodes))
            else:
                node_types = set(self.process_graph.nodes[n].get("type")
                                 for n in self.process_graph.nodes)
                print("[Gateway structure] No gate-type nodes in process graph. "
                      "Node types: {}".format(node_types))

        return pd.DataFrame(records)

    def analize_gateways(self):
        """Estimate gateway branch probabilities using DSIM-style aggregation."""
        structure = self.analize_gateway_structure()
        if structure.empty:
            return pd.DataFrame()

        def get_execution(task):
            if task is None:
                return 0.0
            return float(self.process_graph.nodes[task].get("executions", 0))

        structure = structure.copy()
        structure["task_executions"] = structure["target_task"].apply(
            get_execution
        )

        branches = (
            structure
            .groupby(["gate", "t_path", "sf_id"], as_index=False)
            .agg(
                executions=("task_executions", "sum"),
                task_count=("target_task", lambda x: x.notna().sum()),
                target_tasks=("target_task",
                              lambda x: [t for t in x.tolist() if t is not None])
            )
        )

        records = []
        for gate, group in branches.groupby("gate", sort=False):
            group = group.reset_index(drop=True)
            counts = group["executions"].to_numpy(dtype=float)
            N = float(counts.sum())
            K = len(group)
            if K == 0:
                continue

            if N == 0 or N < self.confidence_threshold:
                probs = np.ones(K) / K
                mode = "uniform"
            else:
                alpha = self.laplace_alpha
                probs = (counts + alpha) / (N + K * alpha)
                mode = "laplace"

            probs = self._normalize(probs)

            diagnostic_branches = []
            for i in range(K):
                diagnostic_branches.append({
                    "t_path": group.loc[i, "t_path"],
                    "sf_id": group.loc[i, "sf_id"],
                    "tasks": group.loc[i, "target_tasks"],
                    "task_count": int(group.loc[i, "task_count"]),
                    "execution_sum": float(counts[i]),
                    "probability": float(probs[i]),
                })

            self.gateway_diagnostics.append({
                "gate": gate,
                "N": N,
                "K": K,
                "mode": mode,
                "counts": counts.tolist(),
                "probabilities": probs.tolist(),
                "branches": diagnostic_branches,
            })

            for i in range(K):
                records.append({
                    "gate": gate,
                    "t_path": group.loc[i, "t_path"],
                    "sf_id": group.loc[i, "sf_id"],
                    "executions": float(counts[i]),
                    "task_count": int(group.loc[i, "task_count"]),
                    "prob": float(probs[i]),
                })

        return pd.DataFrame(records)

    def define_probabilities(self):
        if self.method == "discovery":
            df = self.analize_gateways()

        elif self.method == "equiprobable":
            structure = self.analize_gateway_structure()
            if structure.empty:
                return
            df = (structure[["gate", "t_path", "sf_id"]]
                  .drop_duplicates()
                  .reset_index(drop=True))
            df["prob"] = 1.0 / df.groupby("gate")["gate"].transform("count")

        else:
            structure = self.analize_gateway_structure()
            if structure.empty:
                return
            df = (structure[["gate", "t_path", "sf_id"]]
                  .drop_duplicates()
                  .reset_index(drop=True))
            probabilities = []
            for _, group in df.groupby("gate", sort=False):
                random_values = np.random.random(len(group))
                random_values = self._normalize(random_values)
                probabilities.extend(random_values.tolist())
            df["prob"] = probabilities

        if df.empty:
            return

        df["gatewayid"] = df["gate"].astype(str)
        df["out_path_id"] = df["t_path"].astype(str)

        cols = ["gatewayid", "out_path_id", "prob"]
        if "sf_id" in df.columns:
            df["sf_id"] = df["sf_id"].astype(str)
            cols.append("sf_id")

        self.probabilities = df[cols].to_dict("records")

    def print_diagnostics(self):
        """Print gateway probability diagnostics."""
        if not self.gateway_diagnostics:
            print("[Gateway diagnostics] No gateways found.")
            return

        print("")
        print("=" * 80)
        print("Gateway probability diagnostics")
        print("=" * 80)

        for item in self.gateway_diagnostics:
            gate = item.get("gate", "?")
            mode = item.get("mode", "?")
            N = item.get("N", 0)
            print("")
            print("[Gateway {}] mode={}, total_support={}".format(
                gate, mode, int(N)))

            for index, branch in enumerate(item.get("branches", []), start=1):
                print("  Branch {}: flow={} | target={} | tasks={} | "
                      "execution_sum={} | prob={:.4f}".format(
                          index,
                          branch.get("sf_id"),
                          branch.get("t_path"),
                          branch.get("task_count"),
                          int(branch.get("execution_sum", 0)),
                          branch.get("probability", 0)))

        if self.unsupported_gateways:
            print("")
            print("[Unsupported gateways]")
            for item in self.unsupported_gateways:
                print("  {}".format(item))

        print("=" * 80)
