# -*- coding: utf-8 -*-
"""
Token/marking based BPMN log replay.

Core rule:
    multiple legal paths / duplicate labels != non-conformance

A trace is conformant when at least one legal token execution consumes all
visible log events and reaches the final marking.

Supported:
- sequence flow
- XOR split/join
- AND split/join
- loops composed from these constructs
- duplicate task labels resolved by current marking
- silent timer/intermediate pass-through nodes

True inclusive OR-join semantics are not approximated as XOR.  Such a gateway
is reported as unsupported so that incorrect conformance results are not hidden.
"""

import itertools
import multiprocessing
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm


class UnsupportedGatewayError(RuntimeError):
    pass


class ReplayStateExplosion(RuntimeError):
    pass


@dataclass
class ReplayState:
    # sequence-flow edge -> list of token ready timestamps
    tokens: dict = field(default_factory=dict)
    gateway_edge_counts: Counter = field(default_factory=Counter)
    task_counts: Counter = field(default_factory=Counter)
    mapped_nodes: list = field(default_factory=list)
    enabling_times: list = field(default_factory=list)
    ambiguous: bool = False
    end_count: int = 0


def _clone_state(state):
    return ReplayState(
        tokens={
            edge: list(values)
            for edge, values in state.tokens.items()
        },
        gateway_edge_counts=Counter(state.gateway_edge_counts),
        task_counts=Counter(state.task_counts),
        mapped_nodes=list(state.mapped_nodes),
        enabling_times=list(state.enabling_times),
        ambiguous=state.ambiguous,
        end_count=state.end_count,
    )


def _add_token(state, edge, ready_time):
    state.tokens.setdefault(edge, []).append(ready_time)


def _token_count(state, edge):
    return len(state.tokens.get(edge, []))


def _pop_token(state, edge):
    values = state.tokens.get(edge)
    if not values:
        raise RuntimeError("No token on edge {}".format(edge))

    ready_time = values.pop(0)
    if not values:
        state.tokens.pop(edge, None)
    return ready_time


def _marking_key(state):
    return (
        tuple(
            sorted(
                (source, target, len(values))
                for (source, target), values in state.tokens.items()
                if values
            )
        ),
        min(state.end_count, 1),
    )


def _history_signature(state):
    return (
        tuple(sorted(state.gateway_edge_counts.items())),
        tuple(sorted(state.task_counts.items())),
        tuple(state.mapped_nodes),
    )


def _deduplicate_states(states, max_states):
    """
    Merge equal markings.

    Equal markings reached through different histories remain valid, but the
    merged state is marked ambiguous so gateway branch attribution is not
    treated as uniquely observed.
    """
    by_key = {}

    for state in states:
        key = _marking_key(state)

        if key not in by_key:
            by_key[key] = state
            continue

        existing = by_key[key]
        if _history_signature(existing) != _history_signature(state):
            existing.ambiguous = True
        if state.ambiguous:
            existing.ambiguous = True

    result = list(by_key.values())

    if len(result) > max_states:
        raise ReplayStateExplosion(
            "Replay state count {} exceeds replay_max_states={}".format(
                len(result), max_states
            )
        )

    return result


def _gateway_kind(model, node):
    data = model.nodes[node]
    explicit = data.get("gateway_type")
    if explicit:
        return explicit

    # Compatibility with old process graph node types.
    node_type = data.get("type")
    if node_type in ("gate", "gate2"):
        return "exclusive"
    if node_type == "gate3":
        return "parallel"
    return None


def _latest_ready_time(values):
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _fire_timer(model, state, node):
    incoming = list(model.predecessors(node))
    outgoing = list(model.successors(node))
    results = []

    for source in incoming:
        edge = (source, node)
        if _token_count(state, edge) <= 0:
            continue

        new_state = _clone_state(state)
        ready_time = _pop_token(new_state, edge)

        for target in outgoing:
            _add_token(
                new_state,
                (node, target),
                ready_time
            )

        results.append(new_state)

    return results


def _fire_end(model, state, node):
    results = []

    for source in model.predecessors(node):
        edge = (source, node)

        if _token_count(state, edge) <= 0:
            continue

        new_state = _clone_state(state)
        _pop_token(new_state, edge)
        new_state.end_count += 1
        results.append(new_state)

    return results


def _fire_exclusive_gateway(model, state, gate):
    """
    XOR semantics:
    - join: one marked incoming flow is sufficient
    - split: choose exactly one outgoing flow
    """
    incoming = list(model.predecessors(gate))
    outgoing = list(model.successors(gate))
    results = []

    enabled_sources = [
        source
        for source in incoming
        if _token_count(state, (source, gate)) > 0
    ]

    for source in enabled_sources:
        if outgoing:
            for target in outgoing:
                new_state = _clone_state(state)
                ready_time = _pop_token(
                    new_state,
                    (source, gate)
                )
                _add_token(
                    new_state,
                    (gate, target),
                    ready_time
                )

                if len(outgoing) > 1:
                    new_state.gateway_edge_counts[
                        (gate, target)
                    ] += 1

                results.append(new_state)
        else:
            new_state = _clone_state(state)
            _pop_token(new_state, (source, gate))
            results.append(new_state)

    return results


def _fire_parallel_gateway(model, state, gate):
    """
    AND semantics:
    - join: all incoming flows must contain a token
    - split: produce one token on every outgoing flow
    """
    incoming = list(model.predecessors(gate))
    outgoing = list(model.successors(gate))

    if not incoming:
        return []

    if len(incoming) > 1:
        if not all(
            _token_count(state, (source, gate)) > 0
            for source in incoming
        ):
            return []

        new_state = _clone_state(state)
        consumed_times = [
            _pop_token(new_state, (source, gate))
            for source in incoming
        ]
        ready_time = _latest_ready_time(consumed_times)

        for target in outgoing:
            _add_token(
                new_state,
                (gate, target),
                ready_time
            )

        return [new_state]

    source = incoming[0]
    edge = (source, gate)

    if _token_count(state, edge) <= 0:
        return []

    new_state = _clone_state(state)
    ready_time = _pop_token(new_state, edge)

    for target in outgoing:
        _add_token(
            new_state,
            (gate, target),
            ready_time
        )

    return [new_state]


def _fire_silent_node(model, state, node):
    """
    Fire an invisible BPMN node.

    Supported invisible node types:
    - silent:
        Boundary wrapper tasks such as
        startEvent -> task("Start") -> ...
        ... -> task("End") -> endEvent

        These nodes do not consume any visible log event.
        They only pass the token from incoming flow
        to outgoing flow.

    - timer:
        Silent intermediate timer/pass-through node.

    - end:
        BPMN end event.

    - exclusive gateway:
        XOR split / join.

    - parallel gateway:
        AND split / join.

    Inclusive gateways are deliberately not approximated as XOR.
    """

    node_type = model.nodes[node].get("type")

    # ============================================================
    # 1. Boundary silent task
    #
    # process_structure.py 中识别出来的伪 Start / End task
    # 会被标记为：
    #
    #     type = "silent"
    #
    # 它们不应该和日志中的 activity 进行匹配，
    # 只需要把 token 原样传递到后继节点。
    #
    # 当前 _fire_timer() 的行为本身就是：
    # incoming token -> silent pass-through -> outgoing token
    #
    # 所以可以直接复用 _fire_timer()。
    # ============================================================
    if node_type == "silent":
        return _fire_timer(
            model,
            state,
            node
        )

    # ============================================================
    # 2. Timer / intermediate silent event
    # ============================================================
    if node_type == "timer":
        return _fire_timer(
            model,
            state,
            node
        )

    # ============================================================
    # 3. BPMN End Event
    # ============================================================
    if node_type == "end":
        return _fire_end(
            model,
            state,
            node
        )

    # ============================================================
    # 4. Gateway
    # ============================================================
    kind = _gateway_kind(
        model,
        node
    )

    # XOR gateway
    if kind == "exclusive":
        return _fire_exclusive_gateway(
            model,
            state,
            node
        )

    # AND / Parallel gateway
    if kind == "parallel":
        return _fire_parallel_gateway(
            model,
            state,
            node
        )

    # ============================================================
    # 5. Inclusive gateway
    #
    # 当前 replay 实现没有真正实现 BPMN OR-join 语义，
    # 因此继续保持原来的行为：明确报 unsupported，
    # 不要错误地把 inclusive 当 XOR。
    # ============================================================
    if kind == "inclusive":
        raise UnsupportedGatewayError(
            "Inclusive gateway {} requires true OR-join semantics; "
            "it is not approximated as XOR.".format(node)
        )

    # ============================================================
    # 6. Unknown node
    #
    # 如果节点不是已支持的 silent/control-flow 类型，
    # 当前 marking 下不执行它。
    # ============================================================
    return []


def _silent_nodes(model):
    return [
        node
        for node in model.nodes
        if model.nodes[node].get("type")
        in {"gate", "gate2", "gate3", "timer", "silent", "end"}
    ]


def _silent_closure(model, states, max_states, max_steps):
    """
    Explore states reachable by invisible BPMN nodes until no additional
    invisible transition can fire.

    This is the replacement for unique-shortest-path navigation.
    """
    queue = list(states)
    seen = {}
    terminal = []
    steps = 0
    silent_nodes = _silent_nodes(model)

    while queue:
        state = queue.pop()
        steps += 1

        if steps > max_steps:
            raise ReplayStateExplosion(
                "Silent replay exceeds replay_max_silent_steps={}".format(
                    max_steps
                )
            )

        key = _marking_key(state)

        if key in seen:
            existing = seen[key]
            if _history_signature(existing) != _history_signature(state):
                existing.ambiguous = True
            if state.ambiguous:
                existing.ambiguous = True
            continue

        seen[key] = state
        successors = []

        for node in silent_nodes:
            successors.extend(
                _fire_silent_node(model, state, node)
            )

        if successors:
            queue.extend(
                _deduplicate_states(successors, max_states)
            )
            if len(queue) > max_states:
                queue = _deduplicate_states(
                    queue,
                    max_states
                )
        else:
            terminal.append(state)

    return _deduplicate_states(terminal, max_states)


def _initial_state(model):
    starts = [
        node
        for node in model.nodes
        if model.nodes[node].get("type") == "start"
    ]

    if len(starts) != 1:
        raise ValueError(
            "Process graph must have exactly one BPMN start event; "
            "got {}".format(len(starts))
        )

    start = starts[0]
    outgoing = list(model.successors(start))

    if not outgoing:
        raise ValueError("BPMN start event has no outgoing flow")

    state = ReplayState()

    # Multiple outgoing flows from a start event are concurrent.
    for target in outgoing:
        _add_token(
            state,
            (start, target),
            None
        )

    return state


def _enabled_task_nodes(model, state, label):
    """
    Duplicate labels are allowed.  A task is a candidate only when:
        node.name == log activity label
        AND one of its incoming flows currently contains a token.
    """
    result = []

    for node in model.nodes:
        data = model.nodes[node]

        if data.get("type") != "task":
            continue

        if data.get("name") != label:
            continue

        if any(
            _token_count(state, (source, node)) > 0
            for source in model.predecessors(node)
        ):
            result.append(node)

    return result


def _default_enable_time(event):
    if event.get("start_timestamp") is not None:
        return event["start_timestamp"]
    return event["end_timestamp"]


def _fire_task(model, state, node, event):
    """
    Fire one visible task.

    If a task has several incoming flows without an explicit gateway,
    one marked input is sufficient (uncontrolled merge).

    If a task has several outgoing flows without an explicit gateway,
    a token is produced on every output (parallel split).
    """
    incoming = list(model.predecessors(node))
    outgoing = list(model.successors(node))
    results = []

    for source in incoming:
        edge = (source, node)

        if _token_count(state, edge) <= 0:
            continue

        new_state = _clone_state(state)
        ready_time = _pop_token(new_state, edge)

        if ready_time is None:
            ready_time = _default_enable_time(event)

        for target in outgoing:
            _add_token(
                new_state,
                (node, target),
                event["end_timestamp"]
            )

        new_state.task_counts[node] += 1
        new_state.mapped_nodes.append(node)
        new_state.enabling_times.append(ready_time)
        results.append(new_state)

    return results


def _is_final_state(state):
    return (
        state.end_count > 0
        and not any(state.tokens.values())
    )


def _build_time_records(trace, final_state):
    records = []

    for event, node, enable_time in zip(
        trace,
        final_state.mapped_nodes,
        final_state.enabling_times,
    ):
        record = {
            "caseid": event["caseid"],
            "task": event["task"],
            "end_timestamp": event["end_timestamp"],
            "enable_timestamp": enable_time,
            "resource": event["user"],
            "_mapped_node": node,
        }

        if event.get("start_timestamp") is not None:
            record["start_timestamp"] = event["start_timestamp"]

        if record["resource"] != "AUTO":
            records.append(record)

    return records


class LogReplayer(object):
    def __init__(
        self,
        model,
        log,
        settings,
        msg="",
        source="log",
        run_num=0,
        verbose=True,
        mode="multi",
        st=True,
    ):
        self.source = source
        self.run_num = run_num
        self.one_timestamp = settings["read_options"]["one_timestamp"]
        self.model = model
        self.m_data = pd.DataFrame.from_dict(
            dict(model.nodes.data()),
            orient="index"
        )
        self.msg = msg
        self.verbose = verbose
        self.settings = settings

        self.start_tasks_list = []
        self.end_tasks_list = []
        self.find_start_finish_tasks()

        self.not_conformant_traces = []
        self.conformant_traces = []
        self.process_stats = []
        self.traces = log

        self.gateway_edge_counts = Counter()
        self.task_execution_counts = Counter()
        self.ambiguous_transitions = Counter()
        self.ambiguous_transition_count = 0
        self.conformant_case_count = 0

        self._replay_traces(mode, st)

    def _replay_traces(self, mode, st, **kwargs):
        replay_config = {
            "max_states": int(
                self.settings.get(
                    "replay_max_states",
                    5000
                )
            ),
            "max_silent_steps": int(
                self.settings.get(
                    "replay_max_silent_steps",
                    20000
                )
            ),
        }

        args = [
            (i, trace, self.model, st, replay_config)
            for i, trace in enumerate(self.traces)
        ]
        size = len(args)

        if mode == "multi" and self.verbose:
            cpu_count = multiprocessing.cpu_count()

            with tqdm(total=size, desc=self.msg) as pbar:
                with Pool(processes=cpu_count) as pool:
                    future = pool.map_async(
                        self.replay_trace,
                        args
                    )
                    processed = 0

                    while not future.ready():
                        current = size - future._number_left
                        if current > processed:
                            pbar.update(current - processed)
                            processed = current

                    pbar.update(size - processed)
                    results = future.get()

        elif mode == "multi":
            with Pool(
                processes=multiprocessing.cpu_count()
            ) as pool:
                results = pool.map(
                    self.replay_trace,
                    args
                )

        elif mode == "seq" and self.verbose:
            results = []
            with tqdm(total=size, desc=self.msg) as pbar:
                for arg in args:
                    results.append(
                        self.replay_trace(arg)
                    )
                    pbar.update(1)

        elif mode == "seq":
            results = [
                self.replay_trace(arg)
                for arg in args
            ]

        else:
            raise ValueError(
                "Unsupported replay mode: {}".format(mode)
            )

        self.gateway_edge_counts = Counter()
        self.task_execution_counts = Counter()
        self.ambiguous_transitions = Counter()

        for result in results:
            self.ambiguous_transitions.update(result[5])

            if result[0]:
                self.gateway_edge_counts.update(result[3])
                self.task_execution_counts.update(result[4])

        self.ambiguous_transition_count = int(
            sum(self.ambiguous_transitions.values())
        )
        self.conformant_case_count = sum(
            1 for result in results if result[0]
        )

        for node in self.model.nodes:
            if self.model.nodes[node].get("type") == "task":
                self.model.nodes[node]["executions"] = int(
                    self.task_execution_counts.get(node, 0)
                )

        self.process_stats = (
            [result[2] for result in results if result[0]]
            if st else []
        )

        if st:
            self.process_stats = list(
                itertools.chain(*self.process_stats)
            )

        self.conformant_traces = [
            self.traces[result[1]]
            for result in results
            if result[0]
        ]
        self.conformant_traces = list(
            itertools.chain(*self.conformant_traces)
        )

        self.not_conformant_traces = [
            self.traces[result[1]]
            for result in results
            if not result[0]
        ]
        self.not_conformant_traces = list(
            itertools.chain(*self.not_conformant_traces)
        )

        if (
            self.verbose
            and self.ambiguous_transition_count > 0
        ):
            print(
                "[LogReplayer] {} replay ambiguities/limitations detected. "
                "A trace remains conformant when at least one legal token "
                "execution exists. Ambiguous XOR branch attribution is not "
                "used for gateway probability estimation.".format(
                    self.ambiguous_transition_count
                )
            )

        if self.conformant_traces:
            if st:
                self.calculate_process_metrics()
        else:
            raise AssertionError(
                "Model not valid for testing"
            )

    @staticmethod
    def replay_trace(args):
        index, original_trace, model, st, config = args

        max_states = config["max_states"]
        max_silent_steps = config["max_silent_steps"]
        ambiguities = Counter()

        def fail(reason):
            ambiguities[
                ("TRACE", str(index), reason)
            ] += 1

            return (
                False,
                index,
                [],
                Counter(),
                Counter(),
                ambiguities,
            )

        trace = list(original_trace)

        # Backward compatibility with old internal logs.
        if trace and trace[0].get("task") == "Start":
            trace = trace[1:]

        if trace and trace[-1].get("task") == "End":
            trace = trace[:-1]

        if not trace:
            return fail("empty-visible-trace")

        try:
            states = [_initial_state(model)]

            states = _silent_closure(
                model,
                states,
                max_states,
                max_silent_steps,
            )

            for event in trace:
                next_states = []

                for state in states:
                    candidates = _enabled_task_nodes(
                        model,
                        state,
                        event["task"],
                    )

                    for node in candidates:
                        next_states.extend(
                            _fire_task(
                                model,
                                state,
                                node,
                                event,
                            )
                        )

                if not next_states:
                    return fail(
                        "no-enabled-task:{}".format(
                            event["task"]
                        )
                    )

                states = _silent_closure(
                    model,
                    _deduplicate_states(
                        next_states,
                        max_states
                    ),
                    max_states,
                    max_silent_steps,
                )

            states = _silent_closure(
                model,
                states,
                max_states,
                max_silent_steps,
            )

            completed = [
                state
                for state in states
                if _is_final_state(state)
            ]

            if not completed:
                return fail(
                    "cannot-reach-final-marking"
                )

        except UnsupportedGatewayError:
            return fail(
                "unsupported-inclusive-gateway"
            )

        except ReplayStateExplosion:
            return fail(
                "state-space-explosion"
            )

        except Exception as exc:
            return fail(
                "replay-exception:{}:{}".format(
                type(exc).__name__,
                str(exc)
                )
            )

        # At least one valid execution => conformant trace.
        selected = completed[0]

        edge_signatures = {
            tuple(
                sorted(
                    state.gateway_edge_counts.items()
                )
            )
            for state in completed
        }

        task_signatures = {
            tuple(
                sorted(
                    state.task_counts.items()
                )
            )
            for state in completed
        }

        ambiguous_path = (
            len(edge_signatures) > 1
            or any(
                state.ambiguous
                for state in completed
            )
        )

        if ambiguous_path:
            ambiguities[
                (
                    "TRACE",
                    str(index),
                    "multiple-valid-replays",
                )
            ] += 1
            edge_counts = Counter()
        else:
            edge_counts = Counter(
                selected.gateway_edge_counts
            )

        # If duplicate-label replay maps to different task nodes, do not make
        # an arbitrary per-node execution attribution.
        if len(task_signatures) > 1:
            ambiguities[
                (
                    "TRACE",
                    str(index),
                    "ambiguous-task-mapping",
                )
            ] += 1
            task_counts = Counter()
        else:
            task_counts = Counter(
                selected.task_counts
            )

        time_records = (
            _build_time_records(trace, selected)
            if st else []
        )

        return (
            True,
            index,
            time_records,
            edge_counts,
            task_counts,
            ambiguities,
        )

    def calculate_process_metrics(self):
        stats = pd.DataFrame(self.process_stats)

        if stats.empty:
            self.process_stats = stats
            return

        stats = stats[
            ~stats.task.isin(["Start", "End"])
        ]
        stats = stats[
            stats.resource != "AUTO"
        ]
        stats["source"] = self.source
        stats["run_num"] = self.run_num

        if self.one_timestamp:
            stats["duration"] = (
                stats["end_timestamp"]
                - stats["enable_timestamp"]
            )
            stats["duration"] = (
                stats["duration"].dt.total_seconds()
            )

        else:
            records = stats.to_dict("records")

            for record in records:
                duration = (
                    record["end_timestamp"]
                    - record["start_timestamp"]
                ).total_seconds()

                waiting = (
                    record["start_timestamp"]
                    - record["enable_timestamp"]
                ).total_seconds()

                multitask = 0

                if waiting < 0:
                    waiting = 0

                    if (
                        record["end_timestamp"]
                        > record["enable_timestamp"]
                    ):
                        duration = (
                            record["end_timestamp"]
                            - record["enable_timestamp"]
                        ).total_seconds()

                        multitask = (
                            record["enable_timestamp"]
                            - record["start_timestamp"]
                        ).total_seconds()

                    else:
                        multitask = duration

                record["processing_time"] = duration
                record["waiting_time"] = waiting
                record["multitasking"] = multitask
                record.pop("_mapped_node", None)

            stats = pd.DataFrame(records)

        if "_mapped_node" in stats.columns:
            stats = stats.drop(
                columns=["_mapped_node"]
            )

        self.process_stats = stats

    def find_start_finish_tasks(self):
        starts = [
            node
            for node in self.model.nodes
            if self.model.nodes[node].get("type") == "start"
        ]
        ends = [
            node
            for node in self.model.nodes
            if self.model.nodes[node].get("type") == "end"
        ]

        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                "Process graph must contain exactly one "
                "start and one end node"
            )

        self.start_tasks_list = self.find_next_tasks(
            self.model,
            starts[0],
        )

        self.end_tasks_list = self.find_next_tasks(
            self.model.reverse(copy=True),
            ends[0],
        )

    @staticmethod
    def find_next_tasks(model, num, visited=None):
        visited = (
            set()
            if visited is None
            else set(visited)
        )

        if num in visited:
            return []

        visited.add(num)
        result = []

        for node in model.neighbors(num):
            node_type = model.nodes[node].get("type")

            if node_type == "task":
                result.append(node)

            elif node_type in {"start", "end"}:
                continue

            else:
                result.extend(
                    LogReplayer.find_next_tasks(
                        model,
                        node,
                        visited,
                    )
                )

        return list(dict.fromkeys(result))
