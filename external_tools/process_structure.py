# -*- coding: utf-8 -*-
"""
Build a NetworkX process graph from BPMN.

Main changes:
1. Use real BPMN startEvent/endEvent as process boundaries.
2. Keep visible tasks as type='task' even when their names are Start/End.
3. Preserve gateway_type and gateway_direction explicitly.
4. Do not emulate parallel gateway state with gt_num_paths/executed counters.
5. For backward compatibility, bypass artificial Task(Start)/Task(End) when
   they are directly attached to real BPMN boundary events.
"""

import networkx as nx
import utils.support as sup


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return list(value)


def _pick(item, keys, default=None):
    if not isinstance(item, dict):
        return default
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def _event_id(item, kind):
    value = _pick(
        item,
        ["{}_id".format(kind), "event_id", "node_id", "id"]
    )
    if value is None:
        raise ValueError(
            "Cannot determine BPMN {} event id from {}".format(kind, item)
        )
    return value


def _event_name(item, fallback):
    return _pick(
        item,
        ["event_name", "name", "label", "task_name"],
        fallback
    )


def create_process_structure(bpmn, verbose=True):
    graph = load_process_structure(bpmn, verbose)
    if verbose:
        sup.print_done_task()
    return graph


def find_node_num(graph, element_id):
    for node in graph.nodes:
        if graph.nodes[node].get("id") == element_id:
            return node
    return -1


def _add_node(graph, index, node_type, name, element_id, **extra):
    attrs = {
        "type": node_type,
        "name": name,
        "id": element_id,
        "executions": 0,
        "processing_times": [],
        "waiting_times": [],
        "multi_tasking": [],
        "temp_enable": None,
        "temp_start": None,
        "temp_end": None,
        "tsk_act": False,
        "gtact": False,
        "xor_gtdir": 0,
        # Retained only for compatibility with old code that may read them.
        "gt_num_paths": 0,
        "gt_visited_paths": 0,
    }
    attrs.update(extra)
    graph.add_node(index, **attrs)
    return index + 1


def _progress(verbose, index, total):
    if not verbose:
        return
    sup.print_progress(
        min((index / max(total, 1)) * 100.0, 100.0),
        "Loading of bpmn structure from file "
    )


def _bypass_artificial_boundary_tasks(graph):
    """
    Older discovery logs sometimes contained visible activities named Start/End.
    When a BPMN also contains real start/end events, bypass only the synthetic
    boundary tasks directly adjacent to those events.
    """
    start_nodes = [
        n for n in graph.nodes
        if graph.nodes[n].get("type") == "start"
    ]
    end_nodes = [
        n for n in graph.nodes
        if graph.nodes[n].get("type") == "end"
    ]

    if not start_nodes or not end_nodes:
        return

    remove_nodes = []

    for node in list(graph.nodes):
        data = graph.nodes[node]
        if data.get("type") != "task":
            continue

        name = data.get("name")
        preds = list(graph.predecessors(node))
        succs = list(graph.successors(node))

        if (
            name == "Start"
            and preds
            and all(graph.nodes[p].get("type") == "start" for p in preds)
        ):
            for p in preds:
                for s in succs:
                    graph.add_edge(p, s)
            remove_nodes.append(node)

        elif (
            name == "End"
            and succs
            and all(graph.nodes[s].get("type") == "end" for s in succs)
        ):
            for p in preds:
                for s in succs:
                    graph.add_edge(p, s)
            remove_nodes.append(node)

    if remove_nodes:
        graph.remove_nodes_from(remove_nodes)


def load_process_structure(bpmn, verbose=True):
    graph = nx.DiGraph()

    start_events = _as_list(bpmn.get_start_event_info())
    tasks = _as_list(bpmn.get_tasks_info())
    ex_gates = _as_list(bpmn.get_ex_gates_info())
    inc_gates = _as_list(bpmn.get_inc_gates_info())
    para_gates = _as_list(bpmn.get_para_gates_info())
    end_events = _as_list(bpmn.get_end_event_info())
    timer_events = _as_list(bpmn.get_timer_events_info())
    edges = _as_list(bpmn.get_edges_info())

    total = (
        len(start_events)
        + len(tasks)
        + len(ex_gates)
        + len(inc_gates)
        + len(para_gates)
        + len(end_events)
        + len(timer_events)
    )

    index = 0

    for item in start_events:
        _progress(verbose, index, total)
        index = _add_node(
            graph,
            index,
            "start",
            _event_name(item, "Start"),
            _event_id(item, "start"),
        )

    for item in tasks:
        _progress(verbose, index, total)
        task_id = _pick(item, ["task_id", "id", "node_id"])
        if task_id is None:
            raise ValueError("Cannot determine task id from {}".format(item))
        index = _add_node(
            graph,
            index,
            "task",
            _pick(item, ["task_name", "name", "label"], ""),
            task_id,
        )

    for item in ex_gates:
        _progress(verbose, index, total)
        gate_id = _pick(item, ["gate_id", "id", "node_id"])
        if gate_id is None:
            raise ValueError(
                "Cannot determine exclusive gateway id from {}".format(item)
            )
        direction = _pick(
            item,
            ["gate_dir", "direction"],
            "Unspecified"
        )
        legacy_type = "gate" if direction == "Diverging" else "gate2"
        index = _add_node(
            graph,
            index,
            legacy_type,
            _pick(item, ["gate_name", "name", "label"], ""),
            gate_id,
            gateway_type="exclusive",
            gateway_direction=direction,
        )

    for item in inc_gates:
        _progress(verbose, index, total)
        gate_id = _pick(item, ["gate_id", "id", "node_id"])
        if gate_id is None:
            raise ValueError(
                "Cannot determine inclusive gateway id from {}".format(item)
            )
        index = _add_node(
            graph,
            index,
            "gate2",
            _pick(item, ["gate_name", "name", "label"], ""),
            gate_id,
            gateway_type="inclusive",
            gateway_direction=_pick(
                item,
                ["gate_dir", "direction"],
                "Unspecified"
            ),
        )

    for item in para_gates:
        _progress(verbose, index, total)
        gate_id = _pick(item, ["gate_id", "id", "node_id"])
        if gate_id is None:
            raise ValueError(
                "Cannot determine parallel gateway id from {}".format(item)
            )
        index = _add_node(
            graph,
            index,
            "gate3",
            _pick(item, ["gate_name", "name", "label"], ""),
            gate_id,
            gateway_type="parallel",
            gateway_direction=_pick(
                item,
                ["gate_dir", "direction"],
                "Unspecified"
            ),
        )

    for item in timer_events:
        _progress(verbose, index, total)
        timer_id = _pick(
            item,
            ["timer_id", "event_id", "id", "node_id"]
        )
        if timer_id is None:
            raise ValueError(
                "Cannot determine timer event id from {}".format(item)
            )
        index = _add_node(
            graph,
            index,
            "timer",
            _pick(
                item,
                ["timer_name", "event_name", "name", "label"],
                ""
            ),
            timer_id,
        )

    for item in end_events:
        _progress(verbose, index, total)
        index = _add_node(
            graph,
            index,
            "end",
            _event_name(item, "End"),
            _event_id(item, "end"),
        )

    for edge in edges:
        source_num = find_node_num(graph, edge["source"])
        target_num = find_node_num(graph, edge["target"])
        if source_num != -1 and target_num != -1:
            graph.add_edge(
                source_num,
                target_num,
                sf_id=edge.get("sf_id", ""),
                target_bpmn_id=edge["target"],
            )

    _bypass_artificial_boundary_tasks(graph)

    starts = [
        n for n in graph.nodes
        if graph.nodes[n].get("type") == "start"
    ]
    ends = [
        n for n in graph.nodes
        if graph.nodes[n].get("type") == "end"
    ]

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            "Process graph must contain exactly one BPMN start event and "
            "one BPMN end event; got start={}, end={}".format(
                len(starts), len(ends)
            )
        )

    return graph
