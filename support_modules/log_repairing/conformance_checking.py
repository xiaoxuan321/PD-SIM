# -*- coding: utf-8 -*-
"""
Evaluate complete token replay and optionally repair/remove non-conformant traces.

The reported percentage is explicitly a trace-level complete replay rate, not
continuous token-based fitness.
"""

from operator import itemgetter

from extraction import log_replayer as rpl
from support_modules.log_repairing import traces_replacement as rep
from support_modules.log_repairing import traces_alignment as tal
import utils.support as sup


def evaluate_alignment(process_graph, log, settings):
    traces = log.get_traces()

    test_replayer = rpl.LogReplayer(
        process_graph,
        traces,
        settings,
        msg="evaluating train partition token replay:",
    )

    one_timestamp = settings[
        "read_options"
    ]["one_timestamp"]

    conformant = get_traces(
        test_replayer.conformant_traces,
        one_timestamp,
    )

    not_conformant = get_traces(
        test_replayer.not_conformant_traces,
        one_timestamp,
    )

    print_stats(
        log,
        conformant,
        traces,
    )

    if settings["alg_manag"] == "replacement":
        log.set_data(
            rep.replacement(
                conformant,
                not_conformant,
                log,
                settings,
            )
        )

    elif settings["alg_manag"] == "repair":
        repaired_event_log = []

        for trace in conformant:
            repaired_event_log.extend(trace)

        trace_aligner = tal.TracesAligner(
            log,
            not_conformant,
            settings,
        )

        repaired_event_log.extend(
            trace_aligner.aligned_traces
        )

        log.set_data(repaired_event_log)

    elif settings["alg_manag"] == "removal":
        flattened = []

        for trace in conformant:
            flattened.extend(trace)

        log.set_data(flattened)

    else:
        raise ValueError(
            "Unsupported alg_manag: {}".format(
                settings["alg_manag"]
            )
        )

    aligned_traces = log.get_traces()

    if not aligned_traces:
        print(
            "complete traces: 0, events: 0"
        )
        print(
            "complete token-replay rate: 0.0%"
        )
        return

    test_replayer = rpl.LogReplayer(
        process_graph,
        aligned_traces,
        settings,
        msg=(
            "evaluating token replay after "
            + settings["alg_manag"]
            + ":"
        ),
    )

    conformant = get_traces(
        test_replayer.conformant_traces,
        one_timestamp,
    )

    print_stats(
        log,
        conformant,
        aligned_traces,
    )


def print_stats(log, conformant, traces):
    print(
        "complete traces:",
        len(traces),
        ", events:",
        len(log.data),
        sep=" ",
    )

    rate = (
        len(conformant) / len(traces) * 100.0
        if traces
        else 0.0
    )

    print(
        "complete token-replay rate:",
        str(sup.ffloat(rate, 2)) + "%",
        sep=" ",
    )


def get_traces(data, one_timestamp):
    cases = list(
        set(
            event["caseid"]
            for event in data
        )
    )

    traces = []

    for case in cases:
        order_key = (
            "end_timestamp"
            if one_timestamp
            else "start_timestamp"
        )

        trace = sorted(
            [
                event
                for event in data
                if event["caseid"] == case
            ],
            key=itemgetter(order_key),
        )

        traces.append(trace)

    return traces
