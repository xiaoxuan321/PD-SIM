# -*- coding: utf-8 -*-
"""
Extract simulation parameters from token-replayed training traces.
"""

import math
import traceback
from collections import Counter, defaultdict

from extraction import gateways_probabilities as gt
from extraction import interarrival_definition as arr
from extraction import log_replayer as rpl
from extraction import schedule_tables as sch
from extraction import tasks_evaluator as te


class StructureParametersMiner(object):
    class Decorators(object):
        @classmethod
        def safe_exec(cls, method):
            def safety_check(*args, **kwargs):
                is_safe = kwargs.get(
                    "is_safe",
                    method.__name__.upper()
                )

                if is_safe:
                    try:
                        method(*args)
                    except Exception as error:
                        print(error)
                        traceback.print_exc()
                        is_safe = False

                return is_safe

            return safety_check

    def __init__(
        self,
        log,
        bpmn,
        process_graph,
        settings,
    ):
        self.log = log
        self.bpmn = bpmn
        self.process_graph = process_graph
        self.settings = settings
        self.settings["pdef_method"] = "default"

        self.process_stats = []
        self.parameters = {}
        self.conformant_traces = []
        self.is_safe = True

        self.gateway_edge_counts = Counter()
        self.gateway_diagnostics = []
        self.unsupported_gateways = []
        self.low_confidence_gateways = []

        self.replay_diagnostics = {
            "total_case_count": 0,
            "conformant_case_count": 0,
            "conformant_ratio": 0.0,
            "ambiguous_transition_count": 0,
            "ambiguous_transitions": Counter(),
        }

    def extract_parameters(
        self,
        num_inst,
        start_time,
        resource_pool,
    ):
        self.is_safe = self._replay_process(
            is_safe=self.is_safe
        )

        self.is_safe = self._mine_interarrival(
            is_safe=self.is_safe
        )

        self.is_safe = (
            self._mine_gateways_probabilities(
                is_safe=self.is_safe
            )
        )

        self.is_safe = self._process_tasks(
            resource_pool,
            is_safe=self.is_safe,
        )

        self.parameters["instances"] = num_inst
        self.parameters["start_time"] = start_time

    @Decorators.safe_exec
    def _replay_process(self):
        traces = self.log.get_traces()

        replayer = rpl.LogReplayer(
            self.process_graph,
            traces,
            self.settings,
            msg="reading conformant training traces:",
            mode=self.settings.get(
                "replay_mode",
                "multi"
            ),
            verbose=self.settings.get(
                "replay_verbose",
                True
            ),
            st=True,
        )

        if not hasattr(
            replayer,
            "gateway_edge_counts"
        ):
            raise RuntimeError(
                "Current LogReplayer does not expose "
                "gateway_edge_counts."
            )

        self.process_stats = replayer.process_stats

        if (
            self.process_stats is None
            or self.process_stats.empty
        ):
            raise ValueError(
                "LogReplayer produced no valid process_stats."
            )

        self.process_stats = (
            self.process_stats.copy()
        )

        self.process_stats["role"] = "SYSTEM"
        self.conformant_traces = (
            replayer.conformant_traces
        )
        self.gateway_edge_counts = Counter(
            replayer.gateway_edge_counts
        )

        ambiguous_transitions = Counter(
            getattr(
                replayer,
                "ambiguous_transitions",
                Counter(),
            )
        )

        ambiguous_count = int(
            getattr(
                replayer,
                "ambiguous_transition_count",
                sum(
                    ambiguous_transitions.values()
                ),
            )
        )

        conformant_case_count = int(
            getattr(
                replayer,
                "conformant_case_count",
                0,
            )
        )

        total_case_count = len(traces)

        conformant_ratio = (
            conformant_case_count
            / total_case_count
            if total_case_count
            else 0.0
        )

        self.replay_diagnostics = {
            "total_case_count": total_case_count,
            "conformant_case_count": (
                conformant_case_count
            ),
            "conformant_ratio": conformant_ratio,
            "ambiguous_transition_count": (
                ambiguous_count
            ),
            "ambiguous_transitions": (
                ambiguous_transitions
            ),
        }

        min_ratio = self.settings.get(
            "min_conformant_ratio"
        )

        if (
            min_ratio is not None
            and conformant_ratio
            < float(min_ratio)
        ):
            raise RuntimeError(
                "Complete token-replay rate is too low: "
                "{}/{} ({:.2%}) < {:.2%}".format(
                    conformant_case_count,
                    total_case_count,
                    conformant_ratio,
                    float(min_ratio),
                )
            )

        xor_splits = [
            node
            for node in self.process_graph.nodes
            if (
                self.process_graph.nodes[node].get(
                    "gateway_type"
                ) == "exclusive"
                or self.process_graph.nodes[node].get(
                    "type"
                ) == "gate"
            )
            and self.process_graph.out_degree(node) > 1
        ]

        if (
            self.settings.get(
                "gate_management"
            ) == "discovery"
            and xor_splits
            and not self.gateway_edge_counts
        ):
            raise RuntimeError(
                "The model contains {} XOR split gateways "
                "but replay produced no unambiguous branch "
                "counts. Traces may still be conformant, "
                "but gateway probabilities are not "
                "identifiable.".format(
                    len(xor_splits)
                )
            )

        print(
            "[Token replay] conformant_cases={}/{}, "
            "rate={:.2%}, counted_edges={}, "
            "ambiguous={}".format(
                conformant_case_count,
                total_case_count,
                conformant_ratio,
                len(self.gateway_edge_counts),
                ambiguous_count,
            )
        )

        if ambiguous_count > 0:
            print(
                "[Replay note] Multiple valid replay "
                "histories do not make a trace "
                "non-conformant. Ambiguous XOR branch "
                "counts are excluded from probability "
                "estimation."
            )

            if self.settings.get(
                "print_replay_diagnostics",
                False,
            ):
                print(
                    "[Replay diagnostics] {}".format(
                        dict(
                            ambiguous_transitions
                        )
                    )
                )

    @staticmethod
    def mine_resources(settings, log):
        parameters = {}

        settings["res_cal_met"] = "default"
        settings["res_dtype"] = "247"
        settings["arr_cal_met"] = "default"
        settings["arr_dtype"] = "247"

        creator = sch.TimeTablesCreator(
            settings
        )

        creator.create_timetables({
            "res_cal_met": (
                settings["res_cal_met"]
            ),
            "arr_cal_met": (
                settings["arr_cal_met"]
            ),
        })

        resource_pool = [{
            "id": "QBP_DEFAULT_RESOURCE",
            "name": "SYSTEM",
            "total_amount": "100000",
            "costxhour": "20",
            "timetable_id": (
                creator.res_ttable_name[
                    "arrival"
                ]
            ),
        }]

        parameters["resource_pool"] = (
            resource_pool
        )
        parameters["time_table"] = (
            creator.time_table
        )

        return parameters

    @Decorators.safe_exec
    def _mine_interarrival(self):
        evaluator = arr.InterArrivalEvaluator(
            self.process_graph,
            self.conformant_traces,
            self.settings,
        )

        self.parameters[
            "arrival_rate"
        ] = evaluator.dist

    @Decorators.safe_exec
    def _mine_gateways_probabilities(self):
        evaluator = gt.GatewaysEvaluator(
            self.process_graph,
            self.settings["gate_management"],
            confidence_threshold=(
                self.settings.get(
                    "confidence_threshold",
                    5
                )
            ),
            laplace_alpha=self.settings.get(
                "laplace_alpha",
                0.05
            ),
        )
        sequences = [
            dict(sequence)
            for sequence
            in evaluator.probabilities
        ]

        self.gateway_diagnostics = list(
            getattr(
                evaluator,
                "gateway_diagnostics",
                [],
            )
        )

        self.unsupported_gateways = list(
            getattr(
                evaluator,
                "unsupported_gateways",
                [],
            )
        )

        self.low_confidence_gateways = [
            item["gate"]
            for item in self.gateway_diagnostics
            if item.get(
                "low_confidence",
                False
            )
        ]

        if self.unsupported_gateways:
            print(
                "[Gateway warning] {} XOR split "
                "gateways have no unambiguous "
                "training support.".format(
                    len(
                        self.unsupported_gateways
                    )
                )
            )

            print(
                "[Unsupported gateways] {}".format(
                    self.unsupported_gateways
                )
            )

            if self.settings.get(
                "strict_gateway_support",
                False,
            ):
                raise RuntimeError(
                    "strict_gateway_support=True "
                    "but {} gateways have no "
                    "training support.".format(
                        len(
                            self.unsupported_gateways
                        )
                    )
                )

        max_unsupported_ratio = (
            self.settings.get(
                "max_unsupported_gateway_ratio"
            )
        )

        if (
            max_unsupported_ratio is not None
            and self.gateway_diagnostics
        ):
            unsupported_ratio = (
                len(self.unsupported_gateways)
                / len(self.gateway_diagnostics)
            )

            if unsupported_ratio > float(
                max_unsupported_ratio
            ):
                raise RuntimeError(
                    "Unsupported XOR gateway ratio "
                    "{:.2%} exceeds {:.2%}".format(
                        unsupported_ratio,
                        float(
                            max_unsupported_ratio
                        ),
                    )
                )

        mapped_element_ids = set()
        probability_sums = defaultdict(float)

        for sequence in sequences:
            gateway_id = sequence["gatewayid"]
            out_path_id = sequence["out_path_id"]

            try:
                probability = float(
                    sequence["prob"]
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid gateway probability: "
                    "gateway={}, path={}, prob={}"
                    .format(
                        gateway_id,
                        out_path_id,
                        sequence.get("prob"),
                    )
                )

            if (
                not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    "Gateway probability out of "
                    "range: gateway={}, path={}, "
                    "prob={}".format(
                        gateway_id,
                        out_path_id,
                        probability,
                    )
                )

            # 优先使用预存储的 sf_id（来自边属性）
            element_id = sequence.get("sf_id", "")
            if not element_id:
                # 回退到 BPMN 查找
                element_id = (
                    self.bpmn.find_sequence_id(
                        gateway_id,
                        out_path_id,
                    )
                )

            if element_id in (None, ""):
                raise ValueError(
                    "Cannot map gateway branch to "
                    "BPMN sequenceFlow: gatewayid={}, "
                    "out_path_id={}".format(
                        gateway_id,
                        out_path_id,
                    )
                )

            if element_id in mapped_element_ids:
                raise ValueError(
                    "Multiple gateway branches map "
                    "to the same sequenceFlow: "
                    "{}".format(element_id)
                )

            sequence["prob"] = probability
            sequence["elementid"] = element_id
            mapped_element_ids.add(element_id)
            probability_sums[
                gateway_id
            ] += probability

        for gateway_id, total in (
            probability_sums.items()
        ):
            if not math.isclose(
                total,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "Gateway {} branch probabilities "
                    "sum to {}, not 1.".format(
                        gateway_id,
                        total,
                    )
                )

        self.parameters[
            "sequences"
        ] = sequences

        print(
            "[Gateway probabilities] gateways={}, "
            "sequence_flows={}, low_confidence={}, "
            "unsupported={}".format(
                len(probability_sums),
                len(sequences),
                len(
                    self.low_confidence_gateways
                ),
                len(
                    self.unsupported_gateways
                ),
            )
        )

        if self.settings.get(
            "print_gateway_diagnostics",
            False,
        ):
            evaluator.print_diagnostics()

    @Decorators.safe_exec
    def _process_tasks(self, resource_pool):
        evaluator = te.TaskEvaluator(
            self.process_graph,
            self.process_stats,
            resource_pool,
            self.settings,
        )

        self.parameters[
            "elements_data"
        ] = evaluator.elements_data
