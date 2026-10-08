# -*- coding: utf-8 -*-
"""
Dual processing-time / waiting-time predictor.

This revision keeps the original D-SIM timing semantics:
1. The processing-time model directly predicts activity duration.
2. The waiting-time model directly predicts the delay before the next activity
   becomes ready.
3. The resource scheduler only postpones a ready activity when its role has no
   free resource.

It deliberately removes runtime-profile compression, resource-wait subtraction,
calendar forcing, wait-percentile smoothing, and case-duration post-calibration.
"""

import json
import os
import uuid
import warnings
from datetime import timedelta
from enum import Enum
from pickle import load

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from core_modules.times_allocator import entities as en


class InstanceState(Enum):
    WAITING = 1
    INEXECUTION = 2
    COMPLETE = 3


class DualIntercasesPredictor:
    """Generate timestamps with separate processing and waiting models."""

    def __init__(self, model_path, parms):
        self.parms = parms

        self.execution_state = None
        self.queue = None
        self.ac_dict = None
        self.rl_dict = None
        self.sequences = None

        self.end_inter_scaler = None
        self.inter_scaler = None
        self.scaler = None

        self.g1, self.first_session, self.proc_model = self._load_model(model_path[0])
        self.g2, self.second_session, self.wait_model = self._load_model(model_path[1])

        self.n_feat_proc = self._feature_count(self.proc_model)
        self.n_feat_wait = self._feature_count(self.wait_model)

    @staticmethod
    def _feature_count(model):
        shape = model.get_layer("features").output_shape
        if isinstance(shape, list):
            shape = shape[0]
        return int(shape[2])

    @staticmethod
    def _load_model(path):
        """Load a Keras model in its own TF1-compatible graph/session."""
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session()
            with session.as_default():
                try:
                    model = load_model(path, compile=False)
                except Exception as first_error:
                    try:
                        from tensorflow.keras.optimizers import Adam

                        class LegacyAdam(Adam):
                            def __init__(self, *args, weight_decay=None, **kwargs):
                                kwargs.pop("weight_decay", None)
                                super().__init__(*args, **kwargs)

                        model = load_model(
                            path,
                            compile=False,
                            custom_objects={"Adam": LegacyAdam},
                        )
                    except Exception:
                        raise RuntimeError(
                            "Unable to load time model: {}".format(path)
                        ) from first_error
        return graph, session, model

    def predict(self, sequences, iarr):
        """Prepare simulation state and generate the timestamped event log."""
        metadata = self._load_metadata()

        self.ac_index = metadata["ac_index"]
        self.index_ac = {value: key for key, value in self.ac_index.items()}
        n_size = int(metadata["n_size"])

        role_task = pd.DataFrame(metadata["roles_table"])
        role_table = pd.DataFrame(
            [
                {
                    "role_name": role_name,
                    "size": len(resources),
                    "role_index": index,
                }
                for index, (role_name, resources) in enumerate(
                    metadata["roles"].items()
                )
            ]
        )

        inter_states = metadata["inter_mean_states"]
        initial_wip = self._initial_wip(inter_states)
        initial_activity_states = self._initial_activity_states(inter_states)

        self._load_scalers()

        extra_short, sequences = self._filter_extra_short_cases(sequences)
        _, long_iarr = self._filter_short_iarr(extra_short, iarr)

        self.sequences, num_elements = self._encode_sequences(
            sequences,
            self.ac_index,
            role_task,
            role_table,
            max_trace_length=self.parms.get("max_trace_length"),
        )

        self.rl_dict = self._initialize_roles(
            role_table,
            check_avail=self.parms.get("reschedule", True),
        )
        self.ac_dict = self._initialize_activities(
            self.ac_index,
            initial_activity_states,
        )
        self.queue = self._initialize_queue(long_iarr)
        self.execution_state = self._initialize_execution_state(self.sequences)

        return self._generate(initial_wip, n_size, num_elements)

    def _load_metadata(self):
        suffix = "_diapr" if self.parms.get("all_r_pool", False) else "_dispr"
        metadata_path = os.path.join(
            self.parms["times_gen_path"],
            self.parms["file"].split(".")[0] + suffix + "_meta.json",
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "Time-model metadata was not found: {}".format(metadata_path)
            )
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _load_scalers(self):
        suffix = "_diapr" if self.parms.get("all_r_pool", False) else "_dispr"
        stem = os.path.join(
            self.parms["times_gen_path"],
            self.parms["file"].split(".")[0] + suffix,
        )
        with open(stem + "_scaler.pkl", "rb") as scaler_file:
            self.scaler = load(scaler_file)
        with open(stem + "_inter_scaler.pkl", "rb") as scaler_file:
            self.inter_scaler = load(scaler_file)
        with open(stem + "_end_inter_scaler.pkl", "rb") as scaler_file:
            self.end_inter_scaler = load(scaler_file)

    def _initial_wip(self, inter_states):
        """
        Preserve the D-SIM default, while allowing a clean empty-system run.

        Set parms['initial_state_mode'] = 'empty' to start WIP/activity counters
        at zero. The default 'mean' reproduces the original D-SIM behaviour.
        """
        mode = self.parms.get("initial_state_mode", "mean")
        if mode == "empty":
            return 0
        if mode != "mean":
            raise ValueError("initial_state_mode must be 'mean' or 'empty'")
        return int(round(float(inter_states.get("wip", 0))))

    def _initial_activity_states(self, inter_states):
        if self.parms.get("initial_state_mode", "mean") == "empty":
            return {}
        return inter_states.get("tasks", {})

    def _generate(self, initial_wip, n_size, num_elements):
        event_log = []
        open_events = {}
        active_instances = {}
        process_wip = initial_wip

        progress = tqdm(total=num_elements, desc="generating traces:")

        while not self.queue.get_all().empty():
            element = self.queue.get_remove_first()
            if not element:
                continue

            case_id = element["caseid"]
            if case_id not in self.execution_state:
                continue

            action = element["action"]

            if action == "create_instance":
                transitions = self.execution_state[case_id]["transitions"]
                if not transitions:
                    self.execution_state[case_id]["state"] = InstanceState.COMPLETE
                    continue

                transition = transitions.pop(0)
                self.execution_state[case_id]["state"] = InstanceState.INEXECUTION
                self.queue.add(
                    {
                        "timestamp": element["timestamp"],
                        "action": "create_activity",
                        "caseid": case_id,
                        "transition": transition,
                    }
                )
                process_wip += 1
                active_instances[case_id] = en.ProcessInstance(
                    case_id,
                    n_size,
                    (self.n_feat_proc, self.n_feat_wait),
                    dual=True,
                    n_act=True,
                )

            elif action == "create_activity":
                transition = element["transition"]
                role_index = transition[1]
                role = self.rl_dict[role_index]

                # Critical fix: do not update the LSTM history while an activity
                # is merely retrying for a resource.
                if self.parms.get("reschedule", True) and not self._role_has_free_resource(role):
                    next_release = role.get_next_release()
                    if next_release is None:
                        raise RuntimeError(
                            "Role {} has no free resource and no release event".format(
                                role.get_name()
                            )
                        )
                    element["timestamp"] = max(
                        element["timestamp"],
                        next_release + timedelta(microseconds=1),
                    )
                    self.queue.add(element)
                    continue

                activity_wip = self.ac_dict[transition[0]].get_active_instances()
                role_occupancy = self._role_occupancy_features(role_index)
                scaled_wip = self.inter_scaler.transform(
                    np.asarray([[process_wip, activity_wip]], dtype=np.float64)
                )[0]

                instance = active_instances[case_id]
                instance.update_proc_ngram(
                    transition[0],
                    element["timestamp"],
                    scaled_wip,
                    role_occupancy,
                )
                activity_ngram, feature_ngram = instance.get_proc_ngram()

                raw_processing = self._predict_scalar(
                    self.proc_model,
                    self.g1,
                    self.first_session,
                    activity_ngram,
                    feature_ngram,
                )
                processing_seconds = self._inverse_time_seconds(
                    raw_processing,
                    target="processing_time",
                )
                instance.update_proc(raw_processing)

                start_timestamp = element["timestamp"]
                end_timestamp = start_timestamp + timedelta(
                    seconds=processing_seconds
                )
                resource_id = role.assign_resource(end_timestamp)

                # This should be impossible after the availability check because
                # the event loop is single-threaded. Keep a defensive failure
                # instead of silently corrupting the n-gram.
                if resource_id is None:
                    raise RuntimeError(
                        "Resource availability changed during activity creation"
                    )

                event_id = "event_" + str(uuid.uuid4())
                open_events[event_id] = {
                    "pr_instances": process_wip,
                    "tsk_start_inst": activity_wip,
                    "res_id": resource_id,
                    "start_timestamp": start_timestamp,
                }
                if not self.parms.get("all_r_pool", False):
                    open_events[event_id]["rp_start_oc"] = role_occupancy[0]

                self.ac_dict[transition[0]].add_act()

                element.update(
                    {
                        "timestamp": end_timestamp,
                        "action": "complete_activity",
                        "ev_id": event_id,
                    }
                )
                self.queue.add(element)

            elif action == "complete_activity":
                transition = element["transition"]
                event_id = element["ev_id"]
                event = open_events.pop(event_id)
                resource_id = event["res_id"]

                self.rl_dict[transition[1]].release_resource(resource_id)
                self.ac_dict[transition[0]].remove_act()

                event_log.append(
                    {
                        "caseid": case_id,
                        "task": self.index_ac[transition[0]],
                        "resource": resource_id,
                        "role": self.rl_dict[transition[1]].get_name(),
                        "end_timestamp": element["timestamp"],
                        **event,
                    }
                )
                event_log[-1].pop("res_id", None)

                transitions = self.execution_state[case_id]["transitions"]
                if transitions:
                    next_transition = transitions.pop(0)
                    element["transition"] = next_transition

                    # Keep the training semantics used by D-SIM: rp_end_oc is
                    # measured at completion of the current activity. When
                    # all_r_pool=True this vector already contains every role.
                    end_role_occupancy = self._role_occupancy_features(
                        transition[1]
                    )
                    scaled_end_wip = self.end_inter_scaler.transform(
                        np.asarray([[process_wip]], dtype=np.float64)
                    )[0]

                    instance = active_instances[case_id]
                    instance.update_wait_ngram(
                        next_transition[0],
                        element["timestamp"],
                        scaled_end_wip,
                        end_role_occupancy,
                    )
                    next_activity_ngram, feature_ngram = instance.get_wait_ngram()

                    raw_waiting = self._predict_scalar(
                        self.wait_model,
                        self.g2,
                        self.second_session,
                        next_activity_ngram,
                        feature_ngram,
                    )
                    waiting_seconds = self._inverse_time_seconds(
                        raw_waiting,
                        target="waiting_time",
                    )
                    instance.update_wait(raw_waiting)

                    # D-SIM semantics: the waiting model directly determines
                    # when the next activity becomes ready.
                    element["timestamp"] = element["timestamp"] + timedelta(
                        seconds=waiting_seconds
                    )
                    element["action"] = "create_activity"
                else:
                    element["action"] = "complete_instance"

                progress.update(1)
                self.queue.add(element)

            elif action == "complete_instance":
                self.execution_state[case_id]["state"] = InstanceState.COMPLETE
                process_wip -= 1
                active_instances.pop(case_id, None)

            else:
                raise ValueError("Unknown simulation action: {}".format(action))

        progress.close()
        return event_log

    def _role_occupancy_features(self, role_index):
        if self.parms.get("all_r_pool", False):
            return [
                self.rl_dict[index].get_occupancy()
                for index in range(len(self.rl_dict))
            ]
        return [self.rl_dict[role_index].get_occupancy()]

    @staticmethod
    def _role_has_free_resource(role):
        pool = role.get_resource_pool()
        return any(item.get("release_time") is None for item in pool.values())

    @staticmethod
    def _predict_scalar(model, graph, session, activity_ngram, feature_ngram):
        activity_input = np.asarray([activity_ngram], dtype=np.int32)
        feature_input = np.asarray(feature_ngram, dtype=np.float32)

        if not np.isfinite(feature_input).all():
            raise ValueError("Non-finite value found in time-model features")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with graph.as_default(), session.as_default():
                prediction = model.predict(
                    {
                        "ac_input": activity_input,
                        "features": feature_input,
                    },
                    verbose=0,
                )
        return float(np.asarray(prediction).reshape(-1)[0])

    def _inverse_time_seconds(self, raw_prediction, target):
        """Convert a model output back to non-negative seconds."""
        transformed = np.zeros((1, 2), dtype=np.float64)
        target_index = 0 if target == "processing_time" else 1
        transformed[0, target_index] = float(raw_prediction)

        original = self.scaler.inverse_transform(transformed)
        seconds = float(original[0, target_index])
        if not np.isfinite(seconds):
            raise ValueError(
                "Invalid {} prediction after inverse transform".format(target)
            )
        return max(0.0, seconds)

    @staticmethod
    def _filter_short_iarr(extra_short, iarr):
        short_case_ids = set(extra_short["caseid"].unique())
        long_iarr = {
            row["caseid"]: row["timestamp"]
            for row in iarr.to_dict("records")
            if row["caseid"] not in short_case_ids
        }
        short_iarr = {
            row["caseid"]: row["timestamp"]
            for row in iarr.to_dict("records")
            if row["caseid"] in short_case_ids
        }
        return short_iarr, long_iarr

    @staticmethod
    def _filter_extra_short_cases(sequences):
        valid_case_ids = set(
            sequences.loc[
                ~sequences["task"].isin(["Start", "End"]),
                "caseid",
            ].unique()
        )
        all_case_ids = set(sequences["caseid"].unique())
        extra_short_ids = all_case_ids - valid_case_ids

        extra_short = sequences[sequences["caseid"].isin(extra_short_ids)].copy()
        filtered = sequences[sequences["caseid"].isin(valid_case_ids)].copy()
        filtered = filtered[~filtered["task"].isin(["Start", "End"])].copy()
        return extra_short, filtered

    @staticmethod
    def _initialize_activities(activity_index, initial_states):
        return {
            index: en.ActivityCounter(
                activity,
                index=index,
                initial=int(round(float(initial_states.get(activity, 0)))),
            )
            for activity, index in activity_index.items()
            if activity not in ["Start", "End"]
        }

    @staticmethod
    def _initialize_roles(role_table, check_avail):
        return {
            int(role["role_index"]): en.Role(
                role["role_name"],
                max(1, int(role["size"])),
                index=int(role["role_index"]),
                check_avail=check_avail,
            )
            for role in role_table.to_dict("records")
        }

    @staticmethod
    def _initialize_queue(iarr):
        simulation_queue = en.Queue()
        for case_id, timestamp in iarr.items():
            simulation_queue.add(
                {
                    "timestamp": timestamp,
                    "action": "create_instance",
                    "caseid": case_id,
                }
            )
        return simulation_queue

    @staticmethod
    def _initialize_execution_state(sequences):
        return {
            case_id: {
                "state": InstanceState.WAITING,
                "transitions": list(transitions),
            }
            for case_id, transitions in sequences.items()
        }

    @staticmethod
    def _encode_sequences(
        sequences,
        activity_index,
        role_task,
        role_table,
        max_trace_length=None,
    ):
        sequence_frame = sequences.copy()
        default_role = role_table.loc[
            role_table["size"].idxmax(),
            "role_name",
        ]

        sequence_frame["ac_index"] = sequence_frame["task"].map(activity_index)
        if sequence_frame["ac_index"].isna().any():
            unknown = sorted(
                sequence_frame.loc[
                    sequence_frame["ac_index"].isna(),
                    "task",
                ].astype(str).unique()
            )
            raise KeyError(
                "Activities missing from time-model metadata: {}".format(unknown)
            )

        sequence_frame = sequence_frame.merge(
            role_task,
            how="left",
            on="task",
        )
        sequence_frame["role"] = sequence_frame["role"].fillna(default_role)
        sequence_frame = sequence_frame.merge(
            role_table,
            how="left",
            left_on="role",
            right_on="role_name",
        )
        if sequence_frame["role_index"].isna().any():
            raise ValueError("Unable to assign a role to one or more activities")

        sequence_frame["ac_rl"] = list(
            zip(
                sequence_frame["ac_index"].astype(int),
                sequence_frame["role_index"].astype(int),
            )
        )

        encoded = {}
        num_elements = 0
        for case_id, group in sequence_frame.sort_values(
            ["caseid", "pos_trace"]
        ).groupby("caseid", sort=False):
            trace = group["ac_rl"].tolist()
            if max_trace_length is not None:
                trace = trace[: int(max_trace_length)]
            encoded[case_id] = trace
            num_elements += len(trace)

        return encoded, num_elements
