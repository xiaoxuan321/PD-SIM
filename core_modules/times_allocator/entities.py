# -*- coding: utf-8 -*-
"""Simulation entities used by the time allocator."""

import itertools
import queue
import random
import uuid
from queue import PriorityQueue

import numpy as np


class Queue:
    """Chronological event queue with deterministic tie breaking."""

    def __init__(self):
        self._queue = PriorityQueue()
        self._counter = itertools.count()

    def add(self, element):
        self._queue.put(
            (
                element["timestamp"],
                next(self._counter),
                element,
            )
        )

    def get_remove_first(self):
        try:
            return self._queue.get(block=False)[2]
        except queue.Empty:
            return {}

    def get_all(self):
        return self._queue


class Role:
    """Resource role and its resource pool."""

    def __init__(self, name, size, index=0, check_avail=True):
        size = max(1, int(size))
        self._num_resources = size
        self._resource_pool = self._initialize_resources(size)
        self._name = name
        self._index = index
        self._check_avail = check_avail
        self._execution = 0

    def assign_resource(self, release_time):
        if self._check_avail:
            available = [
                resource_id
                for resource_id, state in self._resource_pool.items()
                if state["release_time"] is None
            ]
            if not available:
                return None

            resource_id = random.choice(available)
            self._resource_pool[resource_id]["release_time"] = release_time
            return resource_id

        resource_id = random.choice(list(self._resource_pool))
        self._execution += 1
        return resource_id

    def release_resource(self, resource_id):
        if self._check_avail:
            if resource_id not in self._resource_pool:
                raise KeyError("Unknown resource: {}".format(resource_id))
            self._resource_pool[resource_id]["release_time"] = None
        else:
            self._execution = max(0, self._execution - 1)

    def has_available_resource(self):
        if not self._check_avail:
            return True
        return any(
            state["release_time"] is None
            for state in self._resource_pool.values()
        )

    def get_occupancy(self):
        if self._check_avail:
            occupied = sum(
                state["release_time"] is not None
                for state in self._resource_pool.values()
            )
            return occupied / self._num_resources

        return min(1.0, self._execution / self._num_resources)

    def get_availability(self):
        """Return the number of currently free resources."""
        if not self._check_avail:
            return self._num_resources
        return sum(
            state["release_time"] is None
            for state in self._resource_pool.values()
        )

    def get_name(self):
        return self._name

    def get_resource_pool(self):
        return self._resource_pool

    def get_execution(self):
        return self._execution

    def get_next_release(self):
        release_times = [
            state["release_time"]
            for state in self._resource_pool.values()
            if state["release_time"] is not None
        ]
        return min(release_times) if release_times else None

    @staticmethod
    def _initialize_resources(size):
        return {
            "res_" + str(uuid.uuid4()): {"release_time": None}
            for _ in range(size)
        }


class ActivityCounter:
    """Number of active executions of one activity."""

    def __init__(self, name, index=0, initial=0):
        self._name = name
        self._index = index
        self._active_instances = int(initial)

    def add_act(self):
        self._active_instances += 1

    def remove_act(self):
        if self._active_instances <= 0:
            raise RuntimeError(
                "Activity counter for {} would become negative".format(
                    self._name
                )
            )
        self._active_instances -= 1

    def get_active_instances(self):
        return self._active_instances

    def get_name(self):
        return self._name


class ProcessInstance:
    """Per-case LSTM histories and lagged target values."""

    def __init__(self, case_id, n_size, n_features, n_act=False, dual=False):
        self._id = case_id
        if dual:
            self.init_dual_ngram(n_size, *n_features)
        else:
            self.init_ngram(n_size, n_features, n_act)
        self.proc_t = 0.0
        self.wait_t = 0.0

    def init_ngram(self, n_size, n_features, n_act):
        self._act_ngram = [0] * n_size
        self._feat_ngram = np.zeros(
            (1, n_size, int(n_features)),
            dtype=np.float32,
        )
        if n_act:
            self._n_act_ngram = [0] * n_size

    def init_dual_ngram(self, n_size, n_feat_proc, n_feat_wait):
        self._act_ngram = [0] * n_size
        self._n_act_ngram = [0] * n_size
        self._proc_feat_ngram = np.zeros(
            (1, n_size, int(n_feat_proc)),
            dtype=np.float32,
        )
        self._wait_feat_ngram = np.zeros(
            (1, n_size, int(n_feat_wait)),
            dtype=np.float32,
        )

    def get_ngram(self, n_act=False):
        if n_act:
            return (
                self._act_ngram,
                self._n_act_ngram,
                self._feat_ngram,
            )
        return self._act_ngram, self._feat_ngram

    def get_proc_ngram(self):
        return self._act_ngram, self._proc_feat_ngram

    def get_wait_ngram(self):
        return self._n_act_ngram, self._wait_feat_ngram

    @staticmethod
    def _get_time_features(timestamp):
        seconds_in_day = 24 * 60 * 60
        days_in_week = 7

        time_seconds = (
            timestamp.second
            + timestamp.minute * 60
            + timestamp.hour * 3600
        )
        weekday = timestamp.weekday()

        return [
            np.sin(2 * np.pi * time_seconds / seconds_in_day),
            np.cos(2 * np.pi * time_seconds / seconds_in_day),
            np.sin(2 * np.pi * weekday / days_in_week),
            np.cos(2 * np.pi * weekday / days_in_week),
        ]

    @staticmethod
    def _append_feature(history, record, label):
        record_array = np.asarray(record, dtype=np.float32)
        expected = history.shape[2]
        if record_array.size != expected:
            raise ValueError(
                "{} feature count mismatch: expected {}, got {}".format(
                    label,
                    expected,
                    record_array.size,
                )
            )
        return np.concatenate(
            [
                history[:, 1:, :],
                record_array.reshape(1, 1, expected),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def update_ngram(self, activity, timestamp, wip, role_occupancy, n_act=None):
        record = (
            [self.proc_t, self.wait_t]
            + list(wip)
            + self._get_time_features(timestamp)
            + list(role_occupancy)
        )
        self._feat_ngram = self._append_feature(
            self._feat_ngram,
            record,
            "single-model",
        )
        self._act_ngram = self._act_ngram[1:] + [int(activity)]
        if n_act is not None:
            self._n_act_ngram = self._n_act_ngram[1:] + [int(n_act)]

    def update_proc_ngram(self, activity, timestamp, wip, role_occupancy):
        record = (
            [self.proc_t]
            + list(wip)
            + self._get_time_features(timestamp)
            + list(role_occupancy)
        )
        self._proc_feat_ngram = self._append_feature(
            self._proc_feat_ngram,
            record,
            "processing-time",
        )
        self._act_ngram = self._act_ngram[1:] + [int(activity)]

    def update_wait_ngram(
        self,
        next_activity,
        timestamp,
        wip,
        role_occupancy,
    ):
        record = (
            [self.wait_t]
            + list(wip)
            + self._get_time_features(timestamp)
            + list(role_occupancy)
        )
        self._wait_feat_ngram = self._append_feature(
            self._wait_feat_ngram,
            record,
            "waiting-time",
        )
        self._n_act_ngram = self._n_act_ngram[1:] + [int(next_activity)]

    def update_proc_wait(self, proc_t, wait_t):
        self.proc_t = float(np.asarray(proc_t).reshape(-1)[0])
        self.wait_t = float(np.asarray(wait_t).reshape(-1)[0])

    def update_proc(self, proc_t):
        self.proc_t = float(np.asarray(proc_t).reshape(-1)[0])

    def update_wait(self, wait_t):
        self.wait_t = float(np.asarray(wait_t).reshape(-1)[0])
