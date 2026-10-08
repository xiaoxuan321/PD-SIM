# -*- coding: utf-8 -*-

import os
import subprocess
import traceback

import readers.bpmn_reader as br
import external_tools.process_structure as gph
from support_modules.log_repairing import conformance_checking as chk


class StructureMiner(object):
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

    def __init__(self, settings, log):
        self.log = log
        self.is_safe = True
        self.settings = settings

    def execute_pipeline(self):
        self.is_safe = self._mining_structure(
            is_safe=self.is_safe
        )

        self.is_safe = self._evaluate_alignment(
            is_safe=self.is_safe
        )

    @Decorators.safe_exec
    def _mining_structure(self, **kwargs):
        miner = self._get_miner(
            self.settings["mining_alg"]
        )
        miner(self.settings)

    def _get_miner(self, miner):
        if miner == "im":
            return self._im_miner

        if miner == "sm1":
            return self._sm1_miner

        raise ValueError(miner)

    @staticmethod
    def _im_miner(settings):
        print(
            " -- Mining Process Structure "
            "with Inductive Miner (IM) --"
        )

        file_name = (
            settings["file"].split(".")[0]
        )

        input_route = os.path.join(
            settings["output"],
            file_name + ".xes",
        )

        output_bpmn = os.path.join(
            settings["output"],
            file_name + ".bpmn",
        )

        if not os.path.exists(input_route):
            raise FileNotFoundError(
                "IM input XES does not exist: "
                "{}".format(input_route)
            )

        noise_threshold = float(
            settings.get(
                "im_noise_threshold",
                0.2
            )
        )

        if not 0.0 <= noise_threshold <= 1.0:
            raise ValueError(
                "im_noise_threshold must be in "
                "[0, 1], got {}".format(
                    noise_threshold
                )
            )

        jar_path = settings.get(
            "im_path",
            os.path.join(
                "external_tools",
                "inductiveminer",
                "im-process-tree-bpmn.jar",
            ),
        )

        args = [
            "java",
            "-jar",
            jar_path,
            input_route,
            output_bpmn,
            str(noise_threshold),
        ]

        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                "Inductive Miner failed with "
                "return code {}".format(
                    result.returncode
                )
            )

        if not os.path.exists(output_bpmn):
            raise RuntimeError(
                "Inductive Miner finished but no "
                "BPMN was produced: {}".format(
                    output_bpmn
                )
            )

    @staticmethod
    def _sm1_miner(settings):
        print(
            " -- Mining Process Structure --"
        )

        file_name = (
            settings["file"].split(".")[0]
        )

        input_route = os.path.join(
            settings["output"],
            file_name + ".xes",
        )

        output_prefix = os.path.join(
            settings["output"],
            file_name,
        )

        args = [
            "java",
            "-jar",
            settings["sm1_path"],
            str(settings["epsilon"]),
            str(settings["eta"]),
            input_route,
            output_prefix,
        ]

        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                "SplitMiner failed with return "
                "code {}".format(
                    result.returncode
                )
            )

    @Decorators.safe_exec
    def _evaluate_alignment(self, **kwargs):
        bpmn_path = os.path.join(
            self.settings["output"],
            self.settings["file"].split(".")[0]
            + ".bpmn",
        )

        if not os.path.exists(bpmn_path):
            raise FileNotFoundError(
                "BPMN file does not exist: "
                "{}".format(bpmn_path)
            )

        self.bpmn = br.BpmnReader(
            bpmn_path
        )

        self.process_graph = (
            gph.create_process_structure(
                self.bpmn
            )
        )

        chk.evaluate_alignment(
            self.process_graph,
            self.log,
            self.settings,
        )
