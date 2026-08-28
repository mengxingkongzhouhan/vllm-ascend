#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import os
import tempfile
import threading
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.startup_diagnostics import (
    ascend_version_summary,
    describe_device_start_failure,
    track_startup_stage,
)

# Abridged form of the failure reported by torch_npu when the CANN runtime
# cannot start the device, as seen through `torch.npu.set_device`.
DEVICE_START_ERROR_TEXT = (
    "ExchangeDevice:../torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp:285 "
    "NPU function error: c10_npu::SetDevice(device), error code is 507033\n"
    "[Error]: Failed to start the device.\n"
    "Inner_Error_Failed_Load_Package_To_Device(E39011): Failed to load the "
    "package cann-hybm-compat.tar.gz on the device.\n"
    "TsdOpen failed. devId=2, tdt error=1."
)


class TestTrackStartupStage(TestBase):
    def test_stage_body_runs_and_watchdog_stays_quiet(self):
        executed = []
        with (
            patch("vllm_ascend.startup_diagnostics.logger") as mock_logger,
            track_startup_stage("bind_npu_device", rank=3, device="npu:3"),
        ):
            executed.append("body")

        self.assertEqual(executed, ["body"])
        mock_logger.warning.assert_not_called()
        # One log when the stage starts, one reporting how long it took.
        self.assertEqual(mock_logger.info.call_count, 2)
        start_message = mock_logger.info.call_args_list[0][0]
        self.assertIn("rank=3", start_message[2])
        self.assertIn("device=npu:3", start_message[2])

    def test_exceptions_propagate_and_stage_duration_is_logged(self):
        with (
            patch("vllm_ascend.startup_diagnostics.logger") as mock_logger,
            self.assertRaises(ValueError),
            track_startup_stage("bind_npu_device"),
        ):
            raise ValueError("boom")

        self.assertEqual(mock_logger.info.call_count, 2)
        self.assertIn("took", mock_logger.info.call_args_list[1][0][0])

    def test_stalled_stage_is_reported_with_hint_and_stacks(self):
        reported = threading.Event()
        with (
            patch("vllm_ascend.startup_diagnostics.logger") as mock_logger,
            patch("vllm_ascend.startup_diagnostics._dump_all_thread_stacks") as mock_dump,
        ):
            mock_logger.warning.side_effect = lambda *args, **kwargs: reported.set()
            with track_startup_stage(
                "init_distributed_environment",
                hint="check every rank",
                report_interval=0.01,
                rank=0,
            ):
                self.assertTrue(reported.wait(timeout=10))

        mock_logger.warning.assert_called()
        mock_dump.assert_called()
        warning_args = mock_logger.warning.call_args[0]
        self.assertIn("init_distributed_environment", warning_args)
        self.assertIn("check every rank ", warning_args)

    def test_non_positive_report_interval_disables_watchdog(self):
        watchdog_threads_before = self._watchdog_thread_names()
        with (
            patch("vllm_ascend.startup_diagnostics.logger"),
            track_startup_stage("bind_npu_device", report_interval=0),
        ):
            self.assertEqual(self._watchdog_thread_names(), watchdog_threads_before)

    @staticmethod
    def _watchdog_thread_names() -> set[str]:
        return {thread.name for thread in threading.enumerate() if thread.name.startswith("ascend-startup-watchdog")}


class TestDescribeDeviceStartFailure(TestBase):
    def test_device_start_failure_is_explained(self):
        explanation = describe_device_start_failure(RuntimeError(DEVICE_START_ERROR_TEXT))

        assert explanation is not None
        self.assertIn("failed to start the NPU", explanation)
        self.assertIn("npu-smi info", explanation)
        self.assertIn("driver=", explanation)

    def test_each_marker_is_recognised_on_its_own(self):
        for marker in ("507033", "E39011", "Failed to start the device", "TsdOpen failed"):
            with self.subTest(marker=marker):
                self.assertIsNotNone(describe_device_start_failure(RuntimeError(marker)))

    def test_unrelated_errors_are_not_explained(self):
        self.assertIsNone(describe_device_start_failure(RuntimeError("NPU out of memory")))


class TestAscendVersionSummary(TestBase):
    def test_versions_are_read_from_install_metadata(self):
        with (
            patch("vllm_ascend.startup_diagnostics._read_version", side_effect=["24.1.rc2", "8.0.RC2"]),
            patch(
                "vllm_ascend.startup_diagnostics.glob.glob",
                return_value=[os.path.join("/opt/cann", "aarch64-linux", "ascend_toolkit_install.info")],
            ),
        ):
            summary = ascend_version_summary()

        self.assertIn("driver=24.1.rc2", summary)
        self.assertIn("toolkit=8.0.RC2", summary)

    def test_missing_metadata_reports_unknown(self):
        with (
            patch("vllm_ascend.startup_diagnostics._read_version", return_value=None),
            patch("vllm_ascend.startup_diagnostics.glob.glob", return_value=[]),
        ):
            summary = ascend_version_summary()

        self.assertIn("driver=unknown", summary)
        self.assertIn("toolkit=unknown", summary)


class TestReadVersion(TestBase):
    def test_version_key_is_looked_up_regardless_of_position(self):
        from vllm_ascend.startup_diagnostics import _read_version

        with tempfile.TemporaryDirectory() as directory:
            install_info = os.path.join(directory, "ascend_toolkit_install.info")
            with open(install_info, "w") as file:
                file.write("package_name=Ascend-cann-toolkit\narch=aarch64\nversion=8.0.RC2\n")

            self.assertEqual(_read_version(install_info), "8.0.RC2")

    def test_missing_file_returns_none(self):
        from vllm_ascend.startup_diagnostics import _read_version

        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_read_version(os.path.join(directory, "absent.info")))
