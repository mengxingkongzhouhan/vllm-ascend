# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostics for the blocking steps of vLLM-Ascend worker startup.

Binding an NPU and building the collective communicators both block inside
CANN/HCCL and emit nothing while they wait. When one of them stalls, the process
looks frozen right after the last configuration log line, and when one of them
fails the traceback only carries a raw CANN error code. Neither outcome says
what an operator should do next.

This module supplies the missing context:

* :func:`track_startup_stage` announces the stage it wraps, records how long it
  took, and while the stage is still running periodically reports the elapsed
  time along with the stacks of every thread, so a stall can be attributed to a
  concrete call.
* :func:`describe_device_start_failure` recognises the CANN errors that mean
  "the device could not be started" and turns them into the checks that
  actually resolve them.
"""

import faulthandler
import glob
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

# How long a startup stage may stay silent before the watchdog starts
# reporting. Startup stages run once per process and are coarse-grained, so tens
# of seconds is short enough to be useful while staying quiet on healthy but
# slow machines.
STARTUP_STAGE_REPORT_INTERVAL_SECONDS = 60.0

DEFAULT_ASCEND_HOME_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
ASCEND_DRIVER_VERSION_FILE = "/usr/local/Ascend/driver/version.info"
ASCEND_TOOLKIT_INSTALL_INFO_GLOB = "*-linux/ascend_toolkit_install.info"

DEVICE_BIND_HINT = (
    "A device bind that never returns usually means the NPU is still held by a "
    "process from a previous run, or that the driver is not responding. Inspect "
    "the card with `npu-smi info`."
)

DISTRIBUTED_INIT_HINT = (
    "Every rank must join the rendezvous before it completes, so a rank that "
    "died or was never launched, an unreachable master address/port, or a "
    "HCCL_IF_IP pointing at the wrong NIC all block here indefinitely. Check "
    "that every rank reported this stage, and set HCCL_CONNECT_TIMEOUT to fail "
    "fast instead of waiting."
)

# CANN markers that all mean "the runtime could not start the device". They are
# reported through several layers, so match on any of them.
_DEVICE_START_FAILURE_MARKERS = (
    # rtSetDevice failure returned by c10_npu::SetDevice.
    "507033",
    # Failed to load a device-side package, e.g. cann-hybm-compat.tar.gz.
    "E39011",
    "Failed to start the device",
    "TsdOpen failed",
)


def _format_context(context: dict[str, Any]) -> str:
    if not context:
        return "no additional context"
    return ", ".join(f"{key}={value}" for key, value in context.items())


def _dump_all_thread_stacks() -> None:
    """Write the stacks of all threads of this process to stderr."""
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        # Stack dumping is best effort; it must never mask the tracked stage.
        logger.debug("Failed to dump thread stacks.", exc_info=True)


def _watch_startup_stage(
    stage: str,
    hint: str | None,
    context_text: str,
    started_at: float,
    finished: threading.Event,
    report_interval: float,
) -> None:
    hint_text = f"{hint} " if hint else ""
    while not finished.wait(report_interval):
        logger.warning(
            "Startup stage '%s' is still running after %.0f s (%s). The process is "
            "blocked here rather than crashed. %sThread stacks follow.",
            stage,
            time.monotonic() - started_at,
            context_text,
            hint_text,
        )
        _dump_all_thread_stacks()


@contextmanager
def track_startup_stage(
    stage: str,
    hint: str | None = None,
    report_interval: float = STARTUP_STAGE_REPORT_INTERVAL_SECONDS,
    **context: Any,
) -> Iterator[None]:
    """Log the boundaries of a startup stage and report it while it stalls.

    Args:
        stage: Short identifier of the stage, used in every log line.
        hint: Guidance appended to the watchdog warning, describing what
            commonly makes this particular stage stall.
        report_interval: Seconds between watchdog reports. A value <= 0 disables
            the watchdog and keeps only the enter/exit logs.
        context: Extra key/value pairs (rank, device, ...) added to the logs so a
            line can be traced back to the process that emitted it.
    """
    context_text = _format_context(context)
    logger.info("Startup stage '%s' started (%s).", stage, context_text)

    started_at = time.monotonic()
    finished = threading.Event()
    watchdog: threading.Thread | None = None
    if report_interval > 0:
        watchdog = threading.Thread(
            target=_watch_startup_stage,
            args=(stage, hint, context_text, started_at, finished, report_interval),
            name=f"ascend-startup-watchdog-{stage}",
            daemon=True,
        )
        watchdog.start()

    try:
        yield
    finally:
        finished.set()
        if watchdog is not None:
            watchdog.join(timeout=report_interval)
        logger.info("Startup stage '%s' took %.2f s.", stage, time.monotonic() - started_at)


def _read_version(path: str) -> str | None:
    """Return the ``version`` entry of an Ascend install-info file.

    Both ``driver/version.info`` and ``ascend_toolkit_install.info`` are flat
    ``key=value`` files, but they order their keys differently, so look the key
    up rather than trusting a fixed line.
    """
    try:
        with open(path) as file:
            lines = [line.strip() for line in file if line.strip()]
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() == "version" and value.strip():
            return value.strip()
    return lines[0] if lines else None


def ascend_version_summary() -> str:
    """Summarise the installed driver and CANN toolkit versions.

    Used to make version mismatches visible in error messages, so the operator
    does not have to go looking for the two files that matter.
    """
    parts = []

    driver_version = _read_version(ASCEND_DRIVER_VERSION_FILE)
    parts.append(f"driver={driver_version or 'unknown'}")

    ascend_home_path = envs.ASCEND_HOME_PATH or DEFAULT_ASCEND_HOME_PATH
    toolkit_version = None
    for install_info in sorted(glob.glob(os.path.join(ascend_home_path, ASCEND_TOOLKIT_INSTALL_INFO_GLOB))):
        toolkit_version = _read_version(install_info)
        if toolkit_version is not None:
            break
    parts.append(f"toolkit={toolkit_version or 'unknown'} (from {ascend_home_path})")

    return ", ".join(parts)


def describe_device_start_failure(error: BaseException) -> str | None:
    """Explain a CANN device-start failure, or return ``None`` if unrecognised.

    ``torch.npu.set_device`` surfaces such failures as a bare error code plus a
    CANN traceback that points at "contact technical support". In practice the
    cause is almost always one of a small number of environment problems, so
    spell those out instead.
    """
    error_text = str(error)
    if not any(marker in error_text for marker in _DEVICE_START_FAILURE_MARKERS):
        return None

    return (
        "The CANN runtime failed to start the NPU. This is an environment-level "
        "failure: the model, vLLM and vllm-ascend configuration are not "
        "involved. Check, in this order:\n"
        "1. Driver/firmware and CANN toolkit compatibility. A device-side "
        "package such as cann-hybm-compat.tar.gz fails to load when the "
        f"toolkit is newer than the driver it runs against. Detected {ascend_version_summary()}; "
        "compare them against the compatibility matrix of your CANN release and "
        "upgrade the driver/firmware or the toolkit so they match.\n"
        "2. Device availability. A card still held by a process from a previous "
        "run, or one in an unhealthy state, cannot be started again. Run "
        "`npu-smi info` to inspect it, terminate leftover processes, and reset "
        "the card with `npu-smi set -t reset -i <card_id> -c <chip_id>`.\n"
        "3. Container setup. The host driver directory and the /dev/davinci* "
        "device nodes must all be mounted into the container; a partial mount "
        "lets the process start but makes device start fail.\n"
        "Host logs under ~/ascend/log and device logs under /var/log/npu carry "
        "the underlying driver error."
    )
