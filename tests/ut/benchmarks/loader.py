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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "scripts"


def load_bench_script(name: str) -> ModuleType:
    """Import a benchmark script by path; they are CLIs, not installed packages."""
    # The scripts only use aiohttp for the HTTP path, which no test exercises.
    sys.modules.setdefault("aiohttp", MagicMock())
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register the
    # module before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    """One token per whitespace word, round-tripping content like a real one."""

    def encode(self, text, add_special_tokens=False):
        self._words = text.split()
        return list(range(len(self._words)))

    def decode(self, ids):
        return " ".join(self._words[index] for index in ids)


class CollapsingTokenizer:
    """Pathological tokenizer whose decode discards the input content."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids):
        return " ".join(f"t{index}" for index in ids)
