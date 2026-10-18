# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a call an approved move turns from the reference's refusal into
an answer.

Where an approved entry moves a runner from the reference's refusal at a
call to a success (the owner's KO2-61 on time.feature-time-of-day@sqlite),
the reference's statements up to its refusal are no expectation of that
call's: its ``sql.count`` and ``sql.result`` differences against the
reference are reported as diagnostic. Every other call, and the same call
under a move that is not to an answer, gates as before.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY = 'time.feature-time-of-day@sqlite'


def _compare() -> ModuleType:
    name = 't21e_compare'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/compare.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _reference() -> dict[str, Any]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    for text in path.read_text(encoding='utf-8').splitlines():
        line = json.loads(text)
        if line['case'] == KEY:
            return line
    raise AssertionError(f'{KEY}: no reference line')


def _failures(answered: bool) -> tuple[list[str], list[str]]:
    r"""The sql rules' failures and notes for a runner whose last
    materialize statement and first from_source statement read other rows
    than the reference's, judged past the reference's materialize refusal."""
    c = _compare()
    reference = _reference()
    observation = copy.deepcopy(reference)
    statements = observation['driver']['capture']['statements']
    for statement in (statements[0], statements[-1]):
        values = statement['results'][0]['values']
        values['unordered'] = values['ordered'] = '0' * 64
    report = c.Report()
    c._sql(
        report,
        {'id': KEY},
        'python-core',
        observation,
        reference,
        {},
        stop='materialize',
        database='sqlite',
        answered=answered,
    )
    return [text for _, text in report.failures], report.notes


def test_an_approved_answer_reports_its_call_s_statements_and_no_other() -> (
    None
):
    failures, notes = _failures(answered=True)
    assert failures
    assert all(': from_source statement' in text for text in failures)
    assert any(
        ': materialize statement' in text and 'sql.result' in text
        for text in notes
    )
    held, _ = _failures(answered=False)
    assert any(': materialize statement' in text for text in held)
    assert any(': from_source statement' in text for text in held)
