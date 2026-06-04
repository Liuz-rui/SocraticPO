# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re

# Strict *block* (not "entire model output must be only this"):
#   <answer>
#   A
#   </answer>
# No spaces inside the block; letter must be uppercase A/B/C/D; internal newlines are '\n'.
# Full outputs may include <reasoning>...</reasoning> before this block.
_STRICT_ANSWER_BLOCK_RE = re.compile(r"<answer>\n([ABCD])\n</answer>")


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _extract_last_strict_answer_letter(text: str | None) -> str | None:
    """Return letter from the last exact <answer>\\nX\\n</answer> block in text."""
    if not isinstance(text, str) or not text:
        return None
    text = _normalize_newlines(text)
    matches = list(_STRICT_ANSWER_BLOCK_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _normalize_ground_truth_letter(gt: str | None) -> str | None:
    if not isinstance(gt, str) or not gt.strip():
        return None
    gt = gt.strip()
    if len(gt) == 1 and gt.upper() in {"A", "B", "C", "D"}:
        return gt.upper()
    return _extract_last_strict_answer_letter(gt)


def grade_answer(predicted: str | None, ground_truth: str | None) -> bool:
    """Grade MC: last strict <answer> block vs ground truth letter or strict block."""
    pred_letter = _extract_last_strict_answer_letter(predicted)
    if pred_letter is None:
        return False

    gt_letter = _normalize_ground_truth_letter(ground_truth)
    if gt_letter is None:
        return False
    return pred_letter == gt_letter


def compute_score(model_output: str, ground_truth: str) -> bool:
    try:
        return grade_answer(model_output, ground_truth)
    except Exception:
        return False
