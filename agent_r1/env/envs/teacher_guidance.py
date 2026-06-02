import logging
import os
from typing import Any

import aiohttp

from verl.utils.reward_score import default_compute_score

from ..base import Action, AgentEnv, Observation

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are an expert teacher. Review the student's latest answer and provide concise, actionable guidance. "
    "Prefer hints, corrections, and next-step suggestions over directly revealing the answer."
)
DEFAULT_RETRY_PROMPT = "Your answer is incorrect. Please try again."

DEFAULT_TEACHER_USER_TEMPLATE = """Question:
{question}

Student's latest answer:
{student_answer}

Reference solution:
{reference_solution}

{ground_truth_block}

Write a short piece of guidance for the student.
- Point out the main issue in the student's answer.
- Give the next step or correction they should try.
- Do not solve the full problem for them unless explicitly allowed.
- Keep the guidance short and concrete.
"""

MAX_INCORRECT_ANSWER_TOKENS = 1024
MAX_TEACHER_GUIDANCE_TOKENS = 2048


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content) if content is not None else ""

async def _generate_with_endpoint(
    endpoint: str,
    model_name: str | None,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name or "teacher-model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Teacher endpoint returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


@AgentEnv.register("teacher_guidance")
class TeacherGuidanceEnv(AgentEnv):
    def __init__(
        self,
        teacher_endpoint: str | None = None,
        teacher_model_name: str | None = None,
        teacher_api_key: str | None = None,
        teacher_system_prompt: str = DEFAULT_TEACHER_SYSTEM_PROMPT,
        teacher_user_template: str = DEFAULT_TEACHER_USER_TEMPLATE,
        teacher_temperature: float = 0.2,
        teacher_max_tokens: int = 256,
        guidance_mode: str = "teacher",
        retry_prompt: str = DEFAULT_RETRY_PROMPT,
        reveal_answer: bool = False,
        agent_max_steps: int | None = None,
        **kwargs,
    ):
        if guidance_mode not in {"teacher", "retry"}:
            raise ValueError("guidance_mode must be either 'teacher' or 'retry'")
        needs_teacher_endpoint = guidance_mode == "teacher" and (agent_max_steps is None or agent_max_steps > 1)
        if needs_teacher_endpoint and not teacher_endpoint:
            raise ValueError("TeacherGuidanceEnv requires teacher_endpoint")
        self.teacher_endpoint = teacher_endpoint
        self.teacher_model_name = teacher_model_name
        self.teacher_api_key = teacher_api_key
        self.teacher_system_prompt = teacher_system_prompt
        self.teacher_user_template = teacher_user_template
        self.teacher_temperature = teacher_temperature
        self.teacher_max_tokens = teacher_max_tokens
        self.guidance_mode = guidance_mode
        self.retry_prompt = retry_prompt
        self.reveal_answer = reveal_answer
        self.agent_max_steps = agent_max_steps
        self.tokenizer = kwargs.get("tokenizer")
        self.tokenizer_skip_special_tokens = kwargs.get("tokenizer_skip_special_tokens", True)

        self._messages: list[dict[str, Any]] = []
        self._data_source: str | None = None
        self._ground_truth: str | None = None
        self._question: str = ""
        self._reference_solution: str = ""
        self._step_idx: int = 0

    def reset(self, **kwargs) -> Observation:
        self._messages = list(kwargs.get("raw_prompt", []))
        reward_model = kwargs.get("reward_model") or {}
        extra_info = kwargs.get("extra_info") or {}

        self._data_source = kwargs.get("data_source")
        self._ground_truth = reward_model.get("ground_truth") or extra_info.get("ground_truth")
        self._reference_solution = extra_info.get("answer")

        question = extra_info.get("question")
        if not question:
            for message in reversed(self._messages):
                if message.get("role") == "user":
                    question = _extract_text_content(message.get("content"))
                    break
        self._question = question or ""
        self._step_idx = 0

        return Observation(messages=list(self._messages))

    def _score_answer(self, student_answer: str) -> float:
        if self._data_source is None or self._ground_truth is None:
            raise ValueError("TeacherGuidanceEnv requires both data_source and reward_model.ground_truth")

        score = default_compute_score(
            data_source=self._data_source,
            solution_str=student_answer,
            ground_truth=self._ground_truth,
            extra_info={"reference_solution": self._reference_solution},
        )

        if isinstance(score, dict):
            score = score.get("score", 0.0)
        return 1.0 if float(score) >= 1.0 else 0.0

    def _truncate_incorrect_answer(self, action: Action) -> str:
        if action.token_ids is None or self.tokenizer is None:
            return action.text or ""

        truncated_token_ids = action.token_ids[:MAX_INCORRECT_ANSWER_TOKENS]
        if len(truncated_token_ids) == len(action.token_ids):
            return action.text or ""

        return self.tokenizer.decode(truncated_token_ids, skip_special_tokens=self.tokenizer_skip_special_tokens)

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        if self.tokenizer is None:
            return text

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        truncated_token_ids = token_ids[:max_tokens]
        if len(truncated_token_ids) == len(token_ids):
            return text

        return self.tokenizer.decode(truncated_token_ids, skip_special_tokens=self.tokenizer_skip_special_tokens)

    def _build_teacher_messages(self, student_answer: str) -> list[dict[str, str]]:
        ground_truth_block = (
            f"Ground-truth final answer:\n{self._ground_truth}\n"
            if self.reveal_answer and self._ground_truth is not None
            else "Do not directly reveal the final answer.\n"
        )
        user_message = self.teacher_user_template.format(
            question=self._question,
            student_answer=student_answer,
            reference_solution=self._reference_solution or "Not provided.",
            ground_truth_block=ground_truth_block,
        )
        return [
            {"role": "system", "content": self.teacher_system_prompt},
            {"role": "user", "content": user_message},
        ]

    async def _generate_guidance(self, student_answer: str) -> str:
        if self.teacher_endpoint is None:
            raise ValueError("TeacherGuidanceEnv requires teacher_endpoint for teacher guidance")
        messages = self._build_teacher_messages(student_answer)
        return await _generate_with_endpoint(
            endpoint=self.teacher_endpoint,
            model_name=self.teacher_model_name,
            api_key=self.teacher_api_key,
            messages=messages,
            max_tokens=self.teacher_max_tokens,
            temperature=self.teacher_temperature,
        )

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if not isinstance(action, Action) or action.text is None:
            raise TypeError("TeacherGuidanceEnv only accepts Action with text")

        student_answer = action.text
        self._step_idx += 1

        reward = self._score_answer(student_answer)
        if reward >= 1.0:
            self._messages.append({"role": "assistant", "content": student_answer})
            return Observation(messages=list(self._messages)), 1.0, True, {}

        student_answer = self._truncate_incorrect_answer(action)
        self._messages.append({"role": "assistant", "content": student_answer})

        if self.agent_max_steps is not None and self._step_idx >= self.agent_max_steps:
            return Observation(messages=list(self._messages)), 0.0, True, {"terminated_at_max_steps": True}

        if self.guidance_mode == "retry":
            guidance = self.retry_prompt
        else:
            guidance = await self._generate_guidance(student_answer)
        guidance = self._truncate_text_by_tokens(guidance, MAX_TEACHER_GUIDANCE_TOKENS)
        self._messages.append({"role": "user", "content": guidance})
        return Observation(messages=list(self._messages)), 0.0, False, {}
