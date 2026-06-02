import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto

from agent_r1.core_algos import compute_reinforce_plus_plus_baseline_outcome_advantage
from agent_r1.ray_agent_trainer import _apply_prior_step_mean_reward_scale
from agent_r1.env.base import Action
from agent_r1.env.envs.teacher_guidance import TeacherGuidanceEnv
from agent_r1.metric_utils import compute_cumulative_guidance_success


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        if "|" in text and all(part.isdigit() for part in text.split("|") if part):
            return [int(part) for part in text.split("|") if part]
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        if token_ids and all(isinstance(token_id, int) and 0 <= token_id <= 255 for token_id in token_ids):
            return "".join(chr(token_id) for token_id in token_ids)
        return "|".join(str(token_id) for token_id in token_ids)


def test_prior_step_mean_reward_scale_on_sentence_reward_before_rfpp():
    """Masked sentence reward; step>=1 scaled by whole-batch mean of earlier steps."""
    rewards = torch.zeros(10, 2, dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]] * 10)
    rewards[3, 0] = 1.0
    rewards[4, 0] = 1.0
    rewards[5, 0] = 0.0
    rewards[6, 0] = 0.0
    rewards[7, 0] = 1.0
    rewards[8, 0] = 0.0
    rewards[9, 0] = 1.0
    step_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    batch = TensorDict({"token_level_rewards": rewards, "response_mask": mask}, batch_size=[10])
    data = DataProto(
        batch=batch,
        non_tensor_batch={"step_indices": step_indices, "uid": np.array(["g"] * 10, dtype=object)},
    )
    _apply_prior_step_mean_reward_scale(data, {"prior_step_mean_reward_scale": True})
    out = (data.batch["token_level_rewards"] * data.batch["response_mask"]).sum(dim=-1)
    assert torch.allclose(out[5:8], torch.tensor([0.0, 0.0, 0.4]))
    assert torch.allclose(out[8:10], torch.tensor([0.0, 0.3]), atol=1e-6)


def test_prior_step_mean_reward_scale_pools_across_whole_batch_including_mixed_uids():
    """Step-1 factor is mean over all step-0 rows (not per-uid)."""
    rewards = torch.zeros(6, 2, dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0]] * 6)
    rewards[0, 0] = 0.0
    rewards[1, 0] = 1.0
    rewards[4, 0] = 1.0
    rewards[2, 0] = 1.0
    rewards[3, 0] = 1.0
    rewards[5, 0] = 1.0
    step_indices = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    uid = np.array(["a", "a", "b", "b", "a", "b"], dtype=object)
    batch = TensorDict({"token_level_rewards": rewards, "response_mask": mask}, batch_size=[6])
    data = DataProto(batch=batch, non_tensor_batch={"step_indices": step_indices, "uid": uid})
    _apply_prior_step_mean_reward_scale(data, {"prior_step_mean_reward_scale": True})
    out = (data.batch["token_level_rewards"] * data.batch["response_mask"]).sum(dim=-1)
    # step-0 mean = (0+1+1+1)/4 = 0.75 -> both step-1 rows
    assert torch.allclose(out[4:6], torch.tensor([0.75, 0.75]))


def test_rfpp_baseline_aggregates_over_full_trajectory():
    token_level_rewards = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Two rollouts for the same uid, each rollout has 3 steps.
    trajectory_uids = ["traj-a", "traj-a", "traj-a", "traj-b", "traj-b", "traj-b"]
    uid = ["sample-0"] * 6

    advantages, returns = compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        trajectory_uids=trajectory_uids,
    )

    # Each step inside the same trajectory should receive the same broadcasted score.
    assert torch.allclose(advantages[0], advantages[1])
    assert torch.allclose(advantages[1], advantages[2])
    assert torch.allclose(advantages[3], advantages[4])
    assert torch.allclose(advantages[4], advantages[5])
    assert torch.allclose(advantages, returns)


def test_cumulative_guidance_success_uses_leq_n_rounds_semantics():
    success_by_round = compute_cumulative_guidance_success(
        trajectory_uids=["traj-a", "traj-a", "traj-b", "traj-b"],
        step_indices=[0, 1, 0, 1],
        step_scores=[0.0, 1.0, 1.0, 0.0],
    )

    assert success_by_round[0].tolist() == [0.0, 1.0]
    assert success_by_round[1].tolist() == [1.0, 1.0]


@pytest.mark.anyio
async def test_teacher_guidance_env_keeps_binary_reward_and_guidance_in_observation():
    env = TeacherGuidanceEnv(teacher_endpoint="http://teacher.example/v1")
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    async def fake_guidance(student_answer: str) -> str:
        assert student_answer == "#### 3"
        return "Check the addition carefully."

    env._generate_guidance = fake_guidance

    next_obs, reward, done, _ = await env.step(Action(text="#### 3"))
    assert reward == 0.0
    assert done is False
    assert next_obs.messages[-1]["role"] == "user"
    assert next_obs.messages[-1]["content"] == "Check the addition carefully."

    final_obs, reward, done, _ = await env.step(Action(text="#### 2"))
    assert reward == 1.0
    assert done is True
    assert final_obs.messages[-1]["role"] == "assistant"


@pytest.mark.anyio
async def test_teacher_guidance_env_skips_guidance_on_last_allowed_step():
    env = TeacherGuidanceEnv(teacher_endpoint="http://teacher.example/v1", agent_max_steps=1)
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    async def fail_if_called(student_answer: str) -> str:
        raise AssertionError(f"Teacher should not be called on the last step: {student_answer}")

    env._generate_guidance = fail_if_called

    final_obs, reward, done, info = await env.step(Action(text="#### 3"))
    assert reward == 0.0
    assert done is True
    assert info["terminated_at_max_steps"] is True
    assert final_obs.messages[-1]["role"] == "assistant"
    assert final_obs.messages[-1]["content"] == "#### 3"


@pytest.mark.anyio
async def test_teacher_guidance_env_allows_single_step_without_teacher_endpoint():
    env = TeacherGuidanceEnv(agent_max_steps=1)
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    final_obs, reward, done, info = await env.step(Action(text="#### 3"))
    assert reward == 0.0
    assert done is True
    assert info["terminated_at_max_steps"] is True
    assert final_obs.messages[-1]["role"] == "assistant"
    assert final_obs.messages[-1]["content"] == "#### 3"


@pytest.mark.anyio
async def test_teacher_guidance_env_retry_prompt_ablation_skips_teacher_call():
    env = TeacherGuidanceEnv(guidance_mode="retry_prompt", retry_prompt="Incorrect. Try again.")
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    async def fail_if_called(student_answer: str) -> str:
        raise AssertionError(f"Teacher should not be called in retry_prompt mode: {student_answer}")

    env._generate_guidance = fail_if_called

    next_obs, reward, done, _ = await env.step(Action(text="#### 3"))
    assert reward == 0.0
    assert done is False
    assert next_obs.messages[-2]["role"] == "assistant"
    assert next_obs.messages[-2]["content"] == "#### 3"
    assert next_obs.messages[-1]["role"] == "user"
    assert next_obs.messages[-1]["content"] == "Incorrect. Try again."


@pytest.mark.anyio
async def test_teacher_guidance_env_truncates_all_wrong_answers_for_retry():
    env = TeacherGuidanceEnv(teacher_endpoint="http://teacher.example/v1", tokenizer=MockTokenizer())
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    captured = {}

    async def fake_guidance(student_answer: str) -> str:
        captured["student_answer"] = student_answer
        return "Try again."

    env._generate_guidance = fake_guidance

    token_ids = list(range(3000))
    expected_truncated = "|".join(str(token_id) for token_id in range(2048))
    next_obs, reward, done, _ = await env.step(Action(text="wrong answer", token_ids=token_ids))

    assert reward == 0.0
    assert done is False
    assert captured["student_answer"] == expected_truncated
    assert next_obs.messages[-2]["role"] == "assistant"
    assert next_obs.messages[-2]["content"] == expected_truncated
    assert next_obs.messages[-1]["content"] == "Try again."


@pytest.mark.anyio
async def test_teacher_guidance_env_truncates_wrong_answer_at_last_step():
    env = TeacherGuidanceEnv(
        teacher_endpoint="http://teacher.example/v1",
        agent_max_steps=1,
        tokenizer=MockTokenizer(),
    )
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    token_ids = list(range(3000))
    expected_truncated = "|".join(str(token_id) for token_id in range(2048))
    final_obs, reward, done, info = await env.step(Action(text="wrong answer", token_ids=token_ids))

    assert reward == 0.0
    assert done is True
    assert info["terminated_at_max_steps"] is True
    assert final_obs.messages[-1]["role"] == "assistant"
    assert final_obs.messages[-1]["content"] == expected_truncated


@pytest.mark.anyio
async def test_teacher_guidance_env_truncates_teacher_guidance_by_tokens():
    env = TeacherGuidanceEnv(teacher_endpoint="http://teacher.example/v1", tokenizer=MockTokenizer())
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    long_guidance = "|".join(str(token_id) for token_id in range(3000))
    expected_truncated = "|".join(str(token_id) for token_id in range(2048))

    async def fake_guidance(student_answer: str) -> str:
        return long_guidance

    env._generate_guidance = fake_guidance

    next_obs, reward, done, _ = await env.step(Action(text="#### 3"))

    assert reward == 0.0
    assert done is False
    assert next_obs.messages[-1]["role"] == "user"
    assert next_obs.messages[-1]["content"] == expected_truncated
