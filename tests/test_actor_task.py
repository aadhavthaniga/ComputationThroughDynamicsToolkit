"""Checks for the ActorTask environment and ActorLoss.

Run with `pytest tests/test_actor_task.py`.
"""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import ctd
from ctd.task_modeling.callbacks.callbacks_actor import ActorLossTermsLogger
from ctd.task_modeling.model.rnn import GRU_RNN
from ctd.task_modeling.task_env.loss_func import ActorLoss
from ctd.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper

N_TIME = 300
VALID_ON = 40
ARRIVAL = 100


@pytest.fixture(scope="module")
def env():
    cfg = OmegaConf.load(
        Path(ctd.__file__).parent / "task_modeling/configs/env_task/ActorTask.yaml"
    )
    env = hydra.utils.instantiate(cfg, _convert_="all")
    env.set_seed(0)
    return env


def make_trial(env, valid_on, at_goal_steps):
    """Fingertip sits at the center, except on at_goal_steps where it is at the goal"""
    goal = env.center + env.reach_dist * torch.tensor([1.0, 0.0])
    target = env.center.repeat(N_TIME, 1)
    if valid_on >= 0:
        target[valid_on:] = goal
    fingertip = env.center.repeat(N_TIME, 1)
    fingertip[at_goal_steps] = goal
    return fingertip, target


def trigger_label(env, valid_on, at_goal_steps):
    fingertip, target = make_trial(env, valid_on, at_goal_steps)
    label = env.loss_func.trigger_label(
        fingertip[None], target[None], torch.tensor([float(valid_on)])
    )
    return label[0]


def test_loss_parameters_come_from_env(env):
    loss = env.loss_func
    assert isinstance(loss, ActorLoss)
    assert (loss.hold_steps, loss.trigger_steps) == (10, 4)
    assert loss.reach_radius < env.reach_dist


def test_pulse_starts_at_arrival_plus_hold(env):
    # At the goal before valid_on (10-30) and for a 5-step touch (60-64): neither
    # counts. Arrives for good at ARRIVAL.
    steps = np.r_[10:31, 60:65, ARRIVAL:N_TIME]
    label = trigger_label(env, VALID_ON, steps)
    start = ARRIVAL + env.loss_func.hold_steps
    expected = list(range(start, start + env.loss_func.trigger_steps))
    assert torch.nonzero(label).flatten().tolist() == expected
    assert not label.requires_grad


def test_no_pulse_if_hand_never_arrives(env):
    assert trigger_label(env, VALID_ON, []).sum() == 0


def test_no_pulse_on_catch_trial(env):
    # The hand sits exactly on its target (the center) for the whole trial
    assert trigger_label(env, -1, []).sum() == 0


def test_trigger_does_not_move_the_arm(env):
    data, _ = env.generate_dataset(4)
    fingertips = []
    for trigger in (0.0, 1.0):
        env.reset(batch_size=4, options={"ic_state": data["ics"]})
        action = torch.full((4, env.action_space.shape[0]), 0.2)
        action[:, -1] = trigger
        for _ in range(20):
            obs, _, _, _, info = env.step(action=action, deterministic=True)
        fingertips.append(info["states"]["fingertip"])
    assert obs.shape == (4, env.observation_space.shape[0])
    assert torch.equal(fingertips[0], fingertips[1])


def test_loss_is_finite_for_untrained_gru_rollout(env):
    data, _ = env.generate_dataset(8)
    model = GRU_RNN(latent_size=128)
    model.init_model(
        env.observation_space.shape[0] + env.context_inputs.shape[0],
        env.action_space.shape[0],
    )
    wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay=0)
    wrapper.set_environment(env)
    wrapper.set_model(model)
    out = wrapper.forward(
        data["ics"],
        torch.tensor(data["inputs"], dtype=torch.float32),
        torch.tensor(data["inputs_to_env"], dtype=torch.float32),
    )
    assert out["controlled"].shape == (8, env.n_timesteps, 2)
    assert out["actions"].shape == (8, env.n_timesteps, 7)

    loss_dict = {
        "controlled": out["controlled"],
        "actions": out["actions"],
        "targets": data["targets"],
        "extra": torch.tensor(data["extra"], dtype=torch.float32),
    }
    # Epoch 0 has an empty position window; it must not give NaN
    for epoch in (0, 1, 100, 200):
        assert torch.isfinite(wrapper.loss_func({**loss_dict, "epoch": epoch}))

    wrapper.loss_func({**loss_dict, "epoch": 0}).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.fixture(scope="module")
def trigger_losses(env):
    """Trigger loss of four Trigger outputs against a pulse label at bins 110-113"""
    label = torch.zeros(1, N_TIME)
    label[0, 110:114] = 1.0
    shifted = torch.zeros(1, N_TIME)
    shifted[0, 112:116] = 1.0
    outputs = {
        "correct": label.clone(),
        "shifted": shifted,
        "never": torch.zeros(1, N_TIME),
        "always": torch.ones(1, N_TIME),
    }
    return {k: env.loss_func.trigger_loss(v, label).item() for k, v in outputs.items()}


def test_correct_pulse_beats_never_firing(trigger_losses):
    assert trigger_losses["correct"] < trigger_losses["never"]


def test_pulse_shifted_by_two_bins_beats_never_firing(trigger_losses):
    assert trigger_losses["shifted"] < trigger_losses["never"]


def test_firing_on_every_bin_is_worse_than_correct_pulse(trigger_losses):
    assert trigger_losses["always"] > trigger_losses["correct"]


def test_default_pos_weight_balances_pulse_and_other_bins(env, trigger_losses):
    # Missing the 4 pulse bins costs as much as firing on the other 296
    assert env.loss_func.trigger_pos_weight == 74.0
    assert trigger_losses["never"] == pytest.approx(trigger_losses["always"])


def test_pos_weight_of_one_is_plain_mse():
    loss = ActorLoss(torch.nn.MSELoss(), 1.0, 1.0, trigger_pos_weight=1.0)
    trigger, label = torch.rand(3, N_TIME), torch.zeros(3, N_TIME)
    label[:, 110:114] = 1.0
    expected = torch.nn.MSELoss()(trigger, label)
    assert loss.trigger_loss(trigger, label).item() == pytest.approx(expected.item())


def test_loss_terms_logger_logs_the_three_parts(env):
    class FakeModule:
        def __init__(self, loss_func):
            self.loss_func = loss_func
            self.logged = {}

        def log(self, name, value):
            self.logged[name] = value

    fingertip, target = make_trial(env, VALID_ON, np.r_[ARRIVAL:N_TIME])
    total = env.loss_func(
        {
            "controlled": fingertip[None],
            "targets": target[None],
            "actions": torch.full((1, N_TIME, 7), 0.1),
            "extra": torch.tensor([[float(VALID_ON), float(N_TIME)]]),
            "epoch": 200,
        }
    )
    module = FakeModule(env.loss_func)
    ActorLossTermsLogger().on_train_batch_end(None, module, None, None, 0)
    names = ["train/position_loss", "train/effort_loss", "train/trigger_loss"]
    assert sorted(module.logged) == sorted(names)
    assert sum(module.logged.values()).item() == pytest.approx(total.item())


def old_pulse_label(loss, fingertip, target, valid_on):
    """ActorLoss.trigger_label as it was before trigger_shape existed"""
    n_trials, n_time, _ = fingertip.shape
    step = torch.arange(n_time)
    at_goal = torch.linalg.norm(fingertip - target, dim=-1) < loss.reach_radius
    at_goal &= (valid_on[:, None] >= 0) & (step[None, :] >= valid_on[:, None])
    held = at_goal.int().unfold(1, loss.hold_steps, 1).min(dim=-1).values > 0
    label = torch.zeros(n_trials, n_time)
    for trial in torch.nonzero(held.any(dim=1)).flatten():
        start = held[trial].int().argmax() + loss.hold_steps
        label[trial, start : start + loss.trigger_steps] = 1.0
    return label


def test_pulse_mode_gives_the_old_labels_and_loss(env):
    # Arrivals early, late, cut off by the trial end, never, and a catch trial
    cases = [
        (VALID_ON, np.r_[ARRIVAL:N_TIME]),
        (VALID_ON, np.r_[200:N_TIME]),
        (VALID_ON, np.r_[N_TIME - 12 : N_TIME]),
        (VALID_ON, []),
        (-1, []),
    ]
    trials = [make_trial(env, v, s) for v, s in cases]
    fingertip = torch.stack([t[0] for t in trials])
    target = torch.stack([t[1] for t in trials])
    valid_on = torch.tensor([float(v) for v, _ in cases])
    loss = env.loss_func
    assert loss.trigger_shape == "pulse"
    label = loss.trigger_label(fingertip, target, valid_on)
    old = old_pulse_label(loss, fingertip, target, valid_on)
    assert torch.equal(label, old)
    trigger = torch.rand(len(cases), N_TIME)
    old_loss = ((1.0 + 73.0 * old) * (trigger - old) ** 2).mean()
    assert loss.trigger_loss(trigger, label).item() == old_loss.item()


def step_decay_loss(**kwargs):
    return ActorLoss(
        torch.nn.MSELoss(),
        1.0,
        1.0,
        trigger_shape="step_decay",
        trigger_ms=300,
        decay_ms=100,
        **kwargs
    )


def test_step_decay_label_shape_and_length(env):
    loss = step_decay_loss()
    fingertip, target = make_trial(env, VALID_ON, np.r_[ARRIVAL:N_TIME])
    label = loss.trigger_label(fingertip[None], target[None], torch.tensor([40.0]))[0]
    start = ARRIVAL + loss.hold_steps
    assert torch.all(label[:start] == 0)
    assert torch.all(label[start : start + 30] == 1)
    # 0.9, 0.8, ..., 0.1, then 0 from the 10th decay bin on
    expected = torch.linspace(1.0, 0.0, 11)[1:]
    assert torch.allclose(label[start + 30 : start + 40], expected)
    assert torch.all(label[start + 39 :] == 0)


def test_step_decay_label_is_cut_at_trial_end(env):
    loss = step_decay_loss()
    arrival = N_TIME - 30
    fingertip, target = make_trial(env, VALID_ON, np.r_[arrival:N_TIME])
    label = loss.trigger_label(fingertip[None], target[None], torch.tensor([40.0]))[0]
    assert label.shape == (N_TIME,)
    assert torch.all(label[arrival + loss.hold_steps :] == 1)


@pytest.mark.parametrize("valid_on", [VALID_ON, -1])
def test_step_decay_no_label_without_arrival(env, valid_on):
    loss = step_decay_loss()
    fingertip, target = make_trial(env, valid_on, [])
    label = loss.trigger_label(fingertip[None], target[None], torch.tensor([valid_on]))
    assert label.sum() == 0


@pytest.mark.parametrize("pos_weight", [74.0, "auto"])
def test_step_decay_two_bins_late_beats_never_firing(env, pos_weight):
    loss = step_decay_loss(trigger_pos_weight=pos_weight)
    trial = [make_trial(env, VALID_ON, np.r_[a:N_TIME]) for a in (ARRIVAL, ARRIVAL + 2)]
    valid_on = torch.tensor([float(VALID_ON)])
    label, late = (loss.trigger_label(f[None], t[None], valid_on) for f, t in trial)
    never = torch.zeros_like(label)
    assert loss.trigger_loss(late, label) < loss.trigger_loss(never, label)


def test_auto_pos_weight_balances_labeled_and_other_bins():
    pulse = ActorLoss(torch.nn.MSELoss(), 1.0, 1.0, trigger_pos_weight="auto")
    assert pulse.labeled_weight(N_TIME) == pytest.approx(74.0)
    loss = step_decay_loss(trigger_pos_weight="auto")
    label = torch.zeros(1, N_TIME)
    label[0, 110 : 110 + 40] = loss.label_shape()
    weight = 1.0 + (loss.labeled_weight(N_TIME) - 1.0) * label
    labeled = label > 0
    assert weight[labeled].sum().item() == pytest.approx((~labeled).sum().item())


def test_unknown_trigger_shape_is_rejected():
    with pytest.raises(ValueError):
        ActorLoss(torch.nn.MSELoss(), 1.0, 1.0, trigger_shape="ramp")
