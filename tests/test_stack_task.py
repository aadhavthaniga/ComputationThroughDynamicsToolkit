"""Checks for the StackTask environment and StackLoss.

Run with `pytest tests/test_stack_task.py`, or run this file directly to also
save one example trial per task to stack_task_examples.png.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from ctd.task_modeling.task_env.stack_task import MIN_SEP, TASKS, StackTask

N_SAMPLES = 200


def onsets(signal):
    """Bins where a boolean signal turns on"""
    return np.flatnonzero(np.diff(np.r_[0, signal.astype(int)]) == 1)


def read_events(true_inputs):
    """Go bin, item onsets and trigger onsets, read from the clean inputs"""
    go = onsets(true_inputs[:, 2] != 0)[0]
    item_on = onsets((true_inputs[:, 0:2] != 0).any(axis=1))
    trig_on = onsets(true_inputs[:, 3] != 0)
    return go, item_on, trig_on


@pytest.fixture(scope="module")
def env():
    return StackTask()


@pytest.fixture(scope="module")
def data(env):
    env.set_seed(0)
    dataset_dict, _ = env.generate_dataset(N_SAMPLES)
    return dataset_dict


@pytest.fixture(scope="module")
def trials(env, data):
    """(task_name, true_inputs, targets, go, item_on, trig_on, trial_end) per trial"""
    out = []
    for k in range(N_SAMPLES * len(env.task_list)):
        task_name = env.task_list[int(data["conds"][k, 0])]
        true_inputs = data["true_inputs"][k]
        go, item_on, trig_on = read_events(true_inputs)
        trial_end = int(data["extra"][k, 1])
        out.append(
            (task_name, true_inputs, data["targets"][k], go, item_on, trig_on, trial_end)
        )
    return out


def test_dataset_shapes_and_labels(env, data, trials):
    n_trials = N_SAMPLES * len(env.task_list)
    assert data["inputs"].shape == (n_trials, env.n_timesteps, 4)
    assert data["targets"].shape == (n_trials, env.n_timesteps, 3)
    # n_samples trials per task, in task_list order
    assert (data["conds"][:, 0] == np.repeat([0, 1], N_SAMPLES)).all()
    for k, (task_name, true_inputs, _, go, item_on, _, _) in enumerate(trials):
        lo, hi = TASKS[task_name]
        assert lo <= len(item_on) <= hi
        assert data["extra"][k, 0] == go
        # Targets are on the unit circle and at least 30 degrees apart
        xy = true_inputs[item_on, 0:2]
        assert np.allclose(np.linalg.norm(xy, axis=1), 1)
        ang = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
        sep = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        assert len(ang) == 1 or sep.min() >= MIN_SEP - 1e-9


def test_output_is_zero_before_go(trials):
    for _, _, targets, go, _, _, _ in trials:
        assert not targets[:go].any()


def test_targets_come_out_in_presentation_order(trials):
    for _, true_inputs, targets, go, item_on, trig_on, _ in trials:
        switch = [go, *trig_on]
        for i, on in enumerate(item_on):
            shown = true_inputs[on, 0:2]
            segment = targets[switch[i] : switch[i + 1]]
            assert np.allclose(segment, [*shown, 1])


def test_one_trigger_per_item_all_after_go(trials):
    for _, true_inputs, _, go, item_on, trig_on, _ in trials:
        assert len(trig_on) == len(item_on)
        assert not true_inputs[: go + 1, 3].any()


def test_output_is_zero_after_last_trigger(trials):
    for _, _, targets, _, _, trig_on, _ in trials:
        assert not targets[trig_on[-1] :].any()


def test_every_trial_fits_in_n_timesteps(env, data, trials):
    for k, (_, true_inputs, _, _, _, trig_on, trial_end) in enumerate(trials):
        assert trial_end <= env.n_timesteps
        # The last trigger and the 300 ms after it are inside the trial
        assert trial_end == trig_on[-1] + 2 + 15
        # Only zero padding after the trial ends
        assert not data["inputs"][k, trial_end:].any()
        assert not true_inputs[trial_end:].any()


def test_too_short_n_timesteps_raises():
    with pytest.raises(ValueError, match="Longest possible ForwardRecall trial"):
        StackTask(n_timesteps=271)
    StackTask(n_timesteps=272)


def test_stack_loss_mask(env, data, trials):
    """Errors in ignored bins cost nothing, errors in kept bins do"""
    targets = torch.tensor(data["targets"], dtype=torch.float)
    extra = torch.tensor(data["extra"], dtype=torch.float)

    def loss_with_error_at(bins):
        pred = targets.clone()
        pred[0, bins] += 1
        return env.loss_func({"controlled": pred, "targets": targets, "extra": extra})

    _, _, _, go, _, trig_on, trial_end = trials[0]
    assert loss_with_error_at([]) == 0
    assert loss_with_error_at(list(range(0, 5))) == 0
    assert loss_with_error_at(list(range(go, go + 4))) == 0
    assert loss_with_error_at(list(range(trig_on[0], trig_on[0] + 4))) == 0
    assert loss_with_error_at(list(range(trial_end, env.n_timesteps))) == 0
    for kept in [5, go - 1, go + 4, trig_on[0] + 4, trial_end - 1]:
        assert loss_with_error_at([kept]) > 0, kept


@pytest.fixture(scope="module")
def lifo_data():
    """Same seed as `data`, with BackwardRecall in place of ForwardRecall"""
    lifo_env = StackTask(
        task_list=["SingleRecall", "BackwardRecall"], dataset_name="StackTaskLIFO"
    )
    lifo_env.set_seed(0)
    return lifo_env.generate_dataset(N_SAMPLES)


def test_backward_recall_outputs_targets_in_reverse_order(lifo_data):
    data, extra_dict = lifo_data
    n_backward = 0
    for k, info in enumerate(extra_dict["trial_info"]):
        true_inputs = data["true_inputs"][k]
        go, item_on, trig_on = read_events(true_inputs)
        backward = info["task_name"] == "BackwardRecall"
        order = list(range(len(item_on)))
        if backward:
            order.reverse()
            n_backward += 1
        assert info["release_order"] == order
        switch = [go, *trig_on]
        for i, item in enumerate(order):
            shown = true_inputs[item_on[item], 0:2]
            segment = data["targets"][k, switch[i] : switch[i + 1]]
            assert np.allclose(segment, [*shown, 1])
    assert n_backward == N_SAMPLES


def test_forward_recall_release_order_is_presentation_order(env):
    env.set_seed(1)
    for task_name in ("SingleRecall", "ForwardRecall"):
        for _ in range(20):
            info = env.generate_trial(task_name)[3]
            assert info["release_order"] == list(range(info["n_items"]))


def test_backward_and_forward_recall_draw_identical_trials(data, lifo_data):
    lifo, _ = lifo_data
    for key in ("inputs", "true_inputs", "extra", "conds", "ics"):
        assert np.array_equal(data[key], lifo[key]), key
    # Only the multi-item targets differ: same SingleRecall half, reversed rest
    single, multi = slice(0, N_SAMPLES), slice(N_SAMPLES, None)
    assert np.array_equal(data["targets"][single], lifo["targets"][single])
    assert not np.array_equal(data["targets"][multi], lifo["targets"][multi])


def save_examples(path):
    """Saves one example trial of each task: inputs on top, target outputs below"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = StackTask()
    env.set_seed(3)
    fig, axes = plt.subplots(
        nrows=2, ncols=len(env.task_list), sharex="col", sharey=True, figsize=(12, 6)
    )
    for col, task_name in enumerate(env.task_list):
        inputs, outputs, _, info = env.generate_trial(task_name)
        for i, label in enumerate(env.input_labels):
            axes[0, col].plot(inputs[:, i], label=label)
        for i, label in enumerate(env.output_labels):
            axes[1, col].plot(outputs[:, i], label=label)
        for ax in axes[:, col]:
            ax.axvline(info["go"], color="k", linestyle="--")
            for on in info["trigger_on"]:
                ax.axvline(on, color="gray", linestyle=":")
            ax.set_ylim(-1.5, 1.5)
            ax.legend(loc="upper right", fontsize=8)
        axes[0, col].set_title(f"{task_name} (n_items = {info['n_items']})")
        axes[1, col].set_xlabel(f"Time (bins of {env.bin_size} ms)")
    axes[0, 0].set_ylabel("Inputs")
    axes[1, 0].set_ylabel("Target outputs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    if pytest.main([__file__, "-q"]) == 0:
        save_examples(Path(__file__).parent.parent / "stack_task_examples.png")
