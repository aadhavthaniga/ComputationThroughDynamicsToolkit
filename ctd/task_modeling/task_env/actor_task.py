# Center-out reaching task for the "actor" network of the two-network model.
# Design spec: docs/actor_task_io.md
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ctd.task_modeling.task_env.loss_func import ActorLoss
from ctd.task_modeling.task_env.random_target import RandomTarget

# Same joint limits (degrees) that RandomTarget uses for its arm
SHO_LIMIT = (0.0, 135.0)
ELB_LIMIT = (0.0, 155.0)


class ActorTask(RandomTarget):
    """A center-out reach driven by the planner's outputs.

    The context inputs are the planner's outputs (TargetX, TargetY, TargetValid).
    Every trial starts at the same posture. While TargetValid is on, the hand goal
    is center + reach_dist * (TargetX, TargetY). The network has 7 outputs: the 6
    muscle activations, which drive the arm, and Trigger, which does not.

    Args:
        start_posture_deg: (shoulder, elbow) start angles in degrees.
        reach_dist: Distance R from the center to every goal, in meters.
        catch_prob: Fraction of trials where TargetValid never turns on.
        valid_on_ms: (min, max) time at which TargetValid turns on, in ms.
        joint_margin_deg: Goals must stay this far inside the joint limits.
        reach_radius: Distance to the goal (m) that counts as arrived.
        hold_ms: How long the hand must stay arrived before Trigger.
        trigger_ms: How long the Trigger label is 1.
        trigger_weight: Weight of the Trigger loss.
        trigger_pos_weight: Weight of the bins where the Trigger label is 1,
            or "auto" to balance them against the other bins.
        trigger_shape: "pulse" or "step_decay" (see :class:`ActorLoss`).
        decay_ms: Length of the fall to 0 in "step_decay" mode.
        full_trial_epoch: Epochs before the full trial is in the position loss.
        *args, **kwargs: Passed as-is to :class:`RandomTarget`.
    """

    def __init__(
        self,
        *args,
        start_posture_deg=(65.0, 90.0),
        reach_dist=0.10,
        catch_prob=0.2,
        valid_on_ms=(300, 600),
        joint_margin_deg=30.0,
        reach_radius=0.02,
        hold_ms=100,
        trigger_ms=40,
        trigger_weight=1.0,
        trigger_pos_weight=74.0,
        trigger_shape="pulse",
        decay_ms=40,
        full_trial_epoch=200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # RandomTarget's observation is fixed at 14 (2 vision + 12 proprioception),
        # so previous actions are never part of it. Stacking them in would also
        # require slicing Trigger out of the action buffer.
        if self.action_frame_stacking != 0:
            raise ValueError("ActorTask requires action_frame_stacking=0")

        self.dataset_name = "ActorTask"
        self.input_labels = ["TargetX", "TargetY", "TargetValid"]
        self.output_labels = self.output_labels + ["Trigger"]
        # The model's output size is read from action_space: 6 muscles + Trigger
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_muscles + 1,), dtype=np.float32
        )

        self.reach_dist = reach_dist
        self.catch_prob = catch_prob
        self.valid_on_ms = valid_on_ms
        self.joint_margin_deg = joint_margin_deg

        sho, elb = np.deg2rad(start_posture_deg)
        self.start_joint = torch.tensor([sho, elb, 0.0, 0.0], dtype=torch.float32)
        self.center = self._fingertip(self.start_joint[None, :])[0]
        self.check_goals_reachable()

        self.loss_func = ActorLoss(
            position_loss=nn.MSELoss(),
            pos_weight=kwargs.get("pos_weight", 1.0),
            act_weight=kwargs.get("act_weight", 1.0),
            full_trial_epoch=full_trial_epoch,
            reach_radius=reach_radius,
            hold_ms=hold_ms,
            trigger_ms=trigger_ms,
            trigger_weight=trigger_weight,
            trigger_pos_weight=trigger_pos_weight,
            trigger_shape=trigger_shape,
            decay_ms=decay_ms,
            dt=self.dt,
            n_muscles=self.n_muscles,
        )

    def _fingertip(self, joint_state):
        """Fingertip (x, y) for a batch of joint states."""
        return self.joint2cartesian(joint_state.to(self.device)).chunk(2, dim=-1)[0]

    def check_goals_reachable(self, n_angles=360):
        """Check that the whole circle of goals is inside the joint limits.

        Solves the two-link inverse kinematics for goals all around the circle,
        checks the joint angles against the limits, and confirms each solution by
        running it back through the arm's own forward kinematics.
        """
        l1, l2 = self.skeleton.L1, self.skeleton.L2
        angle = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        goals = self.center.cpu().numpy() + self.reach_dist * unit

        cos_elb = (np.sum(goals**2, axis=1) - l1**2 - l2**2) / (2 * l1 * l2)
        if np.any(np.abs(cos_elb) > 1):
            raise ValueError("reach_dist puts some goals beyond the arm's length")
        elb = np.arccos(cos_elb)
        sho = np.arctan2(goals[:, 1], goals[:, 0]) - np.arctan2(
            l2 * np.sin(elb), l1 + l2 * np.cos(elb)
        )

        margin = self.joint_margin_deg
        sho_deg, elb_deg = np.rad2deg(sho), np.rad2deg(elb)
        in_limits = (
            (sho_deg >= SHO_LIMIT[0] + margin)
            & (sho_deg <= SHO_LIMIT[1] - margin)
            & (elb_deg >= ELB_LIMIT[0] + margin)
            & (elb_deg <= ELB_LIMIT[1] - margin)
        )
        if not in_limits.all():
            raise ValueError(
                f"{(~in_limits).sum()} of {n_angles} goals are outside the joint "
                f"limits (margin {margin} deg). Reduce reach_dist or move "
                "start_posture_deg."
            )

        joints = np.stack([sho, elb, np.zeros(n_angles), np.zeros(n_angles)], axis=1)
        reached = self._fingertip(torch.tensor(joints, dtype=torch.float32))
        error = np.abs(reached.cpu().numpy() - goals).max()
        if error > 1e-4:
            raise ValueError(f"Inverse kinematics check failed (error {error:.2e} m)")

    def generate_dataset(self, n_samples):
        """
        Generate a dataset for the task

        Args:
            n_samples (int) : Number of trials

        Returns:
            dataset_dict (dict) :
            extra_dict (dict) :
        """
        n_time = int(self.n_timesteps)
        on_min, on_max = (int(round(ms / 1000 / self.dt)) for ms in self.valid_on_ms)

        catch = np.random.random(n_samples) < self.catch_prob
        valid_on = np.random.randint(on_min, on_max + 1, n_samples)
        valid_on[catch] = -1
        angle = np.random.uniform(0, 2 * np.pi, n_samples)
        unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)

        # (n_samples, n_time): True from valid_on to the end, never on catch trials
        valid = (np.arange(n_time)[None, :] >= valid_on[:, None]) & ~catch[:, None]

        # The target is shown for exactly as long as TargetValid is on
        true_inputs = np.zeros((n_samples, n_time, 3))
        true_inputs[:, :, 0:2] = valid[:, :, None] * unit[:, None, :]
        true_inputs[:, :, 2] = valid
        inputs = true_inputs + np.random.normal(
            0, self.context_input_noise, true_inputs.shape
        )

        # Hand goal: the center until TargetValid turns on, then the target
        center = self.center.cpu().numpy()
        targets = center + valid[:, :, None] * self.reach_dist * unit[:, None, :]

        trial_end = np.full(n_samples, n_time)
        dataset_dict = {
            "ics": self.start_joint.repeat(n_samples, 1),
            "inputs": inputs,
            "inputs_to_env": np.zeros((n_samples, n_time, 2)),  # no bumps
            "targets": torch.tensor(targets, dtype=torch.float32),
            "conds": catch.astype(int),
            "extra": np.stack((valid_on, trial_end), axis=1),
            "true_inputs": true_inputs,
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def update_obs_buffer(self, action=None):
        # The parent reshapes the action to action_space's width (7 here) before
        # buffering it. With action_frame_stacking=0 the buffered action is never
        # observed, so skip it.
        super().update_obs_buffer(action=None)

    def step(self, action, **kwargs):
        """Step the arm with the muscle activations only.

        Trigger (the last output) is not sent to the arm.
        """
        return super().step(action[:, : self.n_muscles], **kwargs)


if __name__ == "__main__":
    # Self-check: python -m ctd.task_modeling.task_env.actor_task
    from pathlib import Path

    import hydra
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf

    # Same arm and noise settings as RandomTarget
    cfg = OmegaConf.load(
        Path(__file__).parents[1] / "configs" / "env_task" / "RandomTarget.yaml"
    )
    del cfg["_target_"]
    env = ActorTask(**hydra.utils.instantiate(cfg, _convert_="all"))
    env.set_seed(0)
    print(f"center (m): {env.center.tolist()}  R (m): {env.reach_dist}")
    print(f"dt: {env.dt}  n_timesteps: {env.n_timesteps}")
    print(f"action_space: {env.action_space.shape}  outputs: {env.output_labels}")

    n_trials = 16
    data, _ = env.generate_dataset(n_trials)
    for key, val in data.items():
        print(f"{key}: {tuple(val.shape)}")

    valid_on, catch = data["extra"][:, 0], data["conds"].astype(bool)
    assert data["ics"].unique(dim=0).shape[0] == 1
    assert np.array_equal(valid_on == -1, catch)
    assert np.all((valid_on[~catch] >= 30) & (valid_on[~catch] <= 60))

    # Trigger must not reach the arm: same muscles, different Trigger, same hand
    fingertips = []
    for trigger in (0.0, 1.0):
        obs, _ = env.reset(batch_size=n_trials, options={"ic_state": data["ics"]})
        action = torch.full((n_trials, 7), 0.2)
        action[:, -1] = trigger
        for _ in range(20):
            obs, _, _, _, info = env.step(action=action, deterministic=True)
        fingertips.append(info["states"]["fingertip"])
    assert obs.shape == (n_trials, 14)
    assert torch.equal(fingertips[0], fingertips[1])
    print("checks passed")

    # Figure: inputs vs time for 3 trials, and the workspace
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    styles = ["-", "-", "--"]
    shown = list(np.flatnonzero(~catch)[:2]) + list(np.flatnonzero(catch)[:1])
    time = np.arange(env.n_timesteps) * env.dt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, trial in zip(axes.flat, shown):
        for i, label in enumerate(env.input_labels):
            ax.plot(
                time,
                data["inputs"][trial, :, i],
                color=colors[i],
                ls=styles[i],
                lw=1.5,
                label=label,
            )
        kind = "catch" if catch[trial] else f"Valid on at {valid_on[trial] * 10} ms"
        ax.set_title(f"Trial {trial}: {kind}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Input")
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc="lower center", ncol=3, frameon=False)

    ax = axes.flat[3]
    center = env.center.cpu().numpy()
    goals = data["targets"][:, -1, :].numpy()
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        center[0] + env.reach_dist * np.cos(circle),
        center[1] + env.reach_dist * np.sin(circle),
        color="#52514e",
        lw=1,
        label="Possible goals",
    )
    ax.scatter(*goals[~catch].T, s=45, color=colors[0], label="Trial goal")
    ax.scatter(*center, s=90, marker="+", color="#0b0b0b", label="Center")
    ax.set_title(f"Workspace: {(~catch).sum()} reach trials, {catch.sum()} catch")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("actor_task_examples.png", dpi=150)
    print("saved actor_task_examples.png")
