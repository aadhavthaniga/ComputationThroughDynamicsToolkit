import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from ctd.task_modeling.task_env.loss_func import StackLoss
from ctd.task_modeling.task_env.task_env import DecoupledEnvironment

# Task name -> (min, max) number of targets shown per trial
TASKS = {
    "SingleRecall": (1, 1),
    "ForwardRecall": (2, 3),
    "BackwardRecall": (2, 3),
}

# Tasks that release the targets last-in-first-out instead of in presentation order
LIFO_TASKS = {"BackwardRecall"}

# Trial timings (ms). Ranges are inclusive.
FIXATION_MS = 200
ITEM_MS = 200
GAP_MS = (100, 300)
DELAY_MS = (300, 1000)
TRIGGER_MS = 40
TRIGGER_INTERVAL_MS = (400, 900)
TAIL_MS = 300

# Minimum angular separation between targets in a trial
MIN_SEP = np.deg2rad(30)


class StackTask(DecoupledEnvironment):
    """
    Sequence-memory task for the "Stack" region of a two-region model.

    1 to 3 targets (points on the unit circle) are shown one at a time.
    After a delay, Go turns on and the output switches to the first target to
    be released. Each Trigger pulse advances the output to the next one; the
    trigger after the last target sets the output to zero. ForwardRecall
    releases the targets in presentation order (first-in-first-out),
    BackwardRecall in reverse (last-in-first-out). Both draw their trials
    identically, so with the same seed only the output order differs.

    Inputs:
        ItemX, ItemY: the target currently being shown (zero otherwise)
        Go: turns on after the delay and stays on until the trial ends
        Trigger: brief pulses, one per target, all after Go
    Outputs:
        TargetX, TargetY: the target currently being recalled
        TargetValid: 1 while a target is being output, else 0
    """

    def __init__(
        self,
        task_list: list = ("SingleRecall", "ForwardRecall"),
        n_timesteps: int = 300,
        noise: float = 0.1,
        bin_ms: int = 20,
        dataset_name: str = "StackTask",
    ):
        """
        Args:
            task_list: Tasks to include ("SingleRecall", "ForwardRecall",
                "BackwardRecall")
            n_timesteps: Number of bins per trial (shorter trials are zero-padded)
            noise: Std of the Gaussian noise added to the inputs
            bin_ms: Bin size in ms
            dataset_name: Name of the dataset
        """
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        unknown = [task for task in task_list if task not in TASKS]
        if unknown:
            raise ValueError(
                f"Unknown task(s) {unknown}. Available tasks: {list(TASKS)}"
            )
        self.task_list = list(task_list)
        self.bin_size = bin_ms
        self.dataset_name = dataset_name

        # Fail early if the longest possible trial doesn't fit
        for task in self.task_list:
            max_len = self._max_trial_len(TASKS[task][1])
            if max_len > n_timesteps:
                raise ValueError(
                    f"Longest possible {task} trial is {max_len} bins "
                    f"({max_len * bin_ms} ms at bin_ms={bin_ms}), but "
                    f"n_timesteps={n_timesteps}. Increase n_timesteps or bin_ms."
                )

        self.input_labels = ["ItemX", "ItemY", "Go", "Trigger"]
        self.output_labels = ["TargetX", "TargetY", "TargetValid"]
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(4,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-1.5, high=1.5, shape=(0,), dtype=np.float32
        )
        self.coupled_env = False
        self.loss_func = StackLoss()

    def _bins(self, ms):
        return int(round(ms / self.bin_size))

    def _max_trial_len(self, n_items):
        b = self._bins
        return (
            b(FIXATION_MS)
            + n_items * b(ITEM_MS)
            + (n_items - 1) * b(GAP_MS[1])
            + b(DELAY_MS[1])
            + n_items * b(TRIGGER_INTERVAL_MS[1])
            + b(TRIGGER_MS)
            + b(TAIL_MS)
        )

    def _randbins(self, ms_range):
        return np.random.randint(self._bins(ms_range[0]), self._bins(ms_range[1]) + 1)

    def _sample_angles(self, n_items):
        # Rejection sampling: at most 3 items, so this takes a few tries at worst
        while True:
            angles = np.random.uniform(0, 2 * np.pi, size=n_items)
            diff = np.abs(angles[:, None] - angles[None, :])
            diff = np.minimum(diff, 2 * np.pi - diff)
            if n_items == 1 or diff[np.triu_indices(n_items, k=1)].min() >= MIN_SEP:
                return angles

    def set_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """No step is implemented: trials are generated in generate_trial"""
        pass

    def reset(self):
        """No reset is implemented: trials are generated in generate_trial"""
        pass

    def generate_trial(self, task_name):
        """
        Generates one trial (unpadded, length = trial_end_bin)

        Returns:
            inputs_noise, outputs, true_inputs, info (dict of trial events)
        """
        b = self._bins
        min_items, max_items = TASKS[task_name]
        n_items = np.random.randint(min_items, max_items + 1)
        angles = self._sample_angles(n_items)
        targets = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        # Fixation, then items separated by gaps
        t = b(FIXATION_MS)
        item_on = []
        for i in range(n_items):
            if i > 0:
                t += self._randbins(GAP_MS)
            item_on.append(t)
            t += b(ITEM_MS)

        # Delay, then Go. Trigger intervals are onset-to-onset, starting at Go
        go = t + self._randbins(DELAY_MS)
        intervals = [self._randbins(TRIGGER_INTERVAL_MS) for _ in range(n_items)]
        trig_on = go + np.cumsum(intervals)
        trial_end = trig_on[-1] + b(TRIGGER_MS) + b(TAIL_MS)

        inputs = np.zeros((trial_end, 4))
        outputs = np.zeros((trial_end, 3))
        for on, target in zip(item_on, targets):
            inputs[on : on + b(ITEM_MS), 0:2] = target
        inputs[go:, 2] = 1
        for on in trig_on:
            inputs[on : on + b(TRIGGER_MS), 3] = 1

        # The i-th target released (presentation index release_order[i]) is
        # output from Go (i=0) or trigger i-1 until trigger i.
        # The last trigger leaves the output at zero.
        release_order = list(range(n_items))
        if task_name in LIFO_TASKS:
            release_order.reverse()
        switch = [go, *trig_on]
        for i, item in enumerate(release_order):
            outputs[switch[i] : switch[i + 1], 0:2] = targets[item]
            outputs[switch[i] : switch[i + 1], 2] = 1

        inputs_noise = inputs + self.noise * np.random.randn(*inputs.shape)
        info = {
            "task_name": task_name,
            "n_items": n_items,
            "angles": angles,
            "release_order": release_order,
            "item_on": item_on,
            "go": go,
            "trigger_on": trig_on.tolist(),
            "trial_end": trial_end,
        }
        return inputs_noise, outputs, inputs, info

    def generate_dataset(self, n_samples):
        """
        Generates a dataset for the Stack task

        Args:
            n_samples (int) : Number of samples to generate (from each task)

        Returns:
            dataset_dict (dict) :
            extra_dict (dict) :
        """
        n_trials = n_samples * len(self.task_list)
        inputs_ds = np.zeros((n_trials, self.n_timesteps, 4))
        true_inputs_ds = np.zeros((n_trials, self.n_timesteps, 4))
        outputs_ds = np.zeros((n_trials, self.n_timesteps, 3))
        conds_ds = np.zeros((n_trials, 1))
        extra_ds = np.zeros((n_trials, 2))
        trial_info = []

        for task_num, task in enumerate(self.task_list):
            for i in range(n_samples):
                ind = task_num * n_samples + i
                inputs_noise, outputs, inputs, info = self.generate_trial(task)
                trial_len = info["trial_end"]
                inputs_ds[ind, :trial_len, :] = inputs_noise
                true_inputs_ds[ind, :trial_len, :] = inputs
                outputs_ds[ind, :trial_len, :] = outputs
                conds_ds[ind, 0] = task_num
                extra_ds[ind, :] = [info["go"], trial_len]
                trial_info.append(info)

        dataset_dict = {
            "ics": np.zeros((n_trials, 3)),
            "inputs": inputs_ds,
            "inputs_to_env": np.zeros((n_trials, 0)),
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": conds_ds,
            "extra": extra_ds,
        }
        extra_dict = {
            "trial_info": trial_info,
            "task_names": [info["task_name"] for info in trial_info],
        }
        return dataset_dict, extra_dict

    def plot_trial(self, task_name=None):
        """Plots the inputs and outputs of one trial (default: first task)"""
        task_name = task_name or self.task_list[0]
        inputs, outputs, _, info = self.generate_trial(task_name)
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

        for i in (0, 1):
            axes[0].plot(inputs[:, i], label=self.input_labels[i])
        axes[0].set_ylabel("Items")
        for i in (2, 3):
            axes[1].plot(inputs[:, i], label=self.input_labels[i])
        axes[1].set_ylabel("Go / Trigger")
        for i in range(3):
            axes[2].plot(outputs[:, i], label=self.output_labels[i])
        axes[2].set_ylabel("Outputs")
        axes[2].set_xlabel(f"Time (bins of {self.bin_size} ms)")

        for ax in axes:
            ax.axvline(info["go"], color="k", linestyle="--")
            for on in info["trigger_on"]:
                ax.axvline(on, color="gray", linestyle=":")
            ax.set_ylim(-1.5, 1.5)
            ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(f"Trial: {task_name} ({info['n_items']} items)")
        plt.tight_layout()
        plt.show()
        return fig
