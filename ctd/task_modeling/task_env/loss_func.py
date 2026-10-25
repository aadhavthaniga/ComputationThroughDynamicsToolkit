import numpy as np
import torch
import torch.nn as nn


def _dnms_response_cols(extra):
    if extra.shape[1] >= 13:
        return 8, 9
    if extra.shape[1] >= 11:
        return 6, 7
    raise ValueError(f"Unsupported DNMS extra shape: {tuple(extra.shape)}")


class LossFunc:
    def __init__():
        pass

    def __call__(self, loss_dict):
        pass


class RandomTargetLoss(LossFunc):
    def __init__(
        self, position_loss, pos_weight, act_weight, full_trial_epoch: int = 200
    ):
        """Initialize the loss function
        Args:
            position_loss (nn.Module): The loss function to use for the position
            pos_weight (float): The weight to apply to the position loss
            act_weight (float): The weight to apply to the action loss
            full_trial_epoch (int): The number of epochs
            before the full trial is included in the loss"""
        self.position_loss = position_loss
        self.action_loss = nn.MSELoss()
        self.pos_weight = pos_weight
        self.act_weight = act_weight
        self.full_trial_epoch = full_trial_epoch

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        epoch = loss_dict["epoch"]
        n_time = pred.shape[1]
        # Gradually increase the percent of the trial to include in the loss
        include_loss = np.ceil(n_time * min(1.0, epoch / self.full_trial_epoch)).astype(
            int
        )
        pos_loss = self.pos_weight * self.position_loss(
            pred[:, :include_loss, :], target[:, :include_loss, :]
        )
        act_loss = self.act_weight * self.action_loss(act, torch.zeros_like(act))
        return pos_loss + act_loss


class ActorLoss(RandomTargetLoss):
    def __init__(
        self,
        position_loss,
        pos_weight,
        act_weight,
        full_trial_epoch: int = 200,
        reach_radius: float = 0.02,
        hold_ms: float = 100,
        trigger_ms: float = 40,
        trigger_weight: float = 1.0,
        trigger_pos_weight=74.0,
        trigger_shape: str = "pulse",
        decay_ms: float = 40,
        dt: float = 0.01,
        n_muscles: int = 6,
    ):
        """Initialize the loss function
        Args:
            position_loss (nn.Module): The loss function to use for the position
            pos_weight (float): The weight to apply to the position loss
            act_weight (float): The weight to apply to the muscle effort loss
            full_trial_epoch (int): The number of epochs
            before the full trial is included in the position loss
            reach_radius (float): Distance to the goal (m) that counts as arrived
            hold_ms (float): How long the hand must stay arrived before Trigger
            trigger_ms (float): How long the Trigger label is 1
            trigger_weight (float): The weight to apply to the Trigger loss
            trigger_pos_weight (float or "auto"): Weight of the bins where the
            Trigger label is 1 (bins with a label between 0 and 1 get a weight in
            between); bins with label 0 have weight 1. The default makes the 4
            pulse bins count as much in total as the other 296 bins of a 3 s
            trial. "auto" computes that balance from the label shape and the
            trial length (see labeled_weight)
            trigger_shape (str): "pulse": the label is 1 for trigger_ms.
            "step_decay": 1 for trigger_ms, then falls linearly to 0 over decay_ms
            decay_ms (float): Length of the fall in "step_decay" mode
            dt (float): Timestep in seconds
            n_muscles (int): Number of muscle outputs; Trigger is the next output"""
        super().__init__(position_loss, pos_weight, act_weight, full_trial_epoch)
        self.reach_radius = reach_radius
        self.hold_steps = int(round(hold_ms / 1000 / dt))
        self.trigger_steps = int(round(trigger_ms / 1000 / dt))
        self.trigger_weight = trigger_weight
        self.trigger_pos_weight = trigger_pos_weight
        if trigger_shape not in ("pulse", "step_decay"):
            raise ValueError(f"Unknown trigger_shape: {trigger_shape}")
        self.trigger_shape = trigger_shape
        self.decay_steps = int(round(decay_ms / 1000 / dt))
        self.n_muscles = n_muscles
        # The three parts of the most recent loss, for ActorLossTermsLogger
        self.last_terms = {}

    # Fallbacks for runs pickled before these attributes existed: they were
    # trained with the pulse label, and before trigger_pos_weight with plain MSE
    trigger_shape = "pulse"
    decay_steps = 0
    trigger_pos_weight = 1.0

    def label_shape(self):
        """The Trigger label from its onset on, before the cut at the trial end"""
        on = torch.ones(self.trigger_steps)
        if self.trigger_shape == "pulse":
            return on
        # The k-th decay bin is 1 - k / decay_steps, so the last one is 0
        return torch.cat([on, torch.linspace(1.0, 0.0, self.decay_steps + 1)[1:]])

    def labeled_weight(self, n_time):
        """trigger_pos_weight, or for "auto" the weight that makes the labeled
        bins count as much in total as the other bins of an n_time trial"""
        if self.trigger_pos_weight != "auto":
            return self.trigger_pos_weight
        shape = self.label_shape()
        n_labeled = int((shape > 0).sum())
        # Labeled bins weigh 1 + (w - 1) * label; set their sum to n_time - n_labeled
        return 1.0 + (n_time - 2 * n_labeled) / shape.sum().item()

    def trigger_loss(self, trigger, label):
        """MSE between Trigger and its label, with the labeled bins up-weighted"""
        weight = 1.0 + (self.labeled_weight(label.shape[-1]) - 1.0) * label
        return (weight * (trigger - label) ** 2).mean()

    def trigger_label(self, fingertip, target, valid_on):
        """Compute the correct Trigger trace from the path the hand actually took

        The label (see label_shape) starts once the hand has stayed within
        reach_radius of the goal for hold_steps, counting only timesteps from
        valid_on onward, and is cut off at the trial end. At most one label per
        trial, and none on catch trials (valid_on = -1).
        """
        with torch.no_grad():
            n_trials, n_time, _ = fingertip.shape
            step = torch.arange(n_time, device=fingertip.device)
            at_goal = torch.linalg.norm(fingertip - target, dim=-1) < self.reach_radius
            at_goal &= (valid_on[:, None] >= 0) & (step[None, :] >= valid_on[:, None])
            # held[b, t]: at the goal for hold_steps in a row, starting at t
            held = at_goal.int().unfold(1, self.hold_steps, 1).min(dim=-1).values > 0
            label = torch.zeros(n_trials, n_time, device=fingertip.device)
            shape = self.label_shape().to(fingertip.device)
            for trial in torch.nonzero(held.any(dim=1)).flatten():
                start = int(held[trial].int().argmax()) + self.hold_steps
                end = min(start + len(shape), n_time)
                label[trial, start:end] = shape[: end - start]
        return label

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        epoch = loss_dict["epoch"]
        valid_on = loss_dict["extra"][:, 0]
        n_time = pred.shape[1]
        # Same curriculum as RandomTargetLoss, except that an empty window
        # (epoch 0) contributes nothing instead of NaN
        include_loss = int(np.ceil(n_time * min(1.0, epoch / self.full_trial_epoch)))
        pos_loss = 0.0
        if include_loss > 0:
            pos_loss = self.pos_weight * self.position_loss(
                pred[:, :include_loss, :], target[:, :include_loss, :]
            )
        muscles = act[..., : self.n_muscles]
        trigger = act[..., self.n_muscles]
        act_loss = self.act_weight * self.action_loss(
            muscles, torch.zeros_like(muscles)
        )
        label = self.trigger_label(pred, target, valid_on)
        trigger_loss = self.trigger_weight * self.trigger_loss(trigger, label)
        self.last_terms = {
            "position": torch.as_tensor(pos_loss).detach(),
            "effort": act_loss.detach(),
            "trigger": trigger_loss.detach(),
        }
        return pos_loss + act_loss + trigger_loss


class NBFFLoss(LossFunc):
    def __init__(self, transition_blind):
        """Initialize the loss function
        Args:
            transition_blind (int): The number of steps to
            ignore the effect of transitions for"""

        self.transition_blind = transition_blind

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]

        # Find where the change in the target is not zero
        # Step 1: Find where the transitions occur (change in value)
        transitions = torch.diff(target, dim=1) != 0

        # Initialize the mask with ones, with one less column (due to diff)
        mask = torch.ones_like(transitions, dtype=torch.float)

        # Step 2: Propagate the effect of transitions for 'transition_blind' steps
        for i in range(1, self.transition_blind + 1):
            # Shift the transition marks to the right to affect subsequent values
            shifted_transitions = torch.cat(
                (torch.zeros_like(transitions[:, :i]), transitions[:, :-i]), dim=1
            )
            mask = mask * (1 - shifted_transitions.float())

        # Step 3: Adjust mask size to match the original target tensor
        # Adding a column of ones at the beginning because diff reduces the size by 1
        final_mask = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
        final_mask[:, 0:5, :] = 0.0

        loss = nn.MSELoss(reduction="none")(pred, target) * final_mask
        return loss.mean()


class MatchTargetLossMSE(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.MSELoss()(pred, target)


class MultiTaskLoss(LossFunc):
    def __init__(self, lat_loss_weight=1e-6):
        self.lat_loss_weight = lat_loss_weight
        pass

    def __call__(self, loss_dict):

        """Calculate the loss"""
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        latents = loss_dict["latents"]
        # action = loss_dict["actions"]
        inputs = loss_dict["inputs"]
        extras = loss_dict["extra"]
        resp_start = extras[:, 0].long()
        resp_end = extras[:, 1].long()
        recon_loss = nn.MSELoss(reduction="none")(pred, target)
        mask = torch.ones_like(recon_loss)
        mask_lats = torch.ones_like(latents)

        # Ignore the first 5 time steps and the time steps after the response
        mask[:, 0:5, :] = 0
        for i in range(inputs.shape[0]):
            mask[i, resp_start[i] : resp_end[i], :] = 5.0
            mask[i, resp_start[i] : resp_start[i] + 5, :] = 0.0
            mask[i, resp_end[i] :, :] = 0.0
            # Mask the latents after the response
            mask_lats[i, resp_end[i] :, :] = 0

        masked_loss = recon_loss * mask
        lats_loss = (
            nn.MSELoss(reduction="none")(latents, torch.zeros_like(latents)) * mask_lats
        )

        total_loss = (
            masked_loss.sum(dim=1).mean()
            + self.lat_loss_weight * lats_loss.sum(dim=1).mean()
        )
        return total_loss


class PhaseCodedMemoryLoss(LossFunc):
    """
    MSE on the output ONLY after the stimulus offset (given in `extra`),
    plus an optional L2 penalty on the latents in that same window.

    loss_dict keys expected:
      - "controlled": (B, T, 1)  model outputs
      - "targets"   : (B, T, 1)  ground truth
      - "extra"     : (B,)       integer offset indices (inclusive)
      - "latents"   : (B, T, L)  OPTIONAL, only needed if latent penalty used
    """

    def __init__(
        self,
        post_stim_l2_weight: float = 0.0,
        eps: float = 1e-8,
        lat_l2_full: bool = False,
        pred_loss_full: bool = False,
    ):
        self.post_stim_l2_weight = post_stim_l2_weight
        self.eps = eps
        self.lat_l2_full = lat_l2_full
        self.pred_loss_full = pred_loss_full

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]  # (B, T, 1)
        target = loss_dict["targets"]  # (B, T, 1)
        signal_ind = loss_dict.get("extra")  # (B,2)
        onset_idx = signal_ind[:, 0] if signal_ind is not None else None
        offset_idx = signal_ind[:, 1] if signal_ind is not None else None

        if offset_idx is None:
            raise ValueError("PhaseCodedMemoryLoss requires 'extra' for offsets.")

        B, T, _ = pred.shape
        device = pred.device

        # ----- Mask for post-offset window -----
        time = torch.arange(T, device=device)  # (T,)
        mask_off = time.unsqueeze(0) >= offset_idx.unsqueeze(1)
        mask_on = time.unsqueeze(0) < onset_idx.unsqueeze(1)  # (B, T)
        mask = mask_off | mask_on  # (B, T)
        mask_full = mask.float().squeeze()  # (B, T, 1)
        mask_post = mask_off.float()  # (B, T)

        # ----- Output MSE in masked window -----
        se = (pred - target).pow(2).squeeze(-1)  # (B, T)
        if self.pred_loss_full:
            # If pred_loss_full is True, use the full prediction window
            mse_per_trial = (se * mask_full).sum(dim=1) / mask_full.sum(
                dim=1
            ).clamp_min(1)
        else:
            mse_per_trial = (se * mask_post).sum(dim=1) / mask_post.sum(
                dim=1
            ).clamp_min(1)
        mse_loss = mse_per_trial.mean()

        # ----- Latent L2  -----
        lat_loss = torch.tensor(0.0, device=device)
        if self.post_stim_l2_weight > 0.0:
            lats = loss_dict.get("latents")
            mean_lats = torch.zeros((B, lats.shape[-1]), device=device)  # (B, L)

            if lats is None:
                raise ValueError("Latent penalty requested but 'latents' missing.")
            # L2 norm per timestep (sqrt of sum of squares across latent dims)
            for i in range(lats.shape[0]):
                mean_lats[i, :] = lats[i, offset_idx[i].long() :].sum(dim=0) / (
                    T - offset_idx[i].long()
                )
            mean_lats_sq = mean_lats**2
            sum_mean_lats_sq = mean_lats_sq.sum(dim=1) / B

            lat_loss = sum_mean_lats_sq.mean()

        return mse_loss + self.post_stim_l2_weight * lat_loss


class MatchTargetLossIncrementalMSE(LossFunc):
    def __init__(self, init_bins=5, group_size=10, epochs_per_group=50):
        """Initialize the loss function
        Args:
            init_bins (int): The number of initial bins to use for the loss"""
        self.init_bins = init_bins
        self.group_size = group_size
        self.epochs_per_group = epochs_per_group

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        epoch = loss_dict["epoch"]
        n_time = pred.shape[1]

        # Calculate the number of bins to use
        n_bins = min(
            self.init_bins + self.group_size * (epoch // self.epochs_per_group), n_time
        )
        # Calculate the bin size

        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.MSELoss()(pred[:, :n_bins, :], target[:, :n_bins, :])


class ChaoticDelayedMemoryLoss(LossFunc):
    """Response-window mean MSE for ChaoticDelayedMemory.

    Expects `extra` columns:
    0 cue_on, 1 cue_off, 2 delay_on, 3 delay_off, 4 resp_on, 5 resp_off.
    """

    def __init__(
        self,
        response_weight: float = 5.0,
        delay_weight: float = 1.0,
        baseline_weight: float = 1.0,
    ):
        self.response_weight = response_weight
        self.delay_weight = delay_weight
        self.baseline_weight = baseline_weight

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        extra = loss_dict["extra"]

        resp_on = extra[:, 4].long()
        resp_off = extra[:, 5].long()

        pred_means = []
        target_means = []
        for i in range(pred.shape[0]):
            rs = int(resp_on[i].item())
            re = int(resp_off[i].item())
            pred_means.append(pred[i, rs:re, :].mean(dim=0))
            target_means.append(target[i, rs:re, :].mean(dim=0))

        pred_means = torch.stack(pred_means, dim=0)
        target_means = torch.stack(target_means, dim=0)
        return nn.MSELoss()(pred_means, target_means)


class ChaoticDelayedMatchingLoss(LossFunc):
    """MAE on the response-window mean for a single-neuron DNMS readout.

    Expects `extra` columns:
    0 cue1_on, 1 cue1_off, 2 delay1_on, 3 delay1_off,
    4 cue2_on, 5 cue2_off, 6 delay2_on, 7 delay2_off,
    8 resp_on, 9 resp_off.
    """

    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        extra = loss_dict["extra"]

        resp_on_col, resp_off_col = _dnms_response_cols(extra)
        resp_on = extra[:, resp_on_col].long()
        resp_off = extra[:, resp_off_col].long()

        if pred.shape[-1] != 1 or target.shape[-1] != 1:
            raise ValueError("Loss expects single-neuron preds and targs.")

        pred_means = []
        target_means = []
        for i in range(pred.shape[0]):
            rs = int(resp_on[i].item())
            re = int(resp_off[i].item())
            pred_means.append(pred[i, rs:re, 0].mean())
            target_means.append(target[i, rs:re, 0].mean())

        pred_means = torch.stack(pred_means, dim=0)
        target_means = torch.stack(target_means, dim=0)
        return nn.L1Loss()(pred_means, target_means)
