# StackTask analysis scripts

Analyses of the trained StackTask (planner) GRU. Each script loads the model
and training datamodule from

    content/trained_models/task-trained/<RUN_TAG>/max_epochs=200_batch_size=64_seed=0/

and writes its figures to `content/runs/task-trained/<RUN_TAG>/`. `RUN_TAG` is
set near the top of each script (currently `20261007_StackTask_test`); change
it to analyse a different run. Paths are resolved from the repo root, so the
scripts can be run from any directory.

## Before running

Train the model first:

    python examples/run_task_training.py

with `TASK = "StackTask"`, `MODEL = "GRU_RNN"`, `max_epochs` 200 and
`batch_size` 64 in `SEARCH_SPACE`, and `n_samples: 2000` in
`configs/datamodule_train/datamodule_StackTask.yaml`.

`loss_and_recall_accuracy.py` also needs the Lightning `metrics.csv`. Ray
writes it to a temporary working directory
(`/tmp/ray/session_*/artifacts/*/train_*/working_dirs/<trial>/`); copy that
folder to `content/runs/task-trained/<RUN_TAG>/lightning_logs/`.

## Scripts

| Script | Command | Produces |
| --- | --- | --- |
| `loss_and_recall_accuracy.py` | `python examples/stack_task_analysis/loss_and_recall_accuracy.py` | `planner_loss_curve.png` (train/valid loss vs epoch) and `planner_outputs_vs_targets.png` (output vs target for one SingleRecall, one 2-item and one 3-item ForwardRecall validation trial). Prints the mean angular error during each target's hold period, by task and item, and the fraction of trials with every target within 15 degrees. |
| `delay_state_geometry.py` | `python examples/stack_task_analysis/delay_state_geometry.py` | `planner_delay_states_pca.png` (PCA of the SingleRecall hidden state at the last bin before Go, coloured by target angle). Prints the variance explained by the top PCs, cross-validated R^2 of linear decoders for item 1 and item 2 in two-item ForwardRecall, and the principal angles between the two decoder subspaces. |
| `pointer_vs_conveyor.py` | `python examples/stack_task_analysis/pointer_vs_conveyor.py` | `planner_pointer_vs_conveyor.png` (pre-Go item decoders applied through time around the first trigger, and the share of the delay code that lies along the TargetX/TargetY readout). Prints the same quantities, plus a Go-aligned time course. |

## Notes

- The first script uses the saved validation split. The other two generate
  300 new trials per condition from the model's task environment with a fixed
  seed, so they do not depend on the cached dataset.
- Hold-period errors skip the first 80 ms (4 bins) after each target switch,
  matching the window the training loss ignores.
- Numbers vary slightly between runs (second decimal place).
