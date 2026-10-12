import pytorch_lightning as pl


class ActorLossTermsLogger(pl.Callback):
    """Logs the position, effort and trigger parts of ActorLoss separately.

    ActorLoss keeps the parts of its most recent call in `last_terms`; this
    callback logs them as train/<part>_loss and valid/<part>_loss.
    """

    def _log_terms(self, pl_module, split):
        for name, value in pl_module.loss_func.last_terms.items():
            pl_module.log(f"{split}/{name}_loss", value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_terms(pl_module, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log_terms(pl_module, "valid")
