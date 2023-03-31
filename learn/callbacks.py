import pytorch_lightning as pl
import torch
from learn.models import circular_mse_loss


class LogChiCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch > 0:
            return
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for i in range(4):
                masks = batch.mask[:, i]
                trainer.logger.experiment.add_scalar(f"Chi_{i + 1}/Samples/Training",
                                              masks.long().sum(),
                                              trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Chi_{i + 1}/Target/Training",
                                                    batch.y[:, i][masks], global_step=trainer.global_step)
                
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch > 0:
            return
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for i in range(4):
                masks = batch.mask[:, i]
                trainer.logger.experiment.add_scalar(f"Chi_{i + 1}/Samples/Validation",
                                              masks.long().sum(),
                                              trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Chi_{i + 1}/Target/Validation",
                                                    batch.y[:, i][masks], global_step=trainer.global_step)
                
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        outputs = torch.cat([x['out'] for x in pl_module.validation_step_outputs], dim=0)
        masks = torch.cat([x['mask'] for x in pl_module.validation_step_outputs], dim=0)
        targets = torch.cat([x['y'] for x in pl_module.validation_step_outputs], dim=0)
        for i in range(4):
            trainer.logger.experiment.add_histogram(f"Chi_{i + 1}/Prediction",
                                                outputs[:, i][masks[:, i]].flatten(), trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f"Chi_{i + 1}/Loss",
                                            circular_mse_loss(outputs[:, i], targets[:, i], masks[:, i]),
                                            trainer.current_epoch)
        avg_loss = torch.mean(torch.stack([x['loss'] for x in pl_module.validation_step_outputs]))
        trainer.logger.experiment.add_scalar("Loss", avg_loss, trainer.current_epoch)
        self.log("hp_metric", avg_loss, sync_dist=True)

class LogParametersCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(f"Model/{name}", param, global_step=trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Model/{name}_grad", param.grad, global_step=trainer.global_step)