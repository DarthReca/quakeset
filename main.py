import comet_ml
import hydra
import pytorch_lightning as pl
import torch
from dataset import EarthQuakeDataModule, EarthQuakeDataset
from model import EarthQuakeModel
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader, Dataset, Subset


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    data_module = EarthQuakeDataModule(**args.dataset)

    model = EarthQuakeModel(**args.model)

    api_key = None
    if args.log_comet:
        with open(".comet") as f:
            api_key = f.read().strip()

    comet_logger = CometLogger(
        api_key=api_key,
        project_name="",
        workspace="",
        experiment_name=model.hparams["model_name"],
        save_dir="comet-logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{comet_logger.experiment.id}",
        filename="earthquake-detection-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        **args.trainer,
        logger=comet_logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
