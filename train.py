import argparse
from typing import Dict

import pytorch_lightning as pl
import yaml

from dataloaders import SegmentationDataModule
from utils import average, object_from_dict, state_dict_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config", required=True
    )

    return parser.parse_args()


class SegmentationPipeline(pl.LightningModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()

        self.hparams.update(hparams)

        self.model = object_from_dict(self.hparams["model"])
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.criterion = object_from_dict(self.hparams["criterion"])
        self.metric = object_from_dict(self.hparams["metric"])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)

        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"train_loss": loss, "train_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"val_loss": loss, "val_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return logs

    def validation_epoch_end(self, outputs):
        logs = {}
        avg_val_score = average(outputs, "val_metrics")
        logs["val_score"] = avg_val_score
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_score": avg_val_score, "log": logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.unsqueeze(1)

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        score = self.metric(y_pred, y)

        logs = {"test_loss": loss, "test_metrics": score}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_metrics": score}

    def test_epoch_end(self, outputs):
        logs = {}
        avg_val_score = average(outputs, "test_metrics")
        logs["test_score"] = avg_val_score
        return {"test_score": avg_val_score, "log": logs}


def main():
    args = parse_args()

    with open(args.config) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pl.seed_everything(hparams["seed"])

    datamodule = SegmentationDataModule(hparams)
    model_pipeline = SegmentationPipeline(hparams)

    trainer = object_from_dict(hparams["trainer"])
    trainer.fit(model_pipeline, datamodule)
    trainer.test(
        datamodule=datamodule,
        verbose=True,
    )


if __name__ == "__main__":
    main()
