import argparse
import os
import logging
import numpy as np
import sys

import pytorch_lightning as pl

from neuralnet.model import FasterRcnnModel
from data.voc_dataset import VOCDataModule

np.set_printoptions(suppress=True)


# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_dataloader_num_workers(configuration):
    use_cpu_count = configuration.worker_per_cpu
    if use_cpu_count:
        return os.cpu_count()
    else:
        return configuration.number_of_workers


def is_tuning_required(configuration):
    return configuration.tune is True


def train_and_test(configuration):
    seed = pl.seed_everything(configuration.seed, workers=True)

    model = FasterRcnnModel(
        learning_rate=configuration.learning_rate,
        iou_threshold=configuration.iou_threshold,
        num_classes=configuration.num_of_classes,
        trainable_backbone_layers=configuration.trainable_backbone_layers,
        early_stopping_patience=configuration.early_stopping_patience,
        early_stopping_min_delta=configuration.early_stopping_min_delta,
    )

    datamodule = VOCDataModule(
        classes=configuration.classes,
        batch_size=configuration.batch_size,
        num_workers=get_dataloader_num_workers(configuration),
        random_state=seed,
    )

    trainer = pl.Trainer(
        default_root_dir=configuration.root_dir,
        max_epochs=configuration.max_epochs,
        limit_train_batches=configuration.limit_train_set,
        limit_val_batches=configuration.limit_valid_set,
        limit_test_batches=configuration.limit_test_set,
        fast_dev_run=configuration.fast_dev_run,
        deterministic=configuration.deterministic_trainer,
        devices=2,
        #TODO? maybe select the appropriate Accelerator by recognizing the machine you are on: accelerator="auto"
        accelerator="auto",
        log_every_n_steps=1,
    )

    if is_tuning_required(configuration):
        trainer.tune(model, datamodule)

    print("1. Training model")
    trainer.fit(model, datamodule)

    print("2. Testing model")
    trainer.test(datamodule=datamodule, ckpt_path="best")


def main(args):
    train_and_test(args)


def parse_args():
    parser = argparse.ArgumentParser(prog="train.py", description="""
        This program handles the training of a Faster R-CNN
        model for the task of traffic light detection.
        """)
    parser.add_argument('-m', '--model', default='fasterrcnn_resnet50_fpn_v2', help='name of the model')

    parser.add_argument("--seed", type=int, default=314159, required=False, help="")

    # dataloader args
    parser.add_argument("--worker_per_cpu", type=bool, default=True, required=False, help="")
    parser.add_argument("--num_workers", type=int, default=1, required=False, help="")
    parser.add_argument("--batch_size", type=int, default=4, required=False, help="")

    # model args
    parser.add_argument("--learning_rate", type=float, default=0.02, required=False, help="")
    parser.add_argument("--iou_threshold", type=float, default=0.5, required=False, help="")
    parser.add_argument("--num_of_classes", type=int, default=8, required=False, help="")
    parser.add_argument("--trainable_backbone_layers", type=int, default=3, required=False, help="")
    parser.add_argument("--early_stopping_patience", type=int, default=3, required=False, help="")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001, required=False, help="")

    # trainer args
    parser.add_argument("--root_dir", type=str, default="models", required=False, help="")
    parser.add_argument("--max_epochs", type=int, default=20, required=False, help="")
    parser.add_argument("--tune", type=bool, default=False, required=False, help="")
    parser.add_argument("--limit_train_set", type=float, default=0.1, required=False, help="")
    parser.add_argument("--limit_valid_set", type=float, default=0.1, required=False, help="")
    parser.add_argument("--limit_test_set", type=float, default=0.1, required=False, help="")
    parser.add_argument("--fast_dev_run", type=bool, default=False, required=False, help="")
    parser.add_argument("--deterministic_trainer", type=bool, default=False, required=False, help="")

    parser.add_argument("--classes", type=str, default="person", required=False, help="VOC Classes to be included in training")
    parser.add_argument("-p", "--print", action="store_true", help="Print parsed configuration before training.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())