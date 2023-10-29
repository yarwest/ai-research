import logging

import numpy as np
import torch
import torchvision.models.detection.faster_rcnn as torch_frcnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from neuralnet.metrics.evaluators.pascal_voc_evaluator import (
    get_pascalvoc_metrics,
)
from neuralnet.metrics.utils.enumerators import MethodAveragePrecision
from neuralnet.utils import get_bounding_boxes
from data.voc_dataset import VOCDataset


class FasterRcnnModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.0001,
        iou_threshold=0.5,
        num_classes=8,
        trainable_backbone_layers=3,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.trainable_backbone_layers = trainable_backbone_layers
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Logger wypisujący wartości do terminala
        self.cli_logger = logging.getLogger("lightning")

        # Konfiguracja modelu
        self.model = torch_frcnn.fasterrcnn_resnet50_fpn(
            weights=torch_frcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=torch_frcnn.ResNet50_Weights.DEFAULT,
            trainable_backbone_layers=self.trainable_backbone_layers,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torch_frcnn.FastRCNNPredictor(
            in_features, self.num_classes
        )

        self.save_hyperparameters()

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, names = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, names = batch
        detections = self.model(images)

        ground_truth_boxes = []
        detected_boxes = []
        for name, target, detection in zip(names, targets, detections):
            ground_truth_boxes += get_bounding_boxes(name, target, is_ground_truth=True)
            detected_boxes += get_bounding_boxes(name, detection, is_ground_truth=False)

        return {
            "ground_truth_boxes": ground_truth_boxes,
            "detected_boxes": detected_boxes,
        }

    def on_validation_epoch_end(self, outs):
        ground_truth_boxes = []
        detected_boxes = []
        for out in outs:
            ground_truth_boxes += out["ground_truth_boxes"]
            detected_boxes += out["detected_boxes"]

        metrics = get_pascalvoc_metrics(
            gt_boxes=ground_truth_boxes,
            det_boxes=detected_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, mean_average_precision = metrics["per_class"], metrics["mAP"]
        map_flt = mean_average_precision.astype(np.float64)
        self.log("validation_mAP", map_flt)
        self.cli_logger.info(f"validation_mAP = {map_flt}")

        for k, v in per_class.items():
            name = LTLDDataset.INT_TO_CLS[k]
            ap = v["AP"].astype(np.float64)
            self.log(f"validation_AP_for_{name}", ap)
            self.cli_logger.info(f"validation_AP_for_{name} = {ap}")

    def test_step(self, batch, batch_idx):
        images, targets, names = batch
        detections = self.model(images)

        ground_truth_boxes = []
        detected_boxes = []
        for name, target, detection in zip(names, targets, detections):
            ground_truth_boxes += get_bounding_boxes(name, target, is_ground_truth=True)
            detected_boxes += get_bounding_boxes(name, detection, is_ground_truth=False)

        return {
            "ground_truth_boxes": ground_truth_boxes,
            "detected_boxes": detected_boxes,
        }

    def test_epoch_end(self, outs):
        ground_truth_boxes = []
        detected_boxes = []
        for out in outs:
            ground_truth_boxes += out["ground_truth_boxes"]
            detected_boxes += out["detected_boxes"]

        metrics = get_pascalvoc_metrics(
            gt_boxes=ground_truth_boxes,
            det_boxes=detected_boxes,
            iou_threshold=self.iou_threshold,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True,
        )

        per_class, mean_average_precision = metrics["per_class"], metrics["mAP"]
        map_flt = mean_average_precision.astype(np.float64)
        self.log("test_mAP", map_flt)
        self.cli_logger.info(f"test_mAP = {map_flt}")

        for k, v in per_class.items():
            name = LTLDDataset.INT_TO_CLS[k]
            ap = v["AP"].astype(np.float64)
            self.log(f"test_AP_for_{name}", ap)
            self.cli_logger.info(f"test_AP_for_{name} = {ap}")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

        return optimizer

    # def configure_callbacks(self):
        # early_stop = EarlyStopping(
        #     monitor="validation_mAP",
        #     min_delta=self.early_stopping_min_delta,
        #     patience=self.early_stopping_patience,
        #     mode="max",
        # )
        # checkpoint = ModelCheckpoint(monitor="validation_mAP", mode="max")
        # return [early_stop, checkpoint]