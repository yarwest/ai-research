import numpy as np

from neuralnet.metrics.bounding_box import BoundingBox, BBType, BBFormat


def get_bounding_boxes(img_name, meta_dict, is_ground_truth=False):
    FORMAT = BBFormat.XYX2Y2

    labels = meta_dict["labels"]
    boxes = meta_dict["boxes"]

    if is_ground_truth:
        bb_type = BBType.GROUND_TRUTH
        confidences = np.full(len(labels), None)
    else:
        bb_type = BBType.DETECTED
        confidences = np.array(meta_dict["scores"].cpu())

    bounding_boxes = []
    for label, box, confidence in zip(labels, boxes, confidences):
        rounded_box = tuple(
            box.round()
        )  # Bug związany z obliczeniami zmiennoprzecinkowymi w bibliotece z metrykami
        bb = BoundingBox(
            image_name=img_name,
            class_id=int(label),
            coordinates=rounded_box,
            format=FORMAT,
            bb_type=bb_type,
            confidence=confidence,
        )
        bounding_boxes.append(bb)

    return bounding_boxes
