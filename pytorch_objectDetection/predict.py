import argparse

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from data.voc_dataset import VOCDataModule
from neuralnet.model import FasterRcnnModel

__author__ = "Yarno Boelens"

DEFAULT_THRESHOLD = 0.7


def print_annotated_image(
    image, boxes, scores, labels, threshold=DEFAULT_THRESHOLD, output_path=None
):
    box_thickness = 2
    color = (0, 0, 255)  # Red
    font_type = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    for box, score, label in zip(boxes, scores, labels):
        if score <= threshold:
            continue

        int_box = [int(coordinate) for coordinate in box]
        top_pos = (int_box[0], int_box[1])
        bottom_pos = (int_box[2], int_box[3])
        text_pos = (top_pos[0], top_pos[1] - 10)

        cv2.rectangle(
            image,
            top_pos,
            bottom_pos,
            color,
            box_thickness,
        )

        cv2.putText(
            image,
            f"{label} {score:.3f}",
            text_pos,
            font_type,
            font_scale,
            color,
            font_thickness,
        )

    if output_path is not None:
        cv2.imwrite(output_path, image)

    _show_cv2_img("Predicted image", image)


def _show_cv2_img(window_title, image):
    # Based on https://medium.com/@mh_yip/ee51616f7088
    cv2.imshow(window_title, image)
    wait_time = 1000
    while cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def predict(model_path, image_path, threshold=DEFAULT_THRESHOLD, output_path=None):
    model = FasterRcnnModel.load_from_checkpoint(model_path)

    image = cv2.imread(image_path)
    converted_image = _convert_image(image)

    predictions = model([converted_image])

    boxes = predictions[0]["boxes"].tolist()
    scores = predictions[0]["scores"].tolist()
    labels = [
        # LisaTrafficLightDataset.INT_TO_CLS[label]
        for label in predictions[0]["labels"].tolist()
    ]

    print("Predictions:")
    for box, score, label in zip(boxes, scores, labels):
        print(f"Name: {label}\t| Score: {score}\t| Box: ({box})")

    print_annotated_image(image, boxes, scores, labels, threshold, output_path)


def _convert_image(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    converted = converted.astype("float32")
    converted /= 255

    transforms = A.Compose([ToTensorV2()])
    augmented = transforms(image=converted)
    converted = augmented["image"]

    return converted


def main(args):
    predict(args.model, args.image, args.threshold, args.output)

def parse_args():
    parser = argparse.ArgumentParser(prog="predict.py", description="""
        This program uses a trained Faster R-CNN
        model to detect objects on images.
        """)
    
    parser.add_argument("-m", "--model", type=str, help="A path to the PyTorch Lightning checkpoint file with the model parameters.", required=True)
    parser.add_argument("-i", "--image", type=str, help="A path to an image file for prediction.")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD, help="An annotation will be drawn only if its score is higher than this value.")
    parser.add_argument("-o","--output",type=str, help="A path where to save an annotated image.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())