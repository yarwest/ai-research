import argparse
import os
import cv2
import uuid
import math
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

from data.voc_dataset import VOCDataModule
from neuralnet.model import FasterRcnnModel

__author__ = "Yarno Boelens"

DEFAULT_THRESHOLD = 0.8

def getDatasetPath(fileName):
    return os.path.join(os.path.dirname(__file__), f"{fileName}")

def print_annotated_image(
    image, boxes, scores, labels, threshold, output_path=None
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
        printImg(image, output_path)

    # _show_cv2_img("Predicted image", image)

def printImg(image, path):
    cv2.imwrite(path, image)

def extractForeground(image, boxes, scores, labels, threshold):
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score <= threshold:
            continue
        # create a simple mask image similar 
        # to the loaded image, with the  
        # shape and return type 
        mask = np.zeros(image.shape[:2], np.uint8)

        # specify the background and foreground model 
        # using numpy the array is constructed of 1 row 
        # and 65 columns, and all array elements are 0 
        # Data type for the array is np.float64 (default) 
        backgroundModel = np.zeros((1, 65), np.float64) 
        foregroundModel = np.zeros((1, 65), np.float64) 

        # define the Region of Interest (ROI) 
        # as the coordinates of the rectangle 
        # where the values are entered as 
        # (startingPoint_x, startingPoint_y, width, height) 
        int_box = [int(coordinate) for coordinate in box]
        startingY = int_box[1]
        startingX = int_box[0]
        endingY = int_box[3]
        endingX = int_box[2]

        width = endingX - startingX
        height = endingY - startingY

        if(startingX > 1):
            startingX -= 1
        if(startingY > 1):
            startingY -= 1

        rectangle = (startingX, startingY, width, height)

        # apply the grabcut algorithm with appropriate 
        # values as parameters, number of iterations = 3  
        # cv2.GC_INIT_WITH_RECT is used because 
        # of the rectangle mode is used  
        cv2.grabCut(image, mask, rectangle,   
                backgroundModel, foregroundModel, 
                3, cv2.GC_INIT_WITH_RECT) 

        # In the new mask image, pixels will  
        # be marked with four flags  
        # four flags denote the background / foreground  
        # mask is changed, all the 0 and 2 pixels  
        # are converted to the background 
        # mask is changed, all the 1 and 3 pixels 
        # are now the part of the foreground 
        # the return type is also mentioned, 
        # this gives us the final mask 
        mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 

        # The final mask is multiplied with  
        # the input image to give the segmented image. 
        image = image * mask2[:, :, np.newaxis]
        printImg(image, getDatasetPath(f"out/{label}-{idx}-{score}.png"))
      

def _show_cv2_img(window_title, image):
    # Based on https://medium.com/@mh_yip/ee51616f7088
    cv2.imshow(window_title, image)
    wait_time = 1000
    while cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

INT_TO_CLS = {
    1: "person"
}

def predict(model_path, image_path, threshold=DEFAULT_THRESHOLD, output_path=None):
    model = FasterRcnnModel.load_from_checkpoint(getDatasetPath(f"models/lightning_logs/{model_path}"))

    image = cv2.imread(getDatasetPath(f"data/{image_path}"))
    converted_image = _convert_image(image)

    predictions = model([converted_image])

    boxes = predictions[0]["boxes"].tolist()
    scores = predictions[0]["scores"].tolist()
    labels = [
        INT_TO_CLS[label]
        for label in predictions[0]["labels"].tolist()
    ]

    if(scores[0] < threshold):
        if scores.__len__() > 4:
            threshold = math.floor(scores[5] * 100)/100.0    
        else:
            threshold = math.floor(scores[0] * 100)/100.0

    print("Predictions:")
    for box, score, label in zip(boxes, scores, labels):
        print(f"Name: {label}\t| Score: {score}\t| Box: ({box})")

    # # We delete scores below the threshold here to not needlessly pass around unused predictions
    # for idx, score in enumerate(scores):
    #     if score <= threshold:
    #         del boxes[idx]
    #         del scores[idx]
    #         del labels[idx]

    extractForeground(image, boxes, scores, labels, threshold)

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
    processID = uuid.uuid4()
    print("---- Process ID:", processID)
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')
    predict(args.model, args.image, args.threshold, getDatasetPath(f"out/{processID}.png"))

def parse_args():
    parser = argparse.ArgumentParser(prog="predict.py", description="""
        This program uses a trained Faster R-CNN
        model to detect objects on images.
        """)
    
    parser.add_argument("-m", "--model", type=str, help="A path to the PyTorch Lightning checkpoint file with the model parameters.", required=True)
    parser.add_argument("-i", "--image", type=str, help="A path to an image file for prediction.")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD, help="An annotation will be drawn only if its score is higher than this value.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())