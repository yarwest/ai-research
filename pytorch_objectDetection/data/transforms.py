import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_training_transforms():
    transforms = A.Compose(
        [
            # Zastosuj lustrzane odbicie z prawdopodobieństwem 50%
            A.RandomBrightnessContrast(p=0.25),
            # Skonwertuj obraz PIL do tensora PyTorch
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    return transforms


def get_test_transforms():
    transforms = A.Compose(
        [
            # Skonwertuj obraz PIL do tensora PyTorch
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    return transforms