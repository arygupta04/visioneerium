# Assumption /data is loaded within the mask_rcnn folder

# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pycocotools.coco import COCO
from tqdm.auto import tqdm
import numpy as np

# other libraries
import os
import cv2
import pandas as pd


class TurtleDataset(Dataset):
    """
    Custom Dataset for loading turtle images and annotations.
    """

    def __init__(self, path: str, image_factor: int = 1) -> None:
        """
        Args:
            path (str): Path to the dataset directory.
        """
        self.path = path
        self.annotations_filename = f"{path}/annotations.json"
        self.coco = COCO(self.annotations_filename)
        self.n_image_files = len(
            [
                os.path.join(dp, f)
                for dp, _, fn in os.walk(os.path.expanduser(f"{path}/images"))
                for f in fn
            ]
        )
        self.image_factor = image_factor
        metadata = pd.read_csv(f"{path}/metadata_splits.csv")
        self.img_ids = metadata["id"].tolist()
        self.bboxes, self.labels, self.file_names = self.preload()

        print(f"Dataset 'TurtleDataset' initialized with {len(self)} images.")

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in the dataset.
        """
        return self.n_image_files

    def preload(
        self,
    ) -> tuple[dict[int, list[list[float]]], dict[int, list[int]], list[str]]:
        """
        Preloads bounding boxes, labels, and file names from annotations.

        Returns:
            tuple: A tuple containing dictionaries of bounding boxes and labels, and a list of file names.
        """
        cat_ids = self.coco.getCatIds()
        ann_ids = self.coco.getAnnIds(catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        bboxes = {ann["image_id"]: [] for ann in anns}
        labels = {ann["image_id"]: [] for ann in anns}
        file_names = {}

        print("Preloading bboxes, labels, and file names...")
        for ann in tqdm(anns):
            img = self.coco.loadImgs(ann["image_id"])[0]
            file_names[ann["image_id"]] = img["file_name"]
            bboxes[ann["image_id"]].append(ann["bbox"])
            labels[ann["image_id"]].append(ann["category_id"])

        bboxes = {k: np.array(v) for k, v in bboxes.items()}

        return bboxes, labels, file_names

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and a dictionary of masks, bounding boxes, and labels.
        """
        img_id = self.img_ids[index]
        image_path = f"{self.path}/{self.file_names[img_id]}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[index]
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        labels = self.labels[index]

        masks = self.generate_mask(img_id, image)

        # Resize the image, bounding boxes, and masks
        orig_width, orig_height = image.shape[:2]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / self.image_factor
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / self.image_factor

        image.resize(
            (orig_width // self.image_factor, orig_height // self.image_factor)
        )
        masks.resize(
            (
                masks.shape[0],
                orig_width // self.image_factor,
                orig_height // self.image_factor,
            )
        )

        image = transforms.ToTensor()(image)
        masks = torch.tensor(masks, dtype=torch.uint8)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, {
            "masks": masks,
            "boxes": bboxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

    def generate_mask(self, img_id: int, img: np.ndarray) -> np.ndarray:

        cat_ids = self.coco.getCatIds()

        # Load annotations for this image ID
        anns_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        # Initialize an empty mask for this image
        masks = np.zeros((len(anns), img.shape[0], img.shape[1]), dtype=np.uint8)

        # Generate the mask by adding each annotation to this image's mask
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann).astype(np.uint8)
            masks[i] = mask

        return masks


def display_image_with_masks(
    mask_file: str, coco: COCO = None, path: str = "data/turtles-data/data"
) -> None:
    """
    Display an image with its masks.

    Args:
        mask_file (str): Path to the mask file.
    """
    coco = coco or COCO(f"{path}/annotations.json")

    mask = torch.load(mask_file)
    image_id = int(mask_file.split("/")[-2])
    image_path = f"{path}/{coco.loadImgs(image_id)[0]['file_name']}"
    image = cv2.imread(image_path)

    mask = mask.numpy()
    mask = np.where(mask, 0, 255).astype(np.uint8)

    image = np.array(image)
    image = np.where(mask[..., None], image, 0)

    cv2.imshow("Image", image)


def load_and_save_masks(path: str, coco: COCO = None) -> None:
    """
    Load masks from the dataset and save them as images.

    Args:
        path (str): Path to the dataset directory.
    """
    coco = coco or COCO(f"{path}/annotations.json")
    os.makedirs(f"{path}/masks", exist_ok=True)

    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    print("Saving masks...")
    for ann in tqdm(anns):
        os.makedirs(f"{path}/masks/{ann['image_id']}", exist_ok=True)

        mask = coco.annToMask(ann)
        mask = torch.tensor(mask, dtype=torch.uint8)
        torch.save(mask, f"{path}/masks/{ann['image_id']}/{ann['id']}.pt")


def load_data(path: str, hpp_dict: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for training, validation, and test sets.

    Args:
        path (str): Path of the dataset.
        hpp_dict (dict): Dictionary of hyperparameters.

    Returns:
        tuple: Tuple of dataloaders, train, val, and test in respective order.
    """
    # create datasets
    full_dataset = TurtleDataset(f"{path}", hpp_dict["img_factor"])
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [
            int(0.7 * len(full_dataset)),
            int(0.2 * len(full_dataset)),
            len(full_dataset)
            - int(0.7 * len(full_dataset))
            - int(0.2 * len(full_dataset)),
        ],
    )

    collate_fn = lambda x: tuple(zip(*x))

    # define dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hpp_dict["batch_size"],
        shuffle=True,
        num_workers=hpp_dict["num_workers"],
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hpp_dict["batch_size"],
        shuffle=True,
        num_workers=hpp_dict["num_workers"],
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hpp_dict["batch_size"],
        shuffle=True,
        num_workers=hpp_dict["num_workers"],
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader


def find_max_img_size(path: str) -> tuple[int, int]:
    """
    Find the maximum image size in the dataset.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        tuple: A tuple containing the maximum image size.
    """
    max_width = 0
    max_height = 0
    files = os.walk(os.path.expanduser(f"{path}/images"))
    for dp, _, fn in tqdm(files, desc="Finding max image size"):
        for f in fn:
            img = cv2.imread(os.path.join(dp, f))
            max_width = max(max_width, img.size[0])
            max_height = max(max_height, img.size[1])
    return max_width, max_height


if __name__ == "__main__":
    # load_data("data")
    # pass in the path to the dataset
    # a = TurtleDataset("data/turtles-data/data")
    # print(a[3][1]["labels"])

    # load_and_save_masks("data/turtles-data/data")

    # coco = COCO("data/turtles-data/data/annotations.json")
    # for file in os.listdir("data/turtles-data/data/masks/1"):
    #     display_image_with_masks(f"data/turtles-data/data/masks/1/{file}", coco)

    train_dataloader, val_dataloader, test_dataloader = load_data(
        "data/turtles-data/data", {"batch_size": 2, "num_workers": 0}, (2000, 2000)
    )

    print("Data loaded")
    for data, target in train_dataloader:
        for d in target:
            print(d["boxes"])
        break
