from torchvision.datasets import VisionDataset
from PIL import Image
import glob
import os
from typing import Callable, Optional, List


class MVTecADDataset(VisionDataset):
    """`MVTecAD <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ Dataset.

    Args:
        root (str): path to data directory contains MVTecAD
        target (str): target object. e.g. bottle, cable, capsule.
        train (bool): choise of train data or test data.
        transforms (Optional[Callable], optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
        mask_transforms (Optional[Callable], optional): A function/transform that takes in the mask and transforms it.
        img_ext (str, optional): image file extension that are to be loaded. Defaults to ".png".

    Raises:
        FileNotFoundError: [description]
    """

    def __init__(
        self,
        root: str,
        target: str,
        train: bool,
        transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        img_ext: str = ".png",
    ):
        self.root = root
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.target = target

        target_list = self._get_target_list()
        if target not in target_list:
            raise FileNotFoundError(f"`{target}' not found in `{root}'")

        if train:
            target_root = os.path.join(root, target, "train")
        else:
            target_root = os.path.join(root, target, "test")

        self.labels = self._get_labels(target_root)
        self.label_num_map = dict(
            zip(self.labels, range(len(self.labels))))

        self.img_path_list = glob.glob(
            os.path.join(target_root, f"*/*{img_ext}"))
        self.label_list = [self._get_label_from_path(path)
                           for path in self.img_path_list]
        self.mask_path_list = [self._get_mask_path(
            path, img_ext) for path in self.img_path_list]

    def __getitem__(self, idx):
        img = self._load_image(idx)
        mask = self._load_mask(idx, img)
        label = self.label_num_map[self.label_list[idx]]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        return img, mask, label

    def _load_image(self, idx: int) -> Image.Image:
        return Image.open(self.img_path_list[idx])

    def _load_mask(self, idx: int, img: Image.Image) -> Image.Image:
        path = self.mask_path_list[idx]
        label = self.label_list[idx]

        if not os.path.exists(path):
            if label == "good":
                # generate blank mask for `good'
                mask = Image.new('L', img.size)
            else:
                raise Exception(f"{path} not found")
        else:
            mask = Image.open(path)

        return mask

    def __len__(self) -> int:
        return len(self.img_path_list)

    def _get_target_list(self) -> List[str]:
        return os.listdir(self.root)

    def _get_labels(self, target_root: str) -> List[str]:
        labels = os.listdir(target_root)
        # append `good' at beggning of class list to give it label 0
        labels.pop(labels.index("good"))
        labels = ["good"] + labels
        return labels

    def _get_label_from_path(self, img_path: str) -> str:
        img_path_elems = img_path.split(os.sep)
        return img_path_elems[-2]

    def _get_mask_path(self, img_path: str, img_ext: str) -> str:
        img_path_elems = img_path.split(os.sep)
        img_path_elems[-3] = "ground_truth"
        img_path_elems[-1] = img_path_elems[-1].split(img_ext)[0]
        img_path_elems[-1] = f"{img_path_elems[-1]}_mask{img_ext}"
        return os.sep.join(img_path_elems)
