import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from snorkel.classification.data import DictDataset, XDict, YDict


def union(bbox1, bbox2):
    """Create the union of the two bboxes.

    Parameters
    ----------
    bbox1
        Coordinates of first bounding box
    bbox2
        Coordinates of second bounding box

    Returns
    -------
    [y0, y1, x0, x1]
        Coordinates of union of input bounding boxes

    """
    y0 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x0 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[3], bbox2[3])
    return [y0, y1, x0, x1]


def crop_img_arr(img_arr, bbox):
    """Crop bounding box from image.

    Parameters
    ----------
    img_arr
        Image in array format
    bbox
        Coordinates of bounding box to crop

    Returns
    -------
    img_arr
        Cropped image

    """
    return img_arr[bbox[0] : bbox[1], bbox[2] : bbox[3], :]


class SceneGraphDataset(DictDataset):
    """Dataloader for Scene Graph Dataset."""

    def __init__(
        self,
        name: str,
        split: str,
        image_dir: str,
        df: pandas.DataFrame,
        image_size=224,
    ) -> None:
        self.image_dir = Path(image_dir)
        X_dict = {
            "img_fn": df["source_img"].tolist(),
            "obj_bbox": df["object_bbox"].tolist(),
            "sub_bbox": df["subject_bbox"].tolist(),
            "obj_category": df["object_category"].tolist(),
            "sub_category": df["subject_category"].tolist(),
        }
        Y_dict = {
            "scene_graph_task": torch.LongTensor(df["label"].to_numpy())
        }  # change to take in the rounded train labels
        super(SceneGraphDataset, self).__init__(name, split, X_dict, Y_dict)

        # define standard set of transformations to apply to each image
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        img_fn = self.X_dict["img_fn"][index]
        img_arr = np.array(Image.open(self.image_dir / img_fn))

        obj_bbox = self.X_dict["obj_bbox"][index]
        sub_bbox = self.X_dict["sub_bbox"][index]
        obj_category = self.X_dict["obj_category"][index]
        sub_category = self.X_dict["sub_category"][index]

        # compute crops
        obj_crop = crop_img_arr(img_arr, obj_bbox)
        sub_crop = crop_img_arr(img_arr, sub_bbox)
        union_crop = crop_img_arr(img_arr, union(obj_bbox, sub_bbox))

        # transform each crop
        x_dict = {
            "obj_crop": self.transform(Image.fromarray(obj_crop)),
            "sub_crop": self.transform(Image.fromarray(sub_crop)),
            "union_crop": self.transform(Image.fromarray(union_crop)),
            "obj_category": obj_category,
            "sub_category": sub_category,
        }

        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        return len(self.X_dict["img_fn"])


class WordEmb(nn.Module):
    """Extract and concat word embeddings for obj and sub categories."""

    def __init__(self, glove_fn="data/glove/glove.6B.100d.txt"):
        super(WordEmb, self).__init__()

        self.word_embs = pandas.read_csv(
            glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        )

    def _get_wordvec(self, word):
        return self.word_embs.loc[word].as_matrix()

    def forward(self, obj_category, sub_category):
        obj_emb = self._get_wordvec(obj_category)
        sub_emb = self._get_wordvec(sub_category)
        embs = np.concatenate([obj_emb, sub_emb], axis=1)
        return torch.FloatTensor(embs)


# Classes and helper functions for defining classifier
def init_fc(fc):
    torch.nn.init.xavier_uniform_(fc.weight)
    fc.bias.data.fill_(0.01)


class FlatConcat(nn.Module):
    """Module that flattens and concatenates features"""

    def forward(self, *inputs):
        return torch.cat([input.view(input.size(0), -1) for input in inputs], dim=1)
