import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from collections import Counter

import torch
import torchvision.transforms as transforms

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    
from snorkel.analysis import Scorer
from snorkel.classification import DictDataset, DictDataLoader, MultitaskClassifier, Operation, Task
from snorkel.classification.data import XDict, YDict
from snorkel.classification.loss import cross_entropy_with_probs


##################################################################################################
########################### TRANSFORMS AND HELPERS FOR DATASET LOADING ###########################
##################################################################################################


class StdNormalize(object):
    """
    Normalize torch tensor to have zero mean and unit std deviation
    """

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.sub(_input.mean()).div(_input.std())
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


def standard_transform(input_size):
    """
    Transforms to apply to train and test
    """
    return transforms.Compose(
        [transforms.Resize(input_size), transforms.ToTensor(), StdNormalize()]
    )


def img_to_array(img, data_format="channels_last", dtype="float32"):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
        
    Reproduced from https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/array_to_img
    """
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: %s" % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError("Unsupported image shape: %s" % (x.shape,))
    return x

def array_to_img(x, data_format="channels_last", scale=True, dtype="float32"):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
        
    Reproduced from https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/array_to_img
    """
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `array_to_img` requires PIL."
        )
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3 (single image). "
            "Got array with shape: %s" % (x.shape,)
        )

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Invalid data_format: %s" % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == "channels_first":
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype("uint8"), "RGBA")
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype("uint8"), "RGB")
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype("uint8"), "L")
    else:
        raise ValueError("Unsupported channel number: %s" % (x.shape[2],))


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, "HAMMING"):
        _PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
    if hasattr(pil_image, "BOX"):
        _PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, "LANCZOS"):
        _PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS
        

def load_img(
    path, grayscale=False, color_mode="rgb", target_size=None, interpolation="nearest"
):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn("grayscale is deprecated. Please use " 'color_mode = "grayscale"')
        color_mode = "grayscale"
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `array_to_img` requires PIL."
        )
    img = pil_image.open(path)
    if color_mode == "grayscale":
        if img.mode != "L":
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation, ", ".join(_PIL_INTERPOLATION_METHODS.keys())
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def default_xray_loader(xray_path, img_rows=224, img_cols=224):
    """
    Function to load X-ray images into 3 channels
    """
    xray = load_img(
        xray_path, color_mode="grayscale", target_size=(img_rows, img_cols, 1)
    )
    xray = img_to_array(xray)
    xray = np.dstack([xray, xray, xray])
    xray = array_to_img(xray, "channels_last")
    return xray


class OpenIDataset(DictDataset):
    """
    Snorkel Pytorch DictDataset for OpenI chest X-ray task
    """
    def __init__(
        self,
        name: str,
        split: str,
        paths: list,
        labels: list,
        ref=None,
        transform=None, #transforms.transforms.Compose,
        loader=default_xray_loader,
        image_size=224,
    ) -> None:
        self.transform = transform
        self.ref = ref
        self.loader = loader
        self.paths = paths
        
        X_dict = {
            "paths": paths,
        }
        Y_dict = {
            "openi_task": torch.tensor(labels)
        }
        super(OpenIDataset, self).__init__(name, split, X_dict, Y_dict)

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        
        # This snippet ignores 
        img_fn = self.X_dict["paths"][index]
        idx = 0
        if self.ref is not None and isinstance(self.paths[index], list):
            for i in range(len(img_fn)):
                impath = img_fn[i]
                if impath in self.ref:
                    idx = i
                    break
        else:
            impath = img_fn

        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
            
        x_dict = {"xray": img}
        y_dict =  {name: label[index] for name, label in self.Y_dict.items()}

        return x_dict, y_dict

    def __len__(self):
        return len(self.X_dict["paths"])
    
    
def get_data_loader(paths, labels, split=None, batch_size=32, input_size=224, shuffle=False):
    """
    Helper that builds dataloader for OpenI chest X-ray task
    """
    # Load front image index
    fin = open("./data/front_view_ids.txt", "r")
    front_view_ids = [_.strip() for _ in fin]
    fin.close()

    split = 'valid' if split == 'dev' else split
    
    dataset = OpenIDataset(
        name=f"{split}", paths=paths, labels=labels, transform=standard_transform(input_size), ref=front_view_ids, split=split,
    )

    # Build data loader
    data_loader = DictDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    
    return data_loader
    
    
##################################################################################################
########################### HELPERS FOR MODEL DEFINITION ###########################
##################################################################################################


def init_fc(fc):
    """
    Initializes FC layer
    """
    torch.nn.init.xavier_uniform_(fc.weight)
    fc.bias.data.fill_(0.01)
    
class SqueezeModule(torch.nn.Module):
    """
    Squeezes input -- for use with pretrained models
    """
    def forward(self, x):
        return x.squeeze()

def create_model(resnet_cnn, num_classes):
    """
    Creates CNN model for OpenI chest X-ray task
    """
    # define input features
    in_features = resnet_cnn.fc.in_features
    feature_extractor = torch.nn.Sequential(*list(resnet_cnn.children())[:-1])
    fc = torch.nn.Linear(in_features, num_classes)
    init_fc(fc)
    
    squeeze_module = SqueezeModule()

    # define layers
    module_pool = torch.nn.ModuleDict(
        {
            "feature_extractor": feature_extractor,
            "squeeze_module": squeeze_module,
            "prediction_head": fc,
        }
    )

    # define task flow through modules
    op_sequence = [        
     # define the feature extraction operation
        Operation(
            name="feat_op",
            module_name="feature_extractor",
            inputs=[("_input_","xray")],
        ),
        
        Operation(
            name="squeeze_op",
            module_name="squeeze_module",
            inputs=["feat_op"]
        
        ),

        # define the prediction operation
        Operation(
            name="head_op", 
            module_name="prediction_head", 
            inputs=["squeeze_op"]
        )       
    ]
    
    # Define the task
    pred_cls_task = Task(
        name="openi_task",
        module_pool=module_pool,
        op_sequence=op_sequence,
        scorer=Scorer(metrics=['accuracy','precision', 'recall', 'f1','roc_auc']),
        loss_func=cross_entropy_with_probs
    )
    
    return MultitaskClassifier([pred_cls_task])


##################################################################################################
############################### OLD UTILITIES -- HOW MUCH TO KEEP? ###############################
##################################################################################################


def view_label_matrix(L, colorbar=True):
    """Display an [n, m] matrix of labels"""
    L = L.todense() if sparse.issparse(L) else L
    plt.imshow(L, aspect="auto")
    plt.title("Label Matrix")
    if colorbar:
        labels = sorted(np.unique(np.asarray(L).reshape(-1, 1).squeeze()))
        boundaries = np.array(labels + [max(labels) + 1]) - 0.5
        plt.colorbar(boundaries=boundaries, ticks=labels)
    plt.show()


def view_overlaps(L, self_overlaps=False, normalize=True, colorbar=True):
    """Display an [m, m] matrix of overlaps"""
    L = L.todense() if sparse.issparse(L) else L
    G = _get_overlaps_matrix(L, normalize=normalize)
    if not self_overlaps:
        np.fill_diagonal(G, 0)  # Zero out self-overlaps
    plt.imshow(G, aspect="auto")
    plt.title("Overlaps")
    if colorbar:
        plt.colorbar()
    plt.show()


def view_conflicts(L, normalize=True, colorbar=True):
    """Display an [m, m] matrix of conflicts"""
    L = L.todense() if sparse.issparse(L) else L
    C = _get_conflicts_matrix(L, normalize=normalize)
    plt.imshow(C, aspect="auto")
    plt.title("Conflicts")
    if colorbar:
        plt.colorbar()
    plt.show()


def _get_overlaps_matrix(L, normalize=True):
    n, m = L.shape
    X = np.where(L != -1, 1, 0).T
    G = X @ X.T

    if normalize:
        G = G / n
    return G


def _get_conflicts_matrix(L, normalize=True):
    n, m = L.shape
    C = np.zeros((m, m))

    # Iterate over the pairs of LFs
    for i in range(m):
        for j in range(m):
            # Get the overlapping non-zero indices
            overlaps = list(
                set(np.where(L[:, i] != -1)[0]).intersection(np.where(L[:, j] != -1)[0])
            )
            C[i, j] = np.where(L[overlaps, i] != L[overlaps, j], 1, 0).sum()

    if normalize:
        C = C / n
    return C

def plot_probabilities_histogram(Y_p, title=None):
    """Plot a histogram from a numpy array of probabilities
    Args:
        Y_p: An [n] or [n, 1] np.ndarray of probabilities (floats in [0,1])
    """
    if Y_p.ndim > 1:
        msg = (
            f"Arg Y_p should be a 1-dimensional np.ndarray, not of shape "
            f"{Y_p.shape}."
        )
        raise ValueError(msg)
    plt.hist(Y_p, bins=20)
    plt.xlim((0, 1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")
    if isinstance(title, str):
        plt.title(title)
    plt.show()
    
def plot_predictions_histogram(Y_ph, Y, title=None):
    """Plot a histogram comparing int predictions vs true labels by class
    Args:
        Y_ph: An [n] or [n, 1] np.ndarray of predicted int labels
        Y: An [n] or [n, 1] np.ndarray of gold labels
    """
    labels = list(set(Y).union(set(Y_ph)))
    edges = [x - 0.5 for x in range(min(labels), max(labels) + 2)]

    plt.hist([Y_ph, Y], bins=edges, label=["Predicted", "Gold"])
    ax = plt.gca()
    ax.set_xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("# Predictions")
    plt.legend(loc="best")
    if isinstance(title, str):
        plt.title(title)
    plt.show()
    
##################################################################################################
######################################## HELPERS FOR LUDWIG ######################################
##################################################################################################


def indices_to_one_hot(data):
    """Convert an iterable of indices to one-hot encoded labels."""
    nb_classes = len(np.unique(data))
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def get_ludwig_ap_paths(data):
    
    fin = open("./data/front_view_ids.txt", "r")
    front_view_ids = [_.strip() for _ in fin]
    fin.close()
    
    ref = front_view_ids
    
    impaths = []
    for paths in data['xray_paths'].tolist():
        if isinstance(paths, list):
                for i in range(len(paths)):
                    impath = paths[i]
                    if impath in ref:
                        idx = i
                        break
        else:
            impath = paths
            
        impaths.append(f"../../../cross_modal_ws/{impath}")
            
    return impaths
