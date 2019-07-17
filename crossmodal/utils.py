import os
import numpy as np
import torch
import torchvision.transforms as transforms

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def load_ids(filename):
    fin = open(filename, "r")
    return [_.strip() for _ in fin]

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


def transform(input_size):
    return transforms.Compose(
        [transforms.Resize(input_size), transforms.ToTensor(), StdNormalize()]
    )

# default xray loader from png
def default_xray_loader(xray_path, img_rows=224, img_cols=224):
    xray = load_img(
        xray_path, color_mode="grayscale", target_size=(img_rows, img_cols, 1)
    )
    xray = img_to_array(xray)
    xray = np.dstack([xray, xray, xray])
    xray = array_to_img(xray, "channels_last")
    return xray


class CXRFileList(torch.utils.data.Dataset):
    def __init__(self, paths, label=None, transform=None, loader=default_xray_loader, ref=None, lfs=None, slice_mode=None, get_slice_labels=False):
        self.paths = paths
        self.label = label
        self.transform = transform
        self.loader = loader
        self.ref = ref
        # Note: slice_labels and labels in same order!
        if lfs is not None:
            self.lfs = torch.from_numpy(np.array(lfs).astype(np.float32))

    def __getitem__(self, index):
        idx = 0
        if self.ref is not None and isinstance(self.paths[index],list):
            for i in range(len(self.paths[index])):
                impath = self.paths[index][i]
                if impath in self.ref:
                    idx=i
                    break
        else:
            impath = self.paths[index]
        y = self.label[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.paths)
    
def get_data_loader(
    paths, 
    labels, 
    batch_size=32, 
    input_size=224,
    shuffle=False,
):
    # Load front image index
    fin=open('./data/front_view_ids.txt', "r")
    front_view_ids = [_.strip() for _ in fin]
    fin.close()

    dataset = CXRFileList(
            paths=paths,
            label=labels,
            transform=transform(input_size),
            ref=front_view_ids,
        )

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
    return data_loader 

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

