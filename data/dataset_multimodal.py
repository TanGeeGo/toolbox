import json
from PIL import Image
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import utils.utils_image as util
from utils.utils_mask import MaskingGenerator
import torchvision.transforms.functional as F

from torchvision.datasets.vision import VisionDataset

# ----------------------------------------
# data augmentation statements
# ----------------------------------------
class Compose(transforms.Compose):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, tgt, interpolation1=None, interpolation2=None):
        for t in self.transforms:
            img, tgt = t(img, tgt, interpolation1=interpolation1, interpolation2=interpolation2)
        return img, tgt

class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.
    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic1, pic2, interpolation1=None, interpolation2=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic1), F.to_tensor(pic2)

class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor, interpolation1=None, interpolation2=None):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor1, self.mean, self.std, self.inplace), F.normalize(tensor2, self.mean, self.std, self.inplace)

class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop a random portion of image and resize it to a given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.
    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
    """

    def __init__(
        self,
        size,
        scale=(0.3, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=F.InterpolationMode.BICUBIC,
        antialias=True
    ):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation, antialias=antialias)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if interpolation1 == 'nearest':
            interpolation1 = F.InterpolationMode.NEAREST
        else:
            interpolation1 = F.InterpolationMode.BICUBIC
        if interpolation2 == 'nearest':
            interpolation2 = F.InterpolationMode.NEAREST
        else:
            interpolation2 = F.InterpolationMode.BICUBIC
            
        return F.resized_crop(img, i, j, h, w, self.size, interpolation1), \
                F.resized_crop(tgt, i, j, h, w, self.size, interpolation2)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(tgt)
        return img, tgt

class RandomApply(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability.
    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:
        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)
        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.
    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p=p)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        if self.p < torch.rand(1):
            return img, tgt
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt

class ColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Input image.
        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img, tgt


class RandomErasing(transforms.RandomErasing):
    """Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    Returns:
        Erased Image.
    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (Tensor): Tensor image to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace), tgt
        return img, tgt


class GaussianBlur(transforms.GaussianBlur):
    """Gaussian blur augmentation"""

    def __init__(self, kernel_size=7, sigma=[.1, 2.]):
        super().__init__(kernel_size=kernel_size, sigma=sigma)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (Tensor): Tensor image to be blurred
        Returns:
            img (Tensor): Blurred Tensor image.    
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), tgt

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}( sigma={self.sigma})"
        return s

# ----------------------------------------
# main dataset statement
# ----------------------------------------
class DatasetMultiModal(VisionDataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # But the image pair maybe within different sets
    # Multiple "paths_L" and "paths_H" are needed.
    # This datasets are state with Path to json annotation file
    # 
    # Following this paper:
    # Images Speak in Images: A Generalist Painter for In-Context Visual Learning 
    # (https://arxiv.org/abs/2212.02499)
    # Github source: https://github.com/baaivision/Painter
    # -----------------------------------------
    # e.g., deblur, denoise, dehaze, deraining ...
    # -----------------------------------------
    '''

    def __init__(self, opt, phase='train'):
        # ------------------------------------
        # root path for VisionDataset
        # ------------------------------------
        root = '/' + opt['json_path_list'][0].split('/')[0]
        super().__init__(root)

        self.opt = opt
        self.pairs = []
        self.weights = []
        type_weight_list = [0.4, 0.15, 0.15, 0.3] # weight of different model datasets (sum to 1)

        # ------------------------------------
        # define the transform
        #
        # definition of the transform of the data-pairs
        # transform_train1: for image to image mapping
        # transform_train2: for instance segmentation mapping
        # transform_train3: for pose to image (or reverse) mapping
        # transform_train_seccrop: second crop the image pairs in probability
        # transform_test: for test in training (validation)
        # ------------------------------------
        self.phase = phase
        self.imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std=torch.tensor([0.229, 0.224, 0.225])
        if self.phase == 'train':
            self.transform_train1 = Compose([
                    RandomResizedCrop(self.opt['H_size'][1], scale=(0.3, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
                    RandomApply([
                        ColorJitter(0.4, 0.4, 0.2, 0.1)
                    ], p=0.8),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
                ])
            self.transform_train2 = Compose([
                    RandomResizedCrop(self.opt['H_size'][1], scale=(0.9999, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
                    ToTensor(),
                    Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
                ])
            self.transform_train3 = Compose([
                    RandomResizedCrop(self.opt['H_size'][1], scale=(0.9999, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
                    ToTensor(),
                    Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
                ])
            self.transform_train_seccrop = Compose([
                    RandomResizedCrop(self.opt['H_size'], scale=(0.3, 1.0), ratio=(0.3, 0.7), interpolation=3, antialias=True),  # 3 is bicubic
                ])
        elif self.phase == 'valid':
            self.transform_val = Compose([
                RandomResizedCrop(self.opt['H_size'][1], scale=(0.9999, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
                ToTensor(),
                Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ])

        # ------------------------------------
        # get the path of multimodal datasets
        # ------------------------------------
        for idx, json_path in enumerate(self.opt['json_path_list']):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            print(json_path, type_weight_list[idx])

        # ------------------------------------
        # if you need one target pairs to guide the mapping
        # ------------------------------------
        self.use_two_pairs = self.opt['use_two_pairs']
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))

        # mask postion generator
        self.window_size = (self.opt['H_size'][0] // self.opt['P_size'], 
                            self.opt['H_size'][1] // self.opt['P_size'])
        self.masked_position_generator = MaskingGenerator(
            self.window_size, num_masking_patches=self.window_size[0]*self.window_size[1] // 2,
            max_num_patches=self.window_size[0]*self.window_size[1] // 4, min_num_patches=16,
        )

        if self.phase == 'train':
            self.half_mask_ratio = opt['half_mask_ratio']
        else:
            self.half_mask_ratio = 1.0

    def _load_image(self, path):
        while True:
            try:
                img = util.imread_PIL(path)
            except OSError as e:
                print(f'Catched exception: {str(e)}. Re-trying...')
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.float32(np.array(img) / 10000.)
            img = img * 255.
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return img
    
    def _decide_interpolations(self, pair_type):
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        
        return interpolation1, interpolation2

    def _decide_transforms(self, phase, pair_type):
        if phase == 'train' and "inst" in pair_type:
            cur_transforms = self.transform_train2
        elif phase == 'train' and "pose" in pair_type:
            cur_transforms = self.transform_train3
        elif phase == 'train':
            cur_transforms = self.transform_train1
        elif phase == 'valid':
            cur_transforms = self.transform_val

        return cur_transforms
    
    def _combine_images(self, image, image2):
        # image under image2
        dst = torch.cat([image, image2], dim=1)
        return dst

    def _get_second_pair(self, image, target, pair_type, cur_transforms, interpolation1, interpolation2):
        # sample the second pair belonging to the same type
        pair2_index = random.choice(self.pair_type_dict[pair_type])
        pair2 = self.pairs[pair2_index]
        image2 = self._load_image(pair2['image_path'])
        target2 = self._load_image(pair2['target_path'])
        assert pair2['type'] == pair_type
        image2, target2 = cur_transforms(image2, target2, interpolation1, interpolation2)
        image = self._combine_images(image, image2)
        target = self._combine_images(target, target2)
        return image, target

    def _get_valid_pixels(self, target, pair_type):
        valid = torch.ones_like(target)
        if "nyuv2_image2depth" in pair_type:
            thres = torch.ones(3) * (1e-3 * 0.1)
            thres = (thres - self.imagenet_mean) / self.imagenet_std
            valid[target < thres[:, None, None]] = 0
        elif "ade20k_image2semantic" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - self.imagenet_mean) / self.imagenet_std
            valid[target < thres[:, None, None]] = 0
        elif "coco_image2panoptic_sem_seg" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - self.imagenet_mean) / self.imagenet_std
            valid[target < thres[:, None, None]] = 0
        elif "image2pose" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - self.imagenet_mean) / self.imagenet_std
            valid[target > thres[:, None, None]] = 10.0
            fg = target > thres[:, None, None]
            if fg.sum() < 100*3:
                valid = valid * 0.
        elif "image2panoptic_inst" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - self.imagenet_mean) / self.imagenet_std
            fg = target > thres[:, None, None]
            if fg.sum() < 100*3:
                valid = valid * 0.
        
        return valid

    def __getitem__(self, index):

        # ------------------------------------
        # get paired image
        # ------------------------------------
        pair = self.pairs[index]

        img_L = self._load_image(pair['image_path'])
        img_H = self._load_image(pair['target_path'])

        # ------------------------------------
        # get L/H patch pair
        # decide the interpolation and the transforms for training
        # ------------------------------------
        interpolation1, interpolation2 = self._decide_interpolations(pair['type'])
        cur_transforms = self._decide_transforms(self.phase, pair['type'])
        
        img_L, img_H = cur_transforms(img_L, img_H, interpolation1, interpolation2)
        
        if self.use_two_pairs:
            # ------------------------------------
            # if need one target pairs to guide the mapping
            # ------------------------------------
            img_L, img_H = self._get_second_pair(img_L, img_H, pair['type'], \
                                                 cur_transforms, interpolation1, interpolation2)
            
        use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        if ("inst" in pair['type']) or ("pose" in pair['type']) or use_half_mask:
            pass
        else:
            img_L, img_H = self.transform_train_seccrop(img_L, img_H, interpolation1, interpolation2)

        # ------------------------------------
        # get the valid pixel in target image
        # ------------------------------------
        valid = self._get_valid_pixels(img_H, pair['type'])

        # ------------------------------------
        # get mask for the data pairs
        # ------------------------------------
        if use_half_mask:
            mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0]//2:, :] = 1
        else:
            mask = self.masked_position_generator()

        return {'L': img_L, 'H': img_H, 'Mask': mask, 'Valid': valid}
    
    def __len__(self) -> int:
        return len(self.pairs)
    