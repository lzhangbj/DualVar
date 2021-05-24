import random
import numbers
import math
import torch
import collections
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import ImageOps, Image, ImageFilter
import numpy as np
from joblib import Parallel, delayed

############# mini tools
def _blend_np(img1, img2, ratio):
    dtype = img1.dtype
    img1 = img1.astype(np.float32)
    return np.rint((img1 * ratio + img2 * (1 - ratio))).clip(0, 255).astype(dtype)

def _rgb_to_grayscale_np(img):
    assert img.shape[2] == 3
    dtype = img.dtype
    l_img = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(dtype)
    l_img = l_img[..., np.newaxis]
    return l_img

def _rgb2hsv_np(img):
    # implemented from torch
    r, g, b = np.split(img, 3, axis=2)
    r, g, b = r.squeeze(), g.squeeze(), b.squeeze()

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = img.max(2)
    minc = img.min(2)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = np.ones_like(maxc)
    s = cr / np.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = np.where(eqc, ones, cr)

    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = np.fmod((h / 6.0 + 1.0), 1.0)
    return np.stack((h, s, maxc), axis=2)

def _hsv2rgb_np(img):
    h, s, v = np.split(img, 3, axis=2)
    h, s, v = h.squeeze(), s.squeeze(), v.squeeze()
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.astype(np.int32)

    p = np.clip((v * (1.0 - s)), 0.0, 1.0)
    q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
    t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = (i[np.newaxis, ...] == np.arange(6).reshape((-1, 1, 1))).astype(img.dtype)

    a1 = np.stack((v, q, p, p, t, v), axis=0)  # (6, h, w)
    a2 = np.stack((t, v, v, q, p, p), axis=0)
    a3 = np.stack((p, p, t, v, v, q), axis=0)
    a4 = np.stack((a1, a2, a3), axis=0)  # (3, 6, h, w)

    res = np.einsum("...ijk, ...xijk -> ...xjk", mask, a4)
    res = res.transpose((1, 2, 0))
    return res

def adjust_brightness_np(img, ratio):
    assert img.shape[2] == 3
    return _blend_np(img, 0, ratio)

def adjust_contrast_np(img, factor):
    orig_dtype = img.dtype
    mean = np.mean(_rgb_to_grayscale_np(img).astype(np.float32), axis=(-3, -2, -1), keepdims=True)
    return _blend_np(img, mean, factor)

def adjust_hue_np(img, factor):
    orig_dtype = img.dtype
    img = img.astype(np.float32) / 255.

    img = _rgb2hsv_np(img)
    h, s, v = np.split(img, 3, axis=2)
    h = (h + factor) % 1.0
    img = np.concatenate((h, s, v), axis=2)
    img_hue_adj = _hsv2rgb_np(img)

    img_hue_adj = (img_hue_adj * 255.).astype(orig_dtype)

    return img_hue_adj

def adjust_saturation_np(img, factor):
    return _blend_np(img, _rgb_to_grayscale_np(img), factor)


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)


class Scale:
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class RandomCrop:
    def __init__(self, size, n_seqblock=0):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
        self.n_seqblock = n_seqblock

    def __call__(self, imgmap):
        img1 = imgmap[0]
        h, w = img1.size[0], img1.size[1]
        assert h >= self.size[0] and w >= self.size[1]

        if self.n_seqblock == 0:
            h_start = random.randint(0, h-self.size[0])
            w_start = random.randint(0, w-self.size[1])

            return [i.crop((h_start, w_start, h_start+self.size[0], w_start+self.size[1])) for i in imgmap]
        ret = []
        for i in range(len(imgmap)):
            if i%self.n_seqblock == 0:
                h_start = random.randint(0, h - self.size[0])
                w_start = random.randint(0, w - self.size[1])
            ret.append(imgmap[i].crop((h_start, w_start, h_start+self.size[0], w_start+self.size[1])))
        return ret


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class FiveCrop:
    def __init__(self, size, where=1):
        # 1=topleft, 2=topright, 3=botleft, 4=botright, 5=center
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.where = where

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        if (th > h) or (tw > w):
            raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size, (h,w)))
        if self.where == 1:
            return [i.crop((0, 0, tw, th)) for i in imgmap]
        elif self.where == 2:
            return [i.crop((w-tw, 0, w, th)) for i in imgmap]
        elif self.where == 3:
            return [i.crop((0, h-th, tw, h)) for i in imgmap]
        elif self.where == 4:
            return [i.crop((w-tw, h-tw, w, h)) for i in imgmap]
        elif self.where == 5:
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


# class RandomSizedCrop:
#     def __init__(self, size, interpolation=Image.BICUBIC, n_seqblock=0, p=1.0, seq_len=16, bottom_area=0.2):
#         self.size = size
#         self.interpolation = interpolation
#         self.threshold = p
#         self.seq_len = seq_len
#         self.n_seqblock = n_seqblock if n_seqblock!=0 else seq_len
#         self.bottom_area = bottom_area
#
#     def __call__(self, imgmap):
#         img1 = imgmap[0]
#         if random.random() < self.threshold: # do RandomSizedCrop
#             for attempt in range(10):
#                 area = img1.size[0] * img1.size[1]
#
#                 result = []
#                 for idx, i in enumerate(imgmap):
#                     if idx % self.n_seqblock == 0:
#                         target_area = random.uniform(self.bottom_area, 1) * area
#                         w = min(int(round(math.sqrt(target_area))), img1.size[0])
#                         h = min(int(round(math.sqrt(target_area))), img1.size[1])
#                         x1 = random.randint(0, img1.size[0] - w)
#                         y1 = random.randint(0, img1.size[1] - h)
#                     result.append(i.crop((x1, y1, x1 + w, y1 + h)))
#                     assert(result[-1].size == (w, h))
#
#                 assert len(result) == len(imgmap)
#                 return [i.resize((self.size, self.size), self.interpolation) for i in result]
#
#             # Fallback
#             scale = Scale(self.size, interpolation=self.interpolation)
#             crop = CenterCrop(self.size)
#             return crop(scale(imgmap))
#         else: #don't do RandomSizedCrop, do CenterCrop
#             crop = CenterCrop(self.size)
#             return crop(imgmap)


class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR, consistent=True, p=1.0):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold: # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert(i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert(result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result]

            # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else: # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)

class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None, seq_len=0):
        self.consistent = consistent
        if seq_len != 0:
            self.consistent = False
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5
        self.seq_len = seq_len
    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for idx, i in enumerate(imgmap):
                if idx % self.seq_len == 0: th = random.random()
                if th < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0):
        self.consistent = consistent
        self.degree = degree
        self.threshold = p

    def __call__(self, imgmap):
        if random.random() < self.threshold: # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [i.rotate(np.random.randint(-self.degree, self.degree, 1)[0], expand=True) for i in imgmap]
        else: # don't do RandomRotation, do nothing
            return imgmap


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class ToPIL:
    def __call__(self, imgmap):
        topil = transforms.ToPILImage()
        return [topil(i) for i in imgmap]


class RandomGray:
    '''torch tensor versionActually it is a channel splitting, not strictly grayscale images'''

    def __init__(self, consistent=True, p=0.8, seq_len=16, block=1):
        self.consistent = consistent
        self.p = p  # prob to grayscale
        self.seq_len = seq_len
        self.block = block

    def __call__(self, imgmap):
        assert torch.is_tensor(imgmap[0])
        _, height, width = imgmap[0].size()
        height_unit = height // self.block
        width_unit = width // self.block

        result = []
        for idx, img in enumerate(imgmap):
            if not self.consistent or idx % self.seq_len == 0:
                block_channel = [ random.randint(0,2) if np.random.uniform(0.,1.) < self.p else -1
                                    for _ in range(self.block*self.block)]

            patches = []
            w_patches = []

            for block_ind in range(self.block*self.block):
                channel = block_channel[block_ind]

                height_ind = block_ind // self.block
                width_ind = block_ind % self.block

                height_start = height_unit * height_ind
                height_end = height_start + height_unit if height_ind < self.block - 1 else height

                width_start = width_unit * width_ind
                width_end = width_start + width_unit if width_ind < self.block - 1 else width
                if channel == -1:
                    transformed_block = img[:, height_start:height_end, width_start:width_end]
                else:
                    transformed_block = self.grayscale(img[:, height_start:height_end, width_start:width_end], channel)

                w_patches.append(transformed_block)
                if width_ind == self.block -1:
                    patches.append(torch.cat(w_patches, dim=2))
                    w_patches.clear()

            img = torch.cat(patches, dim=1)

            result.append(img)

        return result

    def grayscale(self, img, channel):
        img = img[channel, :, :].unsqueeze(0).repeat(3, 1, 1)
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
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
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=0.8, block=1, seq_len=16, grad_consistent=False, n_seqblock=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.block = block
        assert not (consistent and grad_consistent)
        self.consistent = consistent
        self.grad_consistent = grad_consistent
        self.threshold = p
        self.seq_len = seq_len
        self.n_seqblock = n_seqblock
        if n_seqblock == 0:
            self.n_seqblock = seq_len
        assert seq_len % self.n_seqblock == 0

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def get_grad_consistent_factors(self):
        brightness_factor_start, brightness_factor_end = random.uniform(*self.brightness), random.uniform(*self.brightness)
        brightness_factors = np.linspace(brightness_factor_start, brightness_factor_end, self.seq_len)

        contrast_factor_start, contrast_factor_end = random.uniform(*self.contrast), random.uniform(*self.contrast)
        contrast_factors = np.linspace(contrast_factor_start, contrast_factor_end, self.seq_len)

        saturation_factor_start, saturation_factor_end = random.uniform(*self.saturation), random.uniform(*self.saturation)
        saturation_factors = np.linspace(saturation_factor_start, saturation_factor_end, self.seq_len)

        hue_factor_start, hue_factor_end = random.uniform(*self.hue), random.uniform(*self.hue)
        hue_factors = np.linspace(hue_factor_start, hue_factor_end, self.seq_len)

        return np.stack([brightness_factors, contrast_factors, saturation_factors, hue_factors], axis=1)

    @staticmethod
    def get_params_fixed(brightness_factor, contrast_factor, saturation_factor, hue_factor, shuffle_indices):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        shuffled_transforms = []
        for i in shuffle_indices:
            shuffled_transforms.append(transforms[i])
        transform = torchvision.transforms.Compose(shuffled_transforms)

        return transform

    @staticmethod
    def get_params_np(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(lambda img: adjust_brightness_np(img, brightness_factor))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(lambda img: adjust_contrast_np(img, contrast_factor))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(lambda img: adjust_saturation_np(img, saturation_factor))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(lambda img: adjust_hue_np(img, hue_factor))

        random.shuffle(transforms)
        def _forward_function(img):
            for transform in transforms:
                img = transform(img)
            return img

        return _forward_function


    def __call__(self, imgmap):
        assert torch.is_tensor(imgmap[0])
        _, height, width = imgmap[0].size()
        height_unit = height // self.block
        width_unit = width // self.block

        result = []
        for idx, img in enumerate(imgmap):
            if not self.grad_consistent:
                if not self.consistent or idx % self.n_seqblock == 0:
                        block_transforms = [self.get_params(self.brightness, self.contrast, self.saturation, self.hue) if np.random.uniform(0.,1.) < self.threshold else lambda x:x
                                        for _ in range(self.block*self.block)]

                img_patches = []
                w_patches = []
                for block_ind in range(self.block*self.block):
                    transform = block_transforms[block_ind]

                    height_ind = block_ind // self.block
                    width_ind = block_ind % self.block

                    height_start = height_unit * height_ind
                    height_end = height_start + height_unit if height_ind < self.block - 1 else height

                    width_start = width_unit * width_ind
                    width_end = width_start + width_unit if width_ind < self.block - 1 else width

                    w_patches.append(transform(img[:, height_start:height_end, width_start:width_end]))
                    if width_ind == self.block-1:
                        img_patches.append(torch.cat(w_patches, dim=2))
                        w_patches.clear()
                img = torch.cat(img_patches, dim=1)
                result.append(img)
            else:
                if idx % self.seq_len == 0:
                    block_transforms = []
                    for _ in range(self.block*self.block):
                        block_factors = self.get_grad_consistent_factors()
                        block_shuffle_indices = [0,1,2,3]
                        random.shuffle(block_shuffle_indices)
                        block_seq_transforms = [self.get_params_fixed(*block_factors[i], block_shuffle_indices) for i in range(self.seq_len)]
                        block_transforms.append(block_seq_transforms)

                frame_ind_in_seq = idx % self.seq_len
                img_patches = []
                w_patches = []
                for block_ind in range(self.block * self.block):
                    transform = block_transforms[block_ind][frame_ind_in_seq]

                    height_ind = block_ind // self.block
                    width_ind = block_ind % self.block

                    height_start = height_unit * height_ind
                    height_end = height_start + height_unit if height_ind < self.block - 1 else height

                    width_start = width_unit * width_ind
                    width_end = width_start + width_unit if width_ind < self.block - 1 else width

                    w_patches.append(transform(img[:, height_start:height_end, width_start:width_end]))
                    if width_ind == self.block - 1:
                        img_patches.append(torch.cat(w_patches, dim=2))
                        w_patches.clear()
                img = torch.cat(img_patches, dim=1)
                result.append(img)

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class ChannelMask(object):
    def __init__(self, consistent=False, block=1, p=0.75, seq_len=16):
        self.consistent = consistent
        self.block = block
        self.p = p
        self.seq_len = seq_len
        assert self.seq_len == 16

    def __call__(self, imgmap):
        assert torch.is_tensor(imgmap[0])
        _, height, width = imgmap[0].size()
        height_unit = height // self.block
        width_unit = width // self.block

        result = []

        for idx, img in enumerate(imgmap):
            if not self.consistent or idx % self.seq_len == 0: # re-init blockwise randfloats for each sequence or each frame
                randfloats = np.random.uniform(0., 1., size=self.block*self.block)

            for block_ind in range(self.block*self.block):
                randfloat = randfloats[block_ind]

                height_ind = block_ind // self.block
                width_ind = block_ind % self.block

                height_start = height_unit*height_ind
                height_end = height_start + height_unit if height_ind<self.block-1 else height

                width_start = width_unit*width_ind
                width_end = width_start + width_unit if width_ind < self.block-1 else width

                if randfloat < 1-self.p: # no mask
                    continue
                else: # mask r g b
                    rgb_ind = int(randfloat*100) // int(100*(self.p/3.)) - 1
                    img[rgb_ind, height_start:height_end, width_start:width_end] = 0

            result.append(img)

        return result


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.], seq_len=16, n_seqblock=0):
        self.sigma = sigma
        self.seq_len = seq_len
        self.n_seqblock = n_seqblock if n_seqblock != 0 else seq_len

    def __call__(self, imgmap):
        assert torch.is_tensor(imgmap[0])
        result = []
        for idx, img in enumerate(imgmap):
            if idx % self.n_seqblock == 0:
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = transforms.ToPILImage()(img)
            result.append(transforms.ToTensor()(img.filter(ImageFilter.GaussianBlur(radius=sigma))))
        return result


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class TwoClipTransform:
    """Take two random transforms on two clips"""
    def __init__(self, base_transform, null_transform, seq_len, p=0.3):
        # p = probability to use base_transform
        self.base = base_transform
        self.null = null_transform
        self.p = p
        self.seq_len = seq_len # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.seq_len

        if random.random() < self.p:
            tr1 = self.base
        else:
            tr1 = self.null

        if random.random() < self.p:
            tr2 = self.base
        else:
            tr2 = self.null

        q = tr1(x[0:self.seq_len])
        k = tr2(x[self.seq_len::])
        return q + k


class MultipleClipTransform:
    """Take two random transforms on two clips"""
    def __init__(self, transform_list, seq_len):
        # p = probability to use base_transform
        self.transforms = transform_list
        self.num_transform = len(transform_list)
        self.seq_len = seq_len # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image

        assert len(x) == self.num_transform * self.seq_len, (len(x), self.num_transform, self.seq_len)
        # assert self.num_transform == 3

        tsfmed_x = []
        for i in range(self.num_transform):
            tsfmed_x.extend(self.transforms[i](
                x[self.seq_len*i:self.seq_len*(i+1)])
            )
        return tsfmed_x

class MultiRandomizedTransform:
    """apply probabilistic transforms for each clip of input x"""
    def __init__(self, transform_list, seq_len, weights=None):
        # p = probability to use base_transform
        self.transforms = transform_list
        self.num_transform = len(transform_list)
        self.seq_len = seq_len # channel to split the tensor into two
        assert weights is not None
        self.weights = []
        for weight in weights:
            self.weights.append(np.cumsum(weight))
            assert self.weights[-1][-1] == 1.

    def __call__(self, x):
        # target: list of image
        assert len(x) % self.seq_len == 0
        num_seqs = len(x) // self.seq_len
        # assert self.num_transform == 3
        assert num_seqs == len(self.weights)

        tsfmed_x = []
        for i in range(num_seqs):
            rand_p = np.random.uniform()
            tsfm_ind = 0
            while rand_p >= self.weights[i][tsfm_ind]: tsfm_ind+=1
            tsfmed_x.extend(self.transforms[tsfm_ind](
                x[self.seq_len*i:self.seq_len*(i+1)])
            )
        return tsfmed_x


class RandomizedTransform:
    """apply probabilistic transforms for each clip of input x"""
    def __init__(self, transform_list, seq_len, weights=None):
        # p = probability to use base_transform
        self.transforms = transform_list
        self.num_transform = len(transform_list)
        self.seq_len = seq_len # channel to split the tensor into two
        self.weights = np.ones(self.num_transform) / self.num_transform if weights is None else weights
        self.weights = np.cumsum(self.weights)
        assert self.weights[-1] == 1.
        assert len(weights) == self.num_transform

    def __call__(self, x):
        # target: list of image
        assert len(x) % self.seq_len == 0
        num_seqs = len(x) // self.seq_len
        # assert self.num_transform == 3

        tsfmed_x = []
        for i in range(num_seqs):
            rand_p = np.random.uniform()
            tsfm_ind = 0
            while rand_p >= self.weights[tsfm_ind]: tsfm_ind+=1
            tsfmed_x.extend(self.transforms[tsfm_ind](
                x[self.seq_len*i:self.seq_len*(i+1)])
            )
        return tsfmed_x


class OneClipTransform:
    """Take two random transforms on one clips"""
    def __init__(self, base_transform, null_transform, seq_len):
        self.base = base_transform
        self.null = null_transform
        self.seq_len = seq_len # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.seq_len

        if random.random() < 0.5:
            tr1, tr2 = self.base, self.null
        else:
            tr1, tr2 = self.null, self.base

        # randomly abandon half
        if random.random() < 0.5:
            xx = x[0:self.seq_len]
        else:
            xx = x[self.seq_len::]

        q = tr1(xx)
        k = tr2(xx)
        return q + k


class TransformController:
    def __init__(self, transform_list, weights):
        self.transform_list = transform_list
        self.weights = weights
        self.num_transform = len(transform_list)
        assert self.num_transform == len(self.weights)

    def __call__(self, x):
        idx = random.choices(range(self.num_transform), weights=self.weights)[0]
        return self.transform_list[idx](x)

    def __str__(self):
        string = 'TransformController: %s with weights: %s' % (str(self.transform_list), str(self.weights))
        return string



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

