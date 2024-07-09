import numpy as np
from PIL import Image, ImageChops, ImageOps
import copy
from typing import Any
from dataclasses import dataclass
from modules.images import resize_image
from modules import shared, errors, masking
from modules.processing import apply_overlay
from .model import LamaInpaint


def crop(image: Image.Image, origMask: Image.Image, padding: int):
    crop_region = masking.get_crop_region(origMask, padding)
    crop_region = masking.expand_crop_region(crop_region, 1, 1, origMask.width, origMask.height)
    return image.crop(crop_region)

def uncrop(image: Image.Image, origImage: Image.Image, origMask: Image.Image, padding: int):
    crop_region = masking.get_crop_region(origMask, padding)
    crop_region = masking.expand_crop_region(crop_region, 1, 1, origMask.width, origMask.height)
    x1, y1, x2, y2 = crop_region
    paste_to = (x1, y1, x2-x1, y2-y1)
    image_masked = Image.new('RGBa', (origImage.width, origImage.height))
    image_masked.paste(origImage.convert("RGBA").convert("RGBa"), mask=ImageChops.invert(origMask))
    overlay_image = image_masked.convert('RGBA')
    return apply_overlay(image, paste_to, overlay_image)[0]




g_cn_HWC3 = None
def convertIntoCNMaskedImageFormat(image, mask):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        try:
            from annotator.util import HWC3
            g_cn_HWC3 = HWC3
        except ImportError as e:
            errors.report(e, exc_info=True)

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    alpha = g_cn_HWC3(np.asarray(mask.convert('L')).astype(np.uint8))[:, :, 0:1]
    image = np.concatenate([color, alpha], axis=2)
    return image



def convertIntoCNImageFormat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    return color



lama_model = LamaInpaint()



def convertImageIntoPILFormat(image):
    return Image.fromarray(
        np.ascontiguousarray(image.clip(0, 255).astype(np.uint8)).copy()
    )


def areImagesTheSame(image_one, image_two):
    if image_one.size != image_two.size:
        return False

    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


@dataclass
class CacheData:
    image: Any
    mask: Any
    invert: Any
    upscaler: Any
    padding: Any
    resolution: Any
    result: Any

cachedData = None


def limitSizeByMinDimension(image: Image.Image, size):
    w, h = image.size
    k = size / min(w, h)
    newW = w * k
    newH = h * k

    return int(newW), int(newH)


def lamaInpaint(image: Image, mask: Image, invert: int, upscaler: str, padding: int|None, resolution: int):
    global cachedData
    result = None
    if cachedData is not None and\
            cachedData.invert == invert and\
            cachedData.upscaler == upscaler and\
            cachedData.padding == padding and\
            cachedData.resolution == resolution and\
            areImagesTheSame(cachedData.image, image) and\
            areImagesTheSame(cachedData.mask, mask):
        result = copy.copy(cachedData.result)
        print("lama inpainted restored from cache")
        shared.state.assign_current_image(result)
    else:
        forCache = CacheData(image.copy(), mask.copy(), invert, upscaler, padding, resolution, None)
        initMaskMaybeInverted = mask.copy()
        if invert == 1:
            mask = ImageOps.invert(mask)
            initMaskMaybeInverted = copy.copy(mask)
        initImage = copy.copy(image)
        image = copy.copy(initImage)
        if padding is not None:
            image = crop(image, initMaskMaybeInverted, padding)
            mask = crop(mask, initMaskMaybeInverted, padding)
        resolution = min(*image.size, resolution)
        newW, newH = limitSizeByMinDimension(image, resolution)
        imageRes = image.resize((newW, newH))
        maskRes = mask.resize((newW, newH))
        shared.state.textinfo = "lama inpainting"
        tmpImage = lama_model(convertIntoCNMaskedImageFormat(imageRes, maskRes), resolution)
        tmpImage = convertImageIntoPILFormat(tmpImage)
        inpaintedImage = imageRes
        inpaintedImage.paste(tmpImage, maskRes)
        shared.state.assign_current_image(inpaintedImage)
        w, h = image.size
        shared.state.textinfo = "upscaling lama inpainted"
        inpaintedImage = resize_image(0, inpaintedImage.convert('RGB'), w, h, upscaler).convert('RGBA')
        result = image
        result.paste(inpaintedImage, mask)
        if padding is not None:
            result = uncrop(result, initImage, initMaskMaybeInverted, padding)
        shared.state.textinfo = ""
        forCache.result = result.copy()
        cachedData = forCache
        print("lama inpainted cached")

    return result
