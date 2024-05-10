import numpy as np
from PIL import Image, ImageChops, ImageOps
import copy
import logging
from typing import Any
from dataclasses import dataclass
from modules.images import resize_image
from modules import shared, errors

try:
    from lib_controlnet import global_state
    IS_WEBUI_FORGE = True
except ImportError:
    IS_WEBUI_FORGE = False


g_cn_HWC3 = None
def convertIntoCNMaskedImageFromat(image, mask):
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



def convertIntoCNImageFromat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    return color

  

supported_preprocessor = None


def lamaCNInpaint(image):
    global supported_preprocessor
    if supported_preprocessor is None:
        from scripts import supported_preprocessor
    lama = supported_preprocessor.Preprocessor.get_preprocessor('inpaint_only+lama')
    return lama(image, None).value


def lamaCNForgeInpaint(image, mask):
    lama = global_state.get_preprocessor('inpaint_only+lama')
    return lama(image, None, input_mask=mask)


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
    result: Any

cachedData = None


def limitSizeByMinDemention(image: Image, size):
    w, h = image.size
    k = size / min(w, h)
    newW = w * k
    newH = h * k

    return int(newW), int(newH)


def lamaInpaint(image: Image, mask: Image, invert: int, upscaler: str):
    global cachedData
    result = None
    if cachedData is not None and\
            cachedData.invert == invert and\
            cachedData.upscaler == upscaler and\
            areImagesTheSame(cachedData.image, image) and\
            areImagesTheSame(cachedData.mask, mask):
        result = copy.copy(cachedData.result)
        print("lama inpainted restored from cache")
        shared.state.assign_current_image(result)
    else:
        initMask = copy.copy(mask)
        if invert == 1:
            mask = ImageOps.invert(mask)
        initImage = copy.copy(image)
        image = copy.copy(initImage)
        newW, newH = limitSizeByMinDemention(image, 256)
        image256 = image.resize((newW, newH))
        mask256 = mask.resize((newW, newH))
        if IS_WEBUI_FORGE:
            tmpImage = lamaCNForgeInpaint(convertIntoCNImageFromat(image256), convertIntoCNImageFromat(mask256))
        else:
            tmpImage = lamaCNInpaint(convertIntoCNMaskedImageFromat(image256, mask256))
        tmpImage = convertImageIntoPILFormat(tmpImage)
        inpaintedImage = image256
        inpaintedImage.paste(tmpImage, mask256)
        shared.state.assign_current_image(inpaintedImage)
        w, h = image.size
        inpaintedImage = resize_image(0, inpaintedImage.convert('RGB'), w, h, upscaler).convert('RGBA')
        result = image
        result.paste(inpaintedImage, mask)
        cachedData = CacheData(initImage, initMask, invert, upscaler, copy.copy(result))
        print("lama inpainted cached")

    return result
