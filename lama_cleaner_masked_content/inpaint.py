from PIL import Image, ImageOps
import copy
from typing import Any
from dataclasses import dataclass
from modules.images import resize_image
from modules import shared
from .model import LamaInpaint
from .tools import (crop, uncrop, convertImageIntoPILFormat, convertIntoCNMaskedImageFormat, areImagesTheSame,
                    applyMaskBlur, limitSizeByMinDimension
)
lama_model = LamaInpaint()




@dataclass
class CacheData:
    image: Any
    mask: Any
    invert: Any
    upscaler: Any
    padding: Any
    resolution: Any
    blur: Any
    result: Any

cachedData = None




def lamaInpaint(image: Image, mask: Image, invert: int, upscaler: str, padding: int|None, resolution: int, blur: int):
    global cachedData
    result = None
    if cachedData is not None and\
            cachedData.invert == invert and\
            cachedData.upscaler == upscaler and\
            cachedData.padding == padding and\
            cachedData.resolution == resolution and\
            cachedData.blur == blur and\
            areImagesTheSame(cachedData.image, image) and\
            areImagesTheSame(cachedData.mask, mask):
        result = copy.copy(cachedData.result)
        print("lama inpainted restored from cache")
        shared.state.assign_current_image(result)
    else:
        forCache = CacheData(image.copy(), mask.copy(), invert, upscaler, padding, resolution, blur, None)
        if invert == 1:
            mask = ImageOps.invert(mask)
        mask = applyMaskBlur(mask, blur)
        initImage = copy.copy(image)
        image = copy.copy(initImage)
        if padding is not None:
            maskNotCropped = mask
            image = crop(image, maskNotCropped, padding, resolution)
            mask = crop(mask, maskNotCropped, padding, resolution)
        resolution = min(*image.size, resolution)
        newW, newH = limitSizeByMinDimension(image.size, resolution)
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
            result = uncrop(result, initImage, maskNotCropped, padding)
        shared.state.textinfo = ""
        forCache.result = result.copy()
        cachedData = forCache
        print("lama inpainted cached")

    return result
