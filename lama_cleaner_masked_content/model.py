import cv2
import numpy as np


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(img: np.ndarray, resolution: int):
    # Convert greyscale image to RGB.
    if img.ndim == 2:
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)

    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad



class LamaInpaint():
    def __init__(self):
        self.model = None

    def __call__(
        self,
        input_image,
        res,
    ):
        img = input_image
        H, W, C = img.shape
        assert C == 4, "No mask is provided!"
        raw_color = img[:, :, 0:3].copy()
        raw_mask = img[:, :, 3:4].copy()

        img_res, remove_pad = resize_image_with_pad(img, res)

        if self.model is None:
            from annotator.lama import LamaInpainting
            self.model = LamaInpainting()

        # applied auto inversion
        prd_color = self.model(img_res)
        prd_color = remove_pad(prd_color)
        prd_color = cv2.resize(prd_color, (W, H))

        alpha = raw_mask.astype(np.float32) / 255.0
        fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(
            np.float32
        ) * (1 - alpha)
        fin_color = fin_color.clip(0, 255).astype(np.uint8)

        result = np.concatenate([fin_color, raw_mask], axis=2)
        return result

