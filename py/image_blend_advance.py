import torch
import copy
from PIL import Image
from .imagefunc import log, pil2tensor, tensor2pil, image2mask, mask2image, chop_image, chop_mode, image_rotate_extend_with_alpha



class ImageBlendAdvance:

    def __init__(self):
        self.NODE_NAME = 'ImageBlendAdvance'

    @classmethod
    def INPUT_TYPES(self):

        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "mirror": (mirror_mode,),  # 镜像翻转
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_blend_advance'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend_advance(self, background_image, layer_image,
                            invert_mask, blend_mode, opacity,
                            x_percent, y_percent,
                            mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                            layer_mask=None
                            ):

        ret_images_pil = []
        ret_masks_pil = []

        b_batch = background_image.shape[0]
        l_batch = layer_image.shape[0]
        m_batch = layer_mask.shape[0] if layer_mask is not None else 0
        max_batch = max(b_batch, l_batch, m_batch)

        for i in range(max_batch):
            b_idx = i if i < b_batch else b_batch - 1
            l_idx = i if i < l_batch else l_batch - 1

            _canvas = tensor2pil(background_image[b_idx:b_idx+1]).convert('RGB')
            _layer_full = tensor2pil(layer_image[l_idx:l_idx+1])
            _layer = _layer_full.convert('RGB')

            if layer_mask is not None:
                m_idx = i if i < m_batch else m_batch - 1
                _m = layer_mask[m_idx:m_idx+1]
                if invert_mask:
                    _m = 1 - _m
                _mask = tensor2pil(_m).convert('L')
            else:
                if _layer_full.mode == 'RGBA':
                    _mask = _layer_full.split()[-1]
                else:
                    _mask = Image.new('L', size=_layer.size, color='white')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')

            orig_layer_width = _layer.width
            orig_layer_height = _layer.height
            _mask = _mask.convert("RGB")

            target_layer_width = int(orig_layer_width * scale)
            target_layer_height = int(orig_layer_height * scale * aspect_ratio)

            # mirror
            if mirror == 'horizontal':
                _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

            # scale
            _layer = _layer.resize((target_layer_width, target_layer_height))
            _mask = _mask.resize((target_layer_width, target_layer_height))
            # rotate
            _layer, _mask, _ = image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)

            # 处理位置
            x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
            y = int(_canvas.height * y_percent / 100 - _layer.height / 2)

            # composit layer
            _comp = copy.copy(_canvas)
            _compmask = Image.new("RGB", _comp.size, color='black')
            _comp.paste(_layer, (x, y))
            _compmask.paste(_mask, (x, y))
            _compmask = _compmask.convert('L')
            _comp = chop_image(_canvas, _comp, blend_mode, opacity)

            # composition background
            _canvas.paste(_comp, mask=_compmask)

            ret_images_pil.append(_canvas)
            ret_masks_pil.append(_compmask)

        log(f"{self.NODE_NAME} Processed {len(ret_images_pil)} image(s).", message_type='finish')
        return (
            torch.cat([pil2tensor(img) for img in ret_images_pil], dim=0),
            torch.cat([image2mask(m) for m in ret_masks_pil], dim=0),
        )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance": ImageBlendAdvance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance": "LayerUtility: ImageBlendAdvance"
}
