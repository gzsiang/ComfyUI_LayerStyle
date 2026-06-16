import torch
import copy
from PIL import Image
from .imagefunc import log, pil2tensor, tensor2pil, image2mask, mask2image, chop_image_v2, chop_mode_v2, image_rotate_extend_with_alpha, ChunkedDiskStore



class ImageBlendAdvanceV3:

    def __init__(self):
        self.NODE_NAME = 'ImageBlendAdvanceV3'

    @classmethod
    def INPUT_TYPES(self):

        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
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
                "background_image": ("IMAGE", ),  #
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_blend_advance_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend_advance_v2(self, layer_image, invert_mask, blend_mode, opacity,
                            x_percent, y_percent, mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing, background_image=None, layer_mask=None
                            ):

        store_img = ChunkedDiskStore()
        store_mask = ChunkedDiskStore()

        # 预先创建透明背景 PIL（不转 tensor），或直接用输入 tensor
        bg_pils = None
        if background_image is None:
            bg_pils = []
            for l in layer_image:
                m = tensor2pil(l)
                bg_pils.append(Image.new('RGBA', (m.width, m.height), (0, 0, 0, 0)))
            b_batch = len(bg_pils)
        else:
            b_batch = background_image.shape[0]

        l_batch = layer_image.shape[0]
        m_batch = layer_mask.shape[0] if layer_mask is not None else 0
        max_batch = max(b_batch, l_batch, m_batch)

        for i in range(max_batch):
            b_idx = i if i < b_batch else b_batch - 1
            l_idx = i if i < l_batch else l_batch - 1

            if bg_pils is not None:
                _canvas = bg_pils[b_idx].copy()
            else:
                _canvas = tensor2pil(background_image[b_idx:b_idx+1]).convert('RGBA')

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
            _mask = _mask.convert("RGBA")

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
            _compmask = Image.new("RGBA", _comp.size, color='black')
            _comp.paste(_layer, (x, y))
            _compmask.paste(_mask, (x, y))
            _compmask = _compmask.convert('L')
            _comp = chop_image_v2(_canvas, _comp, blend_mode, opacity)

            # composition background
            _canvas.paste(_comp, mask=_compmask)

            store_img.add(_canvas)
            store_mask.add(_compmask)

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        result_img = store_img.to_tensor()
        result_mask = store_mask.to_tensor()
        store_img.cleanup()
        store_mask.cleanup()
        return (result_img, result_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance V3": ImageBlendAdvanceV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlendAdvance V3": "LayerUtility: ImageBlendAdvance V3"
}
