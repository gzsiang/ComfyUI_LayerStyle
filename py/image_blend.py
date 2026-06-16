import torch
from PIL import Image
from .imagefunc import log, pil2tensor, tensor2pil, image2mask, mask2image, chop_image, chop_mode, ChunkedDiskStore



class ImageBlend:

    def __init__(self):
        self.NODE_NAME = 'ImageBlend'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  layer_mask=None
                  ):

        store = ChunkedDiskStore()

        b_batch = background_image.shape[0]
        l_batch = layer_image.shape[0]
        m_batch = layer_mask.shape[0] if layer_mask is not None else 0
        max_batch = max(b_batch, l_batch, m_batch)

        for i in range(max_batch):
            # 直接索引，不预收集
            b_idx = i if i < b_batch else b_batch - 1
            l_idx = i if i < l_batch else l_batch - 1

            _canvas = tensor2pil(background_image[b_idx:b_idx+1]).convert('RGB')
            _layer_full = tensor2pil(layer_image[l_idx:l_idx+1])
            _layer = _layer_full.convert('RGB')

            # 获取当前帧 mask
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

            # 合成layer
            _comp = chop_image(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            store.add(_canvas)

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        result = store.to_tensor()
        store.cleanup()
        return (result,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageBlend": ImageBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageBlend": "LayerUtility: ImageBlend"
}
