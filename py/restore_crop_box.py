import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, ChunkedDiskStore



class RestoreCropBox:

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = '😺dzNodes/LayerUtility'

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         croped_mask=None
                         ):

        store_img = ChunkedDiskStore()
        store_mask = ChunkedDiskStore()

        # croped_mask 预处理：确保是 3D
        if croped_mask is not None and croped_mask.dim() == 2:
            croped_mask = torch.unsqueeze(croped_mask, 0)

        b_batch = background_image.shape[0]
        l_batch = croped_image.shape[0]
        m_batch = croped_mask.shape[0] if croped_mask is not None else 0
        max_batch = max(b_batch, l_batch, m_batch)

        for i in range(max_batch):
            # 直接用索引取当前帧，不预收集到列表
            b_idx = i if i < b_batch else b_batch - 1
            l_idx = i if i < l_batch else l_batch - 1

            _canvas = tensor2pil(background_image[b_idx:b_idx+1]).convert('RGB')
            _layer = tensor2pil(croped_image[l_idx:l_idx+1]).convert('RGB')

            # 获取当前帧的 mask
            if croped_mask is not None:
                m_idx = i if i < m_batch else m_batch - 1
                _m = croped_mask[m_idx:m_idx+1]
                if invert_mask:
                    _m = 1 - _m
                _mask = tensor2pil(_m).convert('L')
            else:
                # 从 croped_image 的 alpha 通道自动提取
                _layer_rgba = tensor2pil(croped_image[l_idx:l_idx+1])
                if _layer_rgba.mode == 'RGBA':
                    _mask = _layer_rgba.split()[-1]
                else:
                    _mask = Image.new('L', size=_layer.size, color='white')

            ret_mask = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
            ret_mask.paste(_mask, box=tuple(crop_box))

            store_img.add(_canvas)
            store_mask.add(ret_mask)

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        result_img = store_img.to_tensor()
        result_mask = store_mask.to_tensor()
        store_img.cleanup()
        store_mask.cleanup()
        return (result_img, result_mask,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: RestoreCropBox": RestoreCropBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RestoreCropBox": "LayerUtility: RestoreCropBox"
}
