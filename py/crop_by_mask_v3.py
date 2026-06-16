import torch

from .imagefunc import log, tensor2pil, pil2tensor, mask2image, image2mask, gaussian_blur, min_bounding_rect, max_inscribed_rect, mask_area
from .imagefunc import num_round_up_to_multiple, draw_rect, ChunkedDiskStore



class CropByMaskV3:

    def __init__(self):
        self.NODE_NAME = 'CropByMask V3'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v3'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_by_mask_v3(self, image, mask, invert_mask, detect,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve, round_to_multiple,
                     crop_box=None
                     ):

        store_img = ChunkedDiskStore()
        store_mask = ChunkedDiskStore()

        b_batch = image.shape[0]

        # mask 预处理，只使用第一张
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        if mask.shape[0] > 1:
            log(f"Warning: Multiple mask inputs, using the first.", message_type='warning')
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask

        _mask_pil = tensor2pil(mask).convert('L')
        preview_image = tensor2pil(mask).convert('RGBA')

        if crop_box is None:
            bluredmask = gaussian_blur(mask2image(mask), 20).convert('L')
            x = 0
            y = 0
            w = 0
            h = 0
            if detect == "min_bounding_rect":
                (x, y, w, h) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, w, h) = max_inscribed_rect(bluredmask)
            else:
                (x, y, w, h) = mask_area(_mask_pil)

            canvas_width, canvas_height = tensor2pil(image[0:1]).convert('RGBA').size
            x1 = x - left_reserve if x - left_reserve > 0 else 0
            y1 = y - top_reserve if y - top_reserve > 0 else 0
            x2 = x + w + right_reserve if x + w + right_reserve < canvas_width else canvas_width
            y2 = y + h + bottom_reserve if y + h + bottom_reserve < canvas_height else canvas_height

            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                width = num_round_up_to_multiple(x2 - x1, multiple)
                height = num_round_up_to_multiple(y2 - y1, multiple)
                x1 = x1 - (width - (x2 - x1)) // 2
                y1 = y1 - (height - (y2 - y1)) // 2
                x2 = x1 + width
                y2 = y1 + height
            else:
                width = x2 - x1
                height = y2 - y1

            log(f"{self.NODE_NAME}: Box detected. x={x1},y={y1},width={width},height={height}")
            crop_box = (x1, y1, x2, y2)
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#F00000",
                                      line_width=(w + h) // 100)
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)

        for i in range(b_batch):
            _canvas = tensor2pil(image[i:i+1]).convert('RGBA')
            store_img.add(_canvas.crop(crop_box))
            store_mask.add(_mask_pil.crop(crop_box))

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        result_img = store_img.to_tensor()
        result_mask = store_mask.to_tensor()
        store_img.cleanup()
        store_mask.cleanup()
        return (result_img, result_mask, list(crop_box), pil2tensor(preview_image),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V3": CropByMaskV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V3": "LayerUtility: CropByMask V3"
}
