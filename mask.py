import json
import os
import cv2
import torch
import numpy as np
from glob import glob

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def masking(masks, colors, im_gpu, alpha=0.5):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        colors = torch.tensor(colors, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        image = cv2.imread(im_mask)
        cv2.imshow(image)


ROOT_DIR = os.getcwd()

files = glob(ROOT_DIR + "\\*.json")
files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

colors = colors(4, True)
for file in files:
    image_filename = f'{file}.jpg'
    json_filename = f'{file}.json'
    json_file = json.load(open(json_filename, 'r'))
    image = cv2.imread(image_filename) # shape(h, w, 3)
    masks = json_file['shapes']
    for mask in masks:
        mask = mask['points']
        roi_points = np.array(mask, np.int32)
        
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [roi_points], colors)
        image = cv2.addWeighted(image, 1, mask, 0.25, 0)

    cv2.imwrite(f'{file}_masked.jpg', image)
