#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import PIL
from PIL import ImageDraw, Image
from PIL import Image
import numpy as np

def save_results(image_list, results, labels, output_dir='output/', threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, im in enumerate(image_list):
        print("#########################", flush=True)
        print(f"{results[idx]=}",flush=True)
        # Convert PyTorch-style tensor (1, C, H, W) to PIL Image
        if isinstance(im, np.ndarray):
            # Remove batch dimension (1, C, H, W) -> (C, H, W)
            if im.shape[0] == 1:
                im = im.squeeze(0)
            
            # Transpose (C, H, W) -> (H, W, C)
            im = np.transpose(im, (1, 2, 0))
            
            # Convert float32 [0, 1] to uint8 [0, 255]
            if im.dtype == np.float32:
                im = (im * 255).astype(np.uint8)
            
            # Convert to PIL Image
            im = Image.fromarray(im, mode='RGB')
        im = draw_box(im, results[idx], labels, threshold=threshold)

        out_path = os.path.join(output_dir, f"{idx}.jpg")
        im.save(out_path, quality=95)
        logging.debug("save result to: " + out_path)


def draw_box(im, result, labels, threshold=0.5):
    draw_thickness = min(im.size) // 320
    print(f"{draw_thickness=}", flush=True)
    draw = ImageDraw.Draw(im)
    color_list = get_color_map_list(len(labels))
    clsid2color = {n.lower():color_list[i] for i,n in enumerate(labels)}
    print(f"{color_list=}", flush=True)
    print(f"{clsid2color=}", flush=True)
    # print("**************", flush=True)
    # for r in result:
    #     print(f"{r=}", flush=True)
    # result = [r for r in result if r["score"] >= threshold]
    # print("**************", flush=True)
    # for dt in result:
    dt = result
    color = tuple(clsid2color[dt["type"]])
    xmin, ymin, xmax, ymax = dt["bbox"]
    draw.line(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
            (xmin, ymin)],
        width=draw_thickness,
        fill=color)

    # draw label
    text = "{} {:.4f}".format(dt["type"], dt["score"])
    tw, th = imagedraw_textsize_c(draw, text)
    draw.rectangle(
        [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
    draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def imagedraw_textsize_c(draw, text):
    if int(PIL.__version__.split('.')[0]) < 10:
        tw, th = draw.textsize(text)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tw, th = right - left, bottom - top

    return tw, th
