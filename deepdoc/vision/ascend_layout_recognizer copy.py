import os
import math
import numpy as np
import cv2
import logging
from ais_bench.infer.interface import InferSession
from deepdoc.vision import Recognizer
from deepdoc.vision.operators import nms

#from api.utils.file_utils import get_project_base_directory

class AscendLayoutRecognizer(Recognizer):
    labels = [
        "_background_",
        "Text",
        "Title",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Header",
        "Footer",
        "Reference",
        "Equation",
    ]

    def __init__(self, domain, device_id=0):
        #if not model_dir:
        #    model_dir = os.path.join(
        #        get_project_base_directory(),
        #        "rag/res/deepdoc"
        #    )
        # model_file_path = os.path.join(model_dir, task_name + ".om")
        # TODO: delete me
        model_file_path = "/home/yz/ST/deepdoc/om_infer_test/ragflow/rag/res/deepdoc/layout_1x3x1024x1024.om"
        if not os.path.exists(model_file_path):
            raise ValueError(f"Model file not found: {model_file_path}")
        
        self.session = InferSession(device_id=device_id, model_path=model_file_path)
        self.input_shape = self.session.get_inputs()[0].shape[2:4]  # H,W

    # def preprocess(self, image_list):
    #     inputs = []
    #     hh, ww = self.input_shape
    #     for img in image_list:
    #         h, w = img.shape[:2]
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(np.array(img).astype('float32'), (ww, hh))
    #         img /= 255.0
    #         img = img.transpose(2, 0, 1)
    #         img = img[np.newaxis, :, :, :].astype(np.float32)
    #         inputs.append({"image": img, "scale_factor": [w/ww, h/hh]})
    #     return inputs

    def preprocess(self, image_list):
        inputs = []
        H, W = self.input_shape  # (hh, ww)，保持你原有的获取方式
        for img in image_list:
            h, w = img.shape[:2]
            # 仍然是 BGR->RGB 与归一化，保持你的风格
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            # === 等比缩放 + 居中 padding（letterbox），但不改变返回“形式” ===
            r = min(H / h, W / w)
            new_unpad = (int(round(w * r)), int(round(h * r)))
            dw, dh = (W - new_unpad[0]) / 2.0, (H - new_unpad[1]) / 2.0

            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))

            img /= 255.0
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # 关键：保持“形式”——仍然提供 scale_factor
            # 但 scale_factor 继续用你原来的含义 [w/ww, h/hh]（不变）
            # 新增 pad 与 orig_shape，postprocess 会优先使用它们；没有则回退。
            inputs.append({
                "image": img,
                "scale_factor": [w / new_unpad[0], h / new_unpad[1]],  # 保持旧含义
                "pad": [dw, dh],                    # 新增：letterbox 反变换所需
                "orig_shape": [h, w],               # 新增：仅做兜底/调试
            })
        return inputs


    # def postprocess(self, boxes, inputs, thr=0.7):
    #     boxes = np.squeeze(boxes)  # shape (N,6)
    #     if boxes.shape[-1] != 6:
    #         raise ValueError(f"Unexpected output shape: {boxes.shape}")

    #     print(f"{boxes=}", flush=True)
    #     results = []
    #     for row in boxes:
    #         print(f"{row=}", flush=True)
    #         x1, y1, x2, y2, score, cls_id = row
    #         print("=======================", flush=True)
    #         print(f"{x1=}, {type(x1)=}",flush=True)
    #         print(f"{y1=}, {type(y1)=}",flush=True)
    #         print(f"{x2=}, {type(x2)=}",flush=True)
    #         print(f"{y2=}, {type(y2)=}",flush=True)
    #         print(f"{score=}, {type(score)=}",flush=True)
    #         print(f"{cls_id=}, {type(cls_id)=}",flush=True)
    #         print("=======================", flush=True)
    #         try:
    #             if score < thr:
    #                 continue
    #         except:
    #             print("!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
    #             print(f"{score=}", flush=True)
    #             print(f"{row=}", flush=True)
    #             print("!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
    #         cls_id = int(cls_id)
    #         print(f"{cls_id=}", flush=True)
    #         if cls_id >= len(self.labels):
    #             continue
    #         results.append({
    #             "type": self.labels[cls_id].lower(),
    #             "bbox": [float(x1), float(y1), float(x2), float(y2)],
    #             "score": float(score)
    #         })
    #     return results
    
    def postprocess(self, boxes, inputs, thr=0.25):
        # 保持你当前的形状处理方式
        arr = np.squeeze(boxes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results = []
        # 支持两类常见 YOLOv8 输出：6列 或 5+C 列
        if arr.shape[1] == 6:
            # [x1,y1,x2,y2,score,cls]
            m = arr[:, 4] >= thr
            arr = arr[m]
            if arr.size == 0:
                return []
            xyxy = arr[:, :4].astype(np.float32)
            scores = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)

            # === 关键：优先使用 pad + scale_factor 的反变换 ===
            if "pad" in inputs:
                dw, dh = inputs["pad"]
                sx, sy = inputs["scale_factor"]
                xyxy[:, [0, 2]] -= dw
                xyxy[:, [1, 3]] -= dh
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)
            else:
                # 回退到你原来的两元素 scale_factor 做线性缩放
                sx, sy = inputs["scale_factor"]
                xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)

            # per-class NMS
            keep_indices = []
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                k = nms(xyxy[idx], scores[idx], 0.45)
                keep_indices.extend(idx[k])

            for i in keep_indices:
                cid = int(cls_ids[i])
                if 0 <= cid < len(self.labels):
                    results.append({
                        "type": self.labels[cid].lower(),
                        "bbox": [float(t) for t in xyxy[i].tolist()],
                        "score": float(scores[i])
                    })
            return results

        # if arr.shape[1] > 6:
        #     # [cx,cy,w,h,obj,p1..pC]
        #     obj = arr[:, 4]
        #     cls_scores = arr[:, 5:]
        #     best = np.max(cls_scores, axis=1)
        #     cls_ids = np.argmax(cls_scores, axis=1)
        #     scores = obj * best
        #     m = scores >= thr
        #     if not np.any(m):
        #         return []
        #     arr = arr[m]; scores = scores[m]; cls_ids = cls_ids[m]

        #     xywh = arr[:, :4].astype(np.float32)
        #     xyxy = np.empty_like(xywh)
        #     xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        #     xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        #     xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        #     xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

        #     if "pad" in inputs:
        #         dw, dh = inputs["pad"]
        #         sx, sy = inputs["scale_factor"]
        #         xyxy[:, [0, 2]] -= dw
        #         xyxy[:, [1, 3]] -= dh
        #         xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)
        #     else:
        #         sx, sy = inputs["scale_factor"]
        #         xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)

        #     keep_indices = []
        #     for c in np.unique(cls_ids):
        #         idx = np.where(cls_ids == c)[0]
        #         k = nms(xyxy[idx], scores[idx], 0.45)
        #         keep_indices.extend(idx[k])

        #     for i in keep_indices:
        #         cid = int(cls_ids[i])
        #         if 0 <= cid < len(self.labels):
        #             results.append({
        #                 "type": self.labels[cid].lower(),
        #                 "bbox": [float(t) for t in xyxy[i].tolist()],
        #                 "score": float(scores[i])
        #             })
        #     return results

        raise ValueError(f"Unexpected output shape: {arr.shape}")


    def __call__(self, image_list, thr=0.7, batch_size=16, **kwargs):
        res = []
        images = []
        for i in range(len(image_list)):
            if not isinstance(image_list[i], np.ndarray):
                images.append(np.array(image_list[i]))
            else:
                images.append(image_list[i])

        batch_loop_cnt = math.ceil(float(len(images)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(images))
            batch_image_list = images[start_index:end_index]

            inputs = self.preprocess(batch_image_list)
            logging.debug("preprocess done")

            for ins in inputs:
                feeds = [ins["image"]]
                output = self.session.infer(feeds=feeds, mode="static")
                bb = self.postprocess(output[0], ins, thr)
                res.append(bb)

        return res

if __name__ == "__main__":
    label_list = ["_background_", "Text", "Title", "Figure", "Figure caption",
                  "Table", "Table caption", "Header", "Footer", "Reference", "Equation"]
    rec = AscendLayoutRecognizer("layout")
    results = rec([cv2.imread("/home/yz/ST/images/1024_1024_first_order.png"), cv2.imread("/home/yz/ST/images/1024_1024_first_order.png")])
    print(results)
