import os
import math
import numpy as np
import cv2
import logging
from copy import deepcopy
from ais_bench.infer.interface import InferSession
from deepdoc.vision import Recognizer
from deepdoc.vision.operators import nms

#from api.utils.file_utils import get_project_base_directory

class AscendLayoutRecognizer(Recognizer):
    # labels = [
    #     "_background_",
    #     "Text",
    #     "Title",
    #     "Figure",
    #     "Figure caption",
    #     "Table",
    #     "Table caption",
    #     "Header",
    #     "Footer",
    #     "Reference",
    #     "Equation",
    # ]
    labels = [
        "title",
        "Text",
        "Reference",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Table caption",
        "Equation",
        "Figure caption",
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
        self.garbage_layouts = ["footer", "header", "reference"]


    def preprocess(self, image_list):
        inputs = []
        H, W = self.input_shape  # (hh, ww)，保持你原有的获取方式
        for img in image_list:
            print(f"in preprocess {img.shape=}", flush=True)
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
            print(f"in preprocess after transpose {img.shape=}", flush=True)

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


    
    def postprocess(self, boxes, inputs, thr=0.25):
        # 保持你当前的形状处理方式
        arr = np.squeeze(boxes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results = []
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

        raise ValueError(f"Unexpected output shape: {arr.shape}")


    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):
        import re
        from collections import Counter
        print("&&&&&&&&&&&&", flush=True)
        print(f"{len(image_list)=}", flush=True)
        print(f"{ocr_res=}", flush=True)
        assert len(image_list) == len(ocr_res), "image_list 与 ocr_res 页数不一致"

        # ---- 1) 用你现有的推理链（preprocess -> infer -> postprocess）得到每页 layouts ----
        images = [np.array(im) if not isinstance(im, np.ndarray) else im for im in image_list]
        layouts_all_pages = []  # list of list[{"type","score","bbox":[x1,y1,x2,y2]}]

        # 注意：YOLOv8 一般用较低 conf 阈值；若用户传了更高 thr，取 max 保守一些
        conf_thr = max(thr, 0.08)

        batch_loop_cnt = math.ceil(float(len(images)) / batch_size)
        for bi in range(batch_loop_cnt):
            s = bi * batch_size
            e = min((bi + 1) * batch_size, len(images))
            batch_images = images[s:e]

            # 逐张做 preprocess（也可以自行拼 batch，只要 InferSession 支持）
            inputs_list = self.preprocess(batch_images)
            logging.debug("preprocess done")

            for ins in inputs_list:
                feeds = [ins["image"]]
                out_list = self.session.infer(feeds=feeds, mode="static")
                
                for out in out_list:
                    lts = self.postprocess(out, ins, conf_thr)

                    print("##################", flush=True)
                    print(f"{lts[:5]=}", flush=True)
                    print(f"{len(feeds)=}", flush=True)
                    print(f"{len(feeds)=}", flush=True)
                    print(f"{type(feeds[0])=}", flush=True)
                    print(f"{feeds[0].shape=}", flush=True)

                    print("##################", flush=True)

                    # try:
                    print(f"feeds shape: {feeds[0].shape}, dtype: {feeds[0].dtype}")
                    from deepdoc.vision.seeit import save_results
                    save_results(feeds, lts, self.labels, output_dir='output/', threshold=conf_thr)
                    # except Exception as e:
                    # print(f"ERROR see it {e}", flush=True)   
                    
                    # 统一为后续所需字段
                    page_lts = []
                    for b in lts:
                        if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts:
                            x0, y0, x1, y1 = b["bbox"]
                            page_lts.append({
                                "type": b["type"],
                                "score": float(b["score"]),
                                "x0": float(x0) / scale_factor, "x1": float(x1) / scale_factor,
                                "top": float(y0) / scale_factor, "bottom": float(y1) / scale_factor,
                                "page_number": len(layouts_all_pages)
                            })
                    layouts_all_pages.append(page_lts)

        # ---- 2) 页面级处理：排序、清理、与 OCR 匹配、垃圾文本清除、figure/equation 补空框 ----
        # garbage_layouts = {"footer", "header", "reference"}
        def _is_garbage_text(box):
            patt = [r"^•+$", r"^[0-9]{1,2} / ?[0-9]{1,2}$",
                    r"^[0-9]{1,2} of [0-9]{1,2}$", r"^http://[^ ]{12,}",
                    r"\(cid *: *[0-9]+ *\)"]
            return any(re.search(p, box.get("text", "")) for p in patt)

        boxes_out = []
        page_layout = []
        garbages = {}

        print(f"{layouts_all_pages[:5]=}",flush=True)
        print(f"{len(layouts_all_pages)}", flush=True)
        for pn, lts in enumerate(layouts_all_pages):
            # 排序（按行优先）
            if lts:
                avg_h = np.mean([lt["bottom"] - lt["top"] for lt in lts])
                lts = self.sort_Y_firstly(lts, avg_h / 2 if avg_h > 0 else 0)

            # 清理相互重叠的布局框（复用父类策略，考虑 OCR 覆盖面积）
            bxs = ocr_res[pn]
            lts = self.layouts_cleanup(bxs, lts)
            page_layout.append(lts)

            # 与 OCR 匹配：给 OCR 框打 layout_type 与 layoutno
            def _tag_layout(ty):
                nonlocal bxs, lts
                lts_of_ty = [lt for lt in lts if lt["type"] == ty]
                i = 0
                while i < len(bxs):
                    # 已有标签的略过
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    # 明显垃圾文本先丢
                    if _is_garbage_text(bxs[i]):
                        bxs.pop(i)
                        continue

                    ii = self.find_overlapped_with_threshold(bxs[i], lts_of_ty, thr=0.4)
                    if ii is None:
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue

                    lts_of_ty[ii]["visited"] = True

                    # header/footer 的位置保留规则
                    keep_feats = [
                        lts_of_ty[ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].shape[0] * 0.9 / scale_factor,
                        lts_of_ty[ii]["type"] == "header" and bxs[i]["top"]    > image_list[pn].shape[0] * 0.1 / scale_factor,
                    ]
                    if drop and lts_of_ty[ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        garbages.setdefault(lts_of_ty[ii]["type"], []).append(bxs[i].get("text", ""))
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_of_ty[ii]["type"] if lts_of_ty[ii]["type"] != "equation" else "figure"
                    i += 1

            for ty in ["footer", "header", "reference", "figure caption",
                    "table caption", "title", "table", "text", "figure", "equation"]:
                _tag_layout(ty)

            # 给没有文本覆盖的 figure/equation 增加“空文本框”，便于下游流程统一处理
            figs = [lt for lt in lts if lt["type"] in ["figure", "equation"]]
            for i, lt in enumerate(figs):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                lt.pop("type", None)
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                bxs.append(lt)

            # 累加当前页（含新加的空框）
            boxes_out.extend(bxs)

        print(f"{boxes_out=}", flush=True)
        print(f"{len(boxes_out)=}", flush=True)

        # 去除重复的垃圾文本（页眉/页脚等重复行）
        garbag_set = set()
        for k, lst in garbages.items():
            cnt = Counter(lst)
            for g, c in cnt.items():
                if c > 1:
                    garbag_set.add(g)

        # ocr_res_new = [b for b in boxes_out if b.get("text", "").strip() not in garbag_set]
        ocr_res_new = [b for b in boxes_out if b["text"].strip() not in garbag_set]
        print(f"{ocr_res[:5]=}", flush=True)
        print(f"{ocr_res_new[:5]=}", flush=True)
        print(f"{len(ocr_res_new)=}", flush=True)
        print(f"{len(page_layout)=}", flush=True)
        print(f"{page_layout=}", flush=True)            
        return ocr_res_new, page_layout


if __name__ == "__main__":
    label_list = ["_background_", "Text", "Title", "Figure", "Figure caption",
                  "Table", "Table caption", "Header", "Footer", "Reference", "Equation"]
    rec = AscendLayoutRecognizer("layout")
    results = rec([cv2.imread("/home/yz/ST/images/1024_1024_first_order.png"), cv2.imread("/home/yz/ST/images/1024_1024_first_order.png")])
    print(results)
