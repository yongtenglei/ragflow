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
import json
import logging
import re
import tempfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
import requests
from PIL import Image

from deepdoc.parser.pdf_parser import RAGFlowPdfParser


class PaddleOCRVLParser(RAGFlowPdfParser):
    def __init__(self, vl_rec_backend: Optional[str] = None, vl_rec_server_url: Optional[str] = None, **default_kwargs):
        self.vl_rec_backend = vl_rec_backend
        self.vl_rec_server_url = vl_rec_server_url.rstrip("/") if vl_rec_server_url else None
        self.default_kwargs = default_kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.page_images: List[Image.Image] = []
        self.page_sizes: List[Tuple[int, int]] = []
        self.page_from = 0
        self.page_to = 0
        self._page_coord_max: Dict[int, Dict[str, float]] = {}

    def _is_http_endpoint_valid(self, url: str, timeout: int = 5) -> bool:
        try:
            resp = requests.head(url, timeout=timeout, allow_redirects=True)
            return resp.status_code in [200, 301, 302, 307, 308]
        except Exception:
            return False

    def check_installation(self, vl_rec_backend: Optional[str] = None, vl_rec_server_url: Optional[str] = None) -> tuple[bool, str]:
        try:
            from paddleocr import PaddleOCRVL  # noqa: F401
        except Exception as e:  # pragma: no cover - import check
            return False, f"[PaddleOCRVL] Not available: {e}"

        backend = vl_rec_backend or self.vl_rec_backend
        server = (vl_rec_server_url or self.vl_rec_server_url or "").rstrip("/")
        if backend and "server" in backend and server:
            if self._is_http_endpoint_valid(server):
                return True, ""
            try:
                resp = requests.get(server, timeout=5)
                if resp.status_code in [200, 400, 401, 403, 404, 405]:
                    return True, ""
            except Exception as e:
                return False, f"[PaddleOCRVL] Server check failed: {server}: {e}"
            return False, f"[PaddleOCRVL] Server not reachable: {server}"
        return True, ""

    def __images__(self, fnm, zoomin: int = 1, page_from: int = 0, page_to: int = 600, callback: Optional[Callable] = None):
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for _, p in enumerate(pdf.pages[page_from:page_to])]
                self.page_sizes = [img.size for img in self.page_images]
                self.logger.info("[PaddleOCRVL] Loaded %d page images for cropping support.", len(self.page_images))
        except Exception as e:
            self.page_images = []
            self.page_sizes = []
            self.logger.warning("[PaddleOCRVL] Failed to render page images: %s", e)
            if callback:
                callback(0.05, f"[PaddleOCRVL] Failed to render page images: {e}")

    def _line_tag(self, page_idx: int, bbox: Tuple[float, float, float, float]) -> str:
        if not bbox:
            return ""
        x0, top, x1, bott = bbox
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(page_idx + 1, x0, x1, top, bott)

    def _update_page_coord_stats(self, outputs: List[Dict[str, Any]]) -> None:
        """Collect max coords per page to normalize PaddleOCR outputs into rendered page space."""
        self._page_coord_max = {}
        for out in outputs:
            page_idx = int(out.get("page_index", 0) or 0)
            bbox = out.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except Exception:
                continue
            stats = self._page_coord_max.setdefault(page_idx, {"max_x": 0.0, "max_y": 0.0})
            stats["max_x"] = max(stats["max_x"], x0, x1)
            stats["max_y"] = max(stats["max_y"], y0, y1)

    def _normalize_bbox(self, bbox: Any, page_idx: int) -> Optional[Tuple[float | int, float | int, float | int, float | int]]:
        """Scale and clamp OCR bboxes into page image coordinates to avoid invalid crops."""
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None

        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            return None

        if page_idx < 0:
            return None

        # Use rendered page size for target coordinate space
        page_w, page_h = self.page_sizes[page_idx] if page_idx < len(self.page_sizes) else (None, None)
        if not page_w or not page_h:
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            return float(x0), float(y0), float(x1), float(y1)

        stats = self._page_coord_max.get(page_idx, {})
        max_x = stats.get("max_x", 0.0)
        max_y = stats.get("max_y", 0.0)

        # Derive scaling directly from observed bbox maxima (no rounding), falling back to 1.0.
        src_w = max(float(page_w), float(max_x)) if max_x else float(page_w)
        src_h = max(float(page_h), float(max_y)) if max_y else float(page_h)

        scale_x = float(page_w) / src_w if src_w else 1.0
        scale_y = float(page_h) / src_h if src_h else 1.0

        x0, x1 = sorted([x0 * scale_x, x1 * scale_x])
        y0, y1 = sorted([y0 * scale_y, y1 * scale_y])

        # Clamp into page bounds and ensure a minimal size
        x0 = max(0.0, min(x0, float(page_w) - 1e-3))
        x1 = min(float(page_w), max(x1, x0 + 1.0))
        y0 = max(0.0, min(y0, float(page_h) - 1e-3))
        y1 = min(float(page_h), max(y1, y0 + 1.0))
        return float(x0), float(y0), float(x1), float(y1)

    def _clamp_crop_box(self, page_idx: int, left: float | int, right: float | int, top: float | int, bottom: float | int) -> Tuple[int | float, int | float, int | float, int | float]:
        """Clamp crop box to the rendered page image to avoid PIL errors."""
        if not (0 <= page_idx < len(self.page_images)):
            return left, top, right, bottom

        img_w, img_h = self.page_images[page_idx].size
        x0 = max(0, min(int(left), img_w - 1))
        y0 = max(0, min(int(top), img_h - 1))
        x1 = min(img_w, max(int(right), x0 + 1))
        y1 = min(img_h, max(int(bottom), y0 + 1))
        return x0, y0, x1, y1

    def crop(self, text: str, ZM: int = 1, need_position: bool = False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            self.logger.warning("[PaddleOCRVL] crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1]
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            if bottom <= top:
                bottom = float(top) + 2.0

            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += float(self.page_images[pn - 1].size[1])

            base_page_idx = pns[0]
            if not (0 <= base_page_idx < page_count):
                continue

            img0 = self.page_images[base_page_idx]
            x0, y0, x1, y1 = self._clamp_crop_box(base_page_idx, float(left), float(right), float(top), min(float(bottom), float(img0.size[1])))
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((base_page_idx + self.page_from, x0, x1, y0, y1))

            remain_bottom = float(bottom) - float(img0.size[1])
            for pn in pns[1:]:
                if remain_bottom <= 0:
                    break
                if not (0 <= pn < page_count):
                    continue
                page = self.page_images[pn]
                x0, y0, x1, y1 = self._clamp_crop_box(pn, float(left), float(right), 0.0, min(float(remain_bottom), float(page.size[1])))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                remain_bottom -= float(page.size[1])

        if not imgs:
            if need_position:
                return None, None
            return

        height = sum(img.size[1] + GAP for img in imgs)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, int(height)), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    @staticmethod
    def extract_positions(txt: str):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def _build_pipeline_kwargs(self, **kwargs) -> Dict[str, Any]:
        pipeline_kwargs: Dict[str, Any] = {}
        pipeline_kwargs.update({k: v for k, v in self.default_kwargs.items() if v is not None})
        if self.vl_rec_backend:
            pipeline_kwargs["vl_rec_backend"] = self.vl_rec_backend
        if self.vl_rec_server_url:
            pipeline_kwargs["vl_rec_server_url"] = self.vl_rec_server_url
        for key, value in kwargs.items():
            if value is not None:
                pipeline_kwargs[key] = value
        return pipeline_kwargs

    def _normalize_result_dict(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result.get("res", result)
        if hasattr(result, "json"):
            data = result.json
            if callable(data):
                try:
                    data = data()
                except Exception:
                    data = None
            if isinstance(data, dict):
                return data.get("res", data)
        if hasattr(result, "res"):
            data = result.res
            if callable(data):
                try:
                    data = data()
                except Exception:
                    data = None
            if isinstance(data, dict):
                return data
        if hasattr(result, "__dict__"):
            return result.__dict__
        return {}

    def _read_output(self, out_dir: Path, file_stem: str, saved_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        paths: List[Path] = []
        if saved_paths:
            paths = [Path(p) for p in saved_paths if p]

        if not paths:
            paths = sorted(out_dir.glob(f"{file_stem}_*_res.json"))
            if not paths:
                fallback = out_dir / f"{file_stem}_res.json"
                if fallback.exists():
                    paths = [fallback]

        raw_outputs: List[Dict[str, Any]] = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                raw_outputs.append(json.load(f))

        if not raw_outputs:
            raise FileNotFoundError(f"[PaddleOCRVL] No JSON outputs found under {out_dir}")

        # Process PaddleOCRVL outputs into simplified format
        processed_outputs = []
        for raw_output in raw_outputs:
            page_index = raw_output.get("page_index", 0)
            blocks = raw_output.get("parsing_res_list", [])

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                block_label = block.get("block_label", "text")
                bbox = block.get("block_bbox", [0, 0, 0, 0])
                content = block.get("block_content", "")

                processed_item = {"block_label": block_label, "page_index": page_index, "bbox": bbox, "content": content}

                processed_outputs.append(processed_item)

        return processed_outputs

    def _transfer_to_sections(self, outputs: List[Dict[str, Any]], parse_method: str = "raw") -> list:
        sections = []
        self._update_page_coord_stats(outputs)

        for output in outputs:
            block_label = output.get("block_label", "text")
            page_index = output.get("page_index", 0)
            bbox = output.get("bbox", [0.0, 0.0, 0.0, 0.0])
            content = output.get("content", "")

            norm_bbox = self._normalize_bbox(bbox, page_index)

            # Generate position tag
            tag = self._line_tag(page_index, norm_bbox)

            # Process content based on block type
            section = ""
            if block_label.lower() == "table":
                # For tables, keep the HTML structure
                section = content
            elif block_label.lower() in ["image", "figure_title"]:
                # For images and figure titles, extract clean text
                if isinstance(content, str):
                    section = re.sub(r"<[^>]+>", "", content).strip()
            else:
                # For all other types, extract clean text
                if isinstance(content, str):
                    section = re.sub(r"<[^>]+>", "", content).strip()

            if not section:
                continue

            if parse_method == "manual":
                sections.append((section, block_label, tag))
            elif parse_method == "paper":
                sections.append((section + tag, block_label))
            else:
                sections.append((section, tag))

        return sections

    def _crop_bbox(self, page_idx: int, bbox: Tuple[float | int, float | int, float | int, float | int]) -> Tuple[Optional[Image.Image], List[Tuple[int | float, int | float, int | float, int | float, int | float]]]:
        """Crop a bbox from rendered page image and return crop plus position list."""
        if bbox is None:
            return None, []
        norm_bbox = self._normalize_bbox(bbox, page_idx)
        if norm_bbox is None:
            return None, []
        x0, y0, x1, y1 = norm_bbox
        x0, y0, x1, y1 = self._clamp_crop_box(page_idx, x0, x1, y0, y1)
        if not (0 <= page_idx < len(self.page_images)):
            return None, []
        try:
            img = self.page_images[page_idx].crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        except Exception:
            img = None
        positions = [(page_idx + self.page_from, float(x0), float(x1), float(y0), float(y1))]
        return img, positions

    def _transfer_to_tables(self, outputs: List[Dict[str, Any]]):
        tables = []
        self._update_page_coord_stats(outputs)
        for output in outputs:
            block_label = str(output.get("block_label", "")).lower()
            if block_label not in ["table", "image"]:
                continue
            page_index = int(output.get("page_index", 0) or 0)
            bbox = output.get("bbox", [0.0, 0.0, 0.0, 0.0])
            content = output.get("content", "")

            norm_bbox = self._normalize_bbox(bbox, page_index)
            img, positions = self._crop_bbox(page_index, norm_bbox) if norm_bbox else (None, [])

            text_fallback = ""
            if isinstance(content, str):
                text_fallback = re.sub(r"<[^>]+>", "", content).strip()

            if block_label == "table":
                html = content if isinstance(content, str) and content.strip() else text_fallback or "<table></table>"
                tables.append(((img, html), positions))
            else:  # image
                rows = text_fallback or "[Image]"
                tables.append(((img, rows), positions))

        return tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        delete_output: bool = True,
        format_block_content: bool = True,
        use_chart_recognition: bool = False,
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_queues: Optional[bool] = None,
        vl_rec_backend: Optional[str] = None,
        vl_rec_server_url: Optional[str] = None,
        vl_rec_max_concurrency: Optional[int] = None,
        vl_rec_api_key: Optional[str] = None,
        prompt_label: Optional[str] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        device: Optional[str] = None,
        enable_hpi: Optional[bool] = None,
        use_tensorrt: Optional[bool] = None,
        precision: Optional[str] = None,
        parse_method: str = "raw",
    ) -> tuple:
        import shutil

        ok, reason = self.check_installation(vl_rec_backend=vl_rec_backend, vl_rec_server_url=vl_rec_server_url)
        if not ok:
            raise RuntimeError(reason)

        temp_pdf = None
        created_tmp_dir = False

        file_path = Path(filepath)
        safe_stem = file_path.stem.replace(" ", "_")
        suffix = file_path.suffix or ".pdf"
        pdf_file_name = safe_stem + suffix
        pdf_file_path = file_path.with_name(pdf_file_name)

        if binary:
            temp_dir = Path(tempfile.mkdtemp(prefix="paddleocrvl_bin_pdf_"))
            temp_pdf = temp_dir / pdf_file_name
            with open(temp_pdf, "wb") as f:
                f.write(binary)
            pdf = temp_pdf
            if callback:
                callback(0.10, f"[PaddleOCRVL] Received binary PDF -> {temp_pdf}")
        else:
            if pdf_file_path != file_path and file_path.exists():
                shutil.move(file_path, pdf_file_path)
            pdf = pdf_file_path
            if not pdf.exists():
                raise FileNotFoundError(f"[PaddleOCRVL] PDF not found: {pdf}")

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="paddleocrvl_out_"))
            created_tmp_dir = True

        self.logger.info(f"[PaddleOCRVL] Output directory: {out_dir}")
        if callback:
            callback(0.15, f"[PaddleOCRVL] Output directory: {out_dir}")

        if pdf.suffix.lower() == ".pdf":
            self.__images__(pdf, zoomin=1)

        pipeline_kwargs = self._build_pipeline_kwargs(
            vl_rec_backend=vl_rec_backend,
            vl_rec_server_url=vl_rec_server_url,
            vl_rec_max_concurrency=vl_rec_max_concurrency,
            vl_rec_api_key=vl_rec_api_key,
            format_block_content=format_block_content,
            use_chart_recognition=use_chart_recognition,
            use_layout_detection=use_layout_detection,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_queues=use_queues,
            prompt_label=prompt_label,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device=device,
            enable_hpi=enable_hpi,
            use_tensorrt=use_tensorrt,
            precision=precision,
        )

        filtered_log = {k: ("***" if "api_key" in k else v) for k, v in pipeline_kwargs.items()}
        self.logger.info("[PaddleOCRVL] Init kwargs: %s", filtered_log)

        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(**pipeline_kwargs)
        self.logger.info(f"[PaddleOCRVL] Running predict on {pdf}")
        if callback:
            callback(0.25, f"[PaddleOCRVL] Running predict on {pdf}")

        outputs = pipeline.predict(input=str(pdf))
        if callback:
            callback(0.55, "[PaddleOCRVL] Prediction finished, parsing outputs...")

        if not outputs:
            raise RuntimeError("[PaddleOCRVL] Empty result returned from predict.")

        saved_json_paths: List[str] = []
        for idx, res in enumerate(outputs):
            before = set(out_dir.glob("*.json"))
            if hasattr(res, "save_to_json"):
                res.save_to_json(save_path=str(out_dir))
            else:
                raise RuntimeError("[PaddleOCRVL] Result object does not support save_to_json.")

            new_files = sorted(list(set(out_dir.glob("*.json")) - before), key=lambda p: p.stat().st_mtime)
            if not new_files:
                raise RuntimeError(f"[PaddleOCRVL] No JSON file produced for page {idx}.")

            picked = new_files[-1]
            target_path = out_dir / f"{pdf.stem}_{idx}_res.json"
            if picked != target_path:
                target_path.write_bytes(picked.read_bytes())
                try:
                    picked.unlink()
                except Exception:
                    pass
            saved_json_paths.append(str(target_path))
            self.logger.info(f"[PaddleOCRVL] Saved JSON for page {idx} to {target_path}")

        results = self._read_output(out_dir, pdf.stem, saved_paths=saved_json_paths)

        try:
            sections = self._transfer_to_sections(results, parse_method=parse_method)
            tables = self._transfer_to_tables(results)
            if callback:
                callback(0.80, f"[PaddleOCRVL] Parsed {len(sections)} blocks from PDF.")
            return sections, tables
        finally:
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":  # pragma: no cover
    parser = PaddleOCRVLParser()
    ok, reason = parser.check_installation()
    print("PaddleOCRVL available:", ok, reason)
