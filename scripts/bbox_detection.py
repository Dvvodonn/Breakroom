from __future__ import annotations
import numpy as np
import os
import csv
from typing import Union, Dict, Any, Optional, List

import torch
from mmdet.apis import DetInferencer
from mmengine import DefaultScope
import mmdet  # ensure mmdet registries are imported
import cv2
from mmengine.visualization import Visualizer
# Additional imports for config patching
from mmengine.config import Config
from mmdet.datasets import CocoDataset
from mmengine.dataset import pseudo_collate
from mmdet.structures import DetDataSample
from mmcv.transforms import Compose

class QuietDetInferencer(DetInferencer):
    def _init_visualizer(self, cfg):
        # Create a minimal visualizer that does not depend on self.model
        # (MMDet 3.3 + MMEngine 0.10 sometimes calls this before model is set)
        return Visualizer(name="noop-vis", vis_backends=[], save_dir=None)

class MmdetBBoxDetector:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda:0",
        score_thr: float = 0.3,
    ):
        # Ensure the MMEngine default scope is set so registries resolve to mmdet
        try:
            if not DefaultScope.check_instance_created('mmdet'):
                DefaultScope.get_instance('mmdet', scope_name='mmdet')
        except Exception:
            # Best-effort: ignore if scope is already set in a different name
            pass

        # Load config and ensure test_dataloader exists with a minimal pipeline & metainfo
        if isinstance(config, str):
            cfg = Config.fromfile(config)
        else:
            cfg = config  # already a Config

        # Build a minimal COCO-style test pipeline
        minimal_test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='PackDetInputs'),
        ]

        # Inject test_dataloader if missing or incomplete
        need_inject = False
        if 'test_dataloader' not in cfg:
            need_inject = True
        else:
            ds = cfg.test_dataloader.get('dataset', {})
            if not isinstance(ds, dict) or 'type' not in ds or 'pipeline' not in ds:
                need_inject = True

        if need_inject:
            cfg.test_dataloader = dict(
                dataset=dict(
                    type='CocoDataset',
                    metainfo=dict(classes=CocoDataset.METAINFO['classes']),
                    pipeline=minimal_test_pipeline,
                    lazy_init=True,
                )
            )
            # Keep val consistent to avoid warnings if used later
            cfg.val_dataloader = cfg.test_dataloader

        # Now create the inferencer with the patched cfg
        self.infer = QuietDetInferencer(model=cfg, weights=checkpoint, device=device)
        # Separate pipeline for ndarray inputs (used by our predict on video frames)
        self.nd_pipeline = Compose([
            dict(type='LoadImageFromNDArray'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='PackDetInputs'),
        ])
        self.score_thr = score_thr
        # Pull class names from model metainfo if available
        self.classes = []
        model = getattr(self.infer, 'model', None)
        if model is not None and getattr(model, 'dataset_meta', None):
            classes = model.dataset_meta.get('classes', None)
            if classes is not None:
                self.classes = list(classes)


    @torch.inference_mode()
    def predict(self, img: Union[str, np.ndarray, torch.Tensor, list, tuple]) -> Dict[str, Any]:
        """Run inference using our NDArray pipeline and let MMDet's data_preprocessor
        run exactly once inside model.test_step (avoids list->conv2d issues).
        Also ensures dtype and device safety for torch inputs.
        """
        # If a path is given, read to ndarray for a unified path
        if isinstance(img, str):
            frame = cv2.imread(img)
            if frame is None:
                raise FileNotFoundError(img)
        else:
            frame = img

        # Ensure 3-channel BGR
        frame = self._ensure_bgr3(frame)

        # Pipeline -> collate (do NOT call data_preprocessor here)
        data = self.nd_pipeline(dict(img=frame, img_id=0))
        batch = pseudo_collate([data])

        # IMPORTANT: use model's preprocessor to stack into a Tensor
        processed = self.infer.model.data_preprocessor(batch, False)
        inputs = processed['inputs']
        data_samples = processed['data_samples']

        # Some configs (e.g., with batch augments/flip-test) return a list of tensors.
        # Coerce to a single Tensor of shape (N, C, H, W).
        if isinstance(inputs, list):
            flat: list[torch.Tensor] = []

            def _flatten(xs):
                for x in xs:
                    if isinstance(x, list):
                        _flatten(x)
                    else:
                        flat.append(x)

            _flatten(inputs)

            if not flat:
                raise RuntimeError('Preprocessor produced an empty inputs list')

            # Ensure all tensors are 3D (C,H,W) or 4D (N,C,H,W)
            if all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in flat):
                # Concatenate batch tensors along batch dim
                inputs = torch.cat(flat, dim=0)
            elif all(isinstance(t, torch.Tensor) and t.dim() == 3 for t in flat):
                # Stack single images into a batch
                inputs = torch.stack(flat, dim=0)
            else:
                shapes = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t) for t in flat]
                raise RuntimeError(f'Unexpected preprocessor output shapes: {shapes}')

        # Final guard: inputs must be a Tensor of shape (N,C,H,W)
        if not (isinstance(inputs, torch.Tensor) and inputs.dim() == 4):
            raise RuntimeError(f'inputs to model must be 4D Tensor, got {type(inputs)} with dim={getattr(inputs, "dim", lambda: None)()}')

        # --- DTYPE & DEVICE NORMALIZATION BLOCK (inserted here) ---
        # Ensure input is a float tensor in the correct range
        img_for_infer = inputs
        if isinstance(img_for_infer, torch.Tensor):
            if img_for_infer.dtype == torch.uint8:
                img_for_infer = img_for_infer.float() / 255.0
        elif isinstance(img_for_infer, (list, tuple)):
            img_for_infer = [t.float() / 255.0 if t.dtype == torch.uint8 else t for t in img_for_infer]

        # Move to model device
        model_device = next(self.infer.model.parameters()).device
        if isinstance(img_for_infer, torch.Tensor):
            img_for_infer = img_for_infer.to(model_device)
        elif isinstance(img_for_infer, (list, tuple)):
            img_for_infer = [t.to(model_device) for t in img_for_infer]
        # ----------------------------------------------------------

        # Single inference call; apply thresholding in this function only
        results = self.infer.model.predict(img_for_infer, data_samples)
        result = results[0] if isinstance(results, (list, tuple)) else results

        if isinstance(result, DetDataSample):
            inst = result.pred_instances
            bboxes = inst.bboxes.cpu().numpy() if hasattr(inst, 'bboxes') else np.empty((0, 4))
            scores = inst.scores.cpu().numpy() if hasattr(inst, 'scores') else np.empty((0,))
            labels = inst.labels.cpu().numpy() if hasattr(inst, 'labels') else np.empty((0,), dtype=int)
        else:
            bboxes = np.empty((0, 4))
            scores = np.empty((0,))
            labels = np.empty((0,), dtype=int)

        # Keep only the 'person' class (COCO index 0). If class names are available,
        # look up the index dynamically; otherwise default to 0.
        if self.classes:
            try:
                person_idx = self.classes.index('person')
            except ValueError:
                person_idx = 0
        else:
            person_idx = 0

        cls_mask = (labels == person_idx)
        keep = cls_mask & (scores >= self.score_thr)
        return {
            'boxes': bboxes[keep],
            'scores': scores[keep],
            'labels': labels[keep],
        }
    def draw(
        self,
        img: np.ndarray,
        pred: Optional[Dict[str, Any]] = None,
        score_thr: Optional[float] = None,
    ) -> np.ndarray:
        """Draw bounding boxes on an image."""
        if pred is None:
            # Optionally override threshold for this call
            old_thr = self.score_thr
            if score_thr is not None:
                self.score_thr = score_thr
            try:
                pred = self.predict(img)
            finally:
                self.score_thr = old_thr
        else:
            # If a prediction dict is provided, use it as-is to avoid double filtering.
            # Only apply an extra filter if caller explicitly passes a different score_thr.
            if score_thr is not None:
                thr = float(score_thr)
                import numpy as np
                keep = pred.get('scores', np.empty((0,))) >= thr
                pred = {
                    'boxes': pred.get('boxes', np.empty((0, 4)))[keep],
                    'scores': pred.get('scores', np.empty((0,)))[keep],
                    'labels': pred.get('labels', np.empty((0,), dtype=int))[keep],
                }

        vis = img.copy()
        for (x1, y1, x2, y2), s, l in zip(pred['boxes'], pred['scores'], pred['labels']):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls_name = str(int(l))
            if self.classes and int(l) < len(self.classes):
                cls_name = self.classes[int(l)]
            cv2.putText(vis, f"{cls_name}:{s:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

    @staticmethod
    def imread(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    @staticmethod
    def _ensure_bgr3(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is standard 8-bit 3-channel BGR.
        Converts grayscale or YUV planes to BGR when needed.
        """
        if frame is None:
            return frame
        if frame.ndim == 2:  # grayscale
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        # Fallback conversion for unusual formats
        try:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        except Exception:
            return frame

    def video(self, video_path: str, out_dir: str, every_n: int = 1) -> List[np.ndarray]:
        """
        Break a video into frames saved under `out_dir`, then load each frame using
        `self.imread()` and return a list of images (BGR np.ndarrays).

        Args:
            video_path: Path to the input video file.
            out_dir: Directory to write extracted frames (will be created if needed).
            every_n: Keep 1 frame every `n` frames (e.g., 1 = keep all, 5 = keep every 5th).

        Returns:
            List of images where each item is loaded via `self.imread()`.
        """
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._ensure_bgr3(frame)
            if idx % every_n == 0:
                frame_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
                # Save the frame to disk
                cv2.imwrite(frame_path, frame)
                # Load it back using the standardized reader
                frames.append(self.imread(frame_path))
            idx += 1

        cap.release()
        return frames

    def video_to_csv(
        self,
        video_path: str,
        out_dir: Optional[str] = None,
        every_n: int = 1,
        score_thr: Optional[float] = None,
    ) -> str:
        """
        Process `video_path`, count detections per frame, and write a CSV named
        `<video_name>.csv` with lines `<timestamp>,<count_boxes>`.

        Args:
            video_path: Input video file path.
            out_dir: Where to write the CSV (defaults to the video's directory).
            every_n: Sample every N-th frame.
            score_thr: Optional override of score threshold for counting.

        Returns:
            Path to the written CSV file.
        """
        # Determine output CSV path
        base = os.path.splitext(os.path.basename(video_path))[0]
        target_dir = out_dir or os.path.dirname(video_path)
        os.makedirs(target_dir, exist_ok=True)
        csv_path = os.path.join(target_dir, f"{base}.csv")

        # Open video and read FPS
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # sane default if FPS is missing

        # Prepare CSV writer (no header, per user spec)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % every_n != 0:
                    idx += 1
                    continue

                frame = self._ensure_bgr3(frame)

                # Inference and counting
                if score_thr is not None:
                    # Temporarily override threshold
                    old_thr = self.score_thr
                    self.score_thr = score_thr
                    try:
                        pred = self.predict(frame)
                    finally:
                        self.score_thr = old_thr
                else:
                    pred = self.predict(frame)

                count = int(len(pred["boxes"]))
                timestamp = idx / fps

                # Write `<timestamp>,<count_boxes>`
                scores_list = [f"{s:.4f}" for s in pred.get("scores", [])]
                writer.writerow([f"{timestamp:.6f}", str(count), ";".join(scores_list)])

                idx += 1

        cap.release()
        return csv_path


    def annotate_video(
        self,
        video_path: str,
        out_path: str,
        score_thr: Optional[float] = None,
        font_scale: float = 0.8,
        font_thickness: int = 2,
        every_n: int = 1,
        codec: str = "mp4v",
    ) -> str:
        """
        Regenerate a video with bounding boxes drawn (using `draw`) and a count
        of detections overlaid at the bottom-right corner of each frame.

        Args:
            video_path: Path to the input video file.
            out_path: Output video path (e.g., "out/annotated.mp4").
            score_thr: Optional temporary score threshold override.
            font_scale: OpenCV font scale for count overlay.
            font_thickness: OpenCV font thickness for count overlay.
            every_n: Run detection on every Nth frame; intermediate frames will
                     re-use the last prediction to keep FPS up. Use 1 to detect
                     on every frame.
            codec: FourCC string for the video writer (default "mp4v").

        Returns:
            The `out_path` written.
        """
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #if width is None or height is None or width <= 0 or height <= 0:
            #width, height = 640, 480  # sane defaults
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

        # For re-using predictions on skipped frames
        last_pred: Optional[Dict[str, Any]] = None

        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self._ensure_bgr3(frame)

                # Run detection based on sampling policy
                run_det = (idx % every_n == 0) or (last_pred is None)
                if run_det:
                    if score_thr is not None:
                        # Temporarily override threshold
                        old_thr = self.score_thr
                        self.score_thr = score_thr
                        try:
                            last_pred = self.predict(frame)
                        finally:
                            self.score_thr = old_thr
                    else:
                        last_pred = self.predict(frame)

                # Draw boxes using our drawer
                vis = self.draw(frame, pred=last_pred, score_thr=None)

                # Overlay count at bottom-right
                count = 0 if last_pred is None else int(len(last_pred["boxes"]))
                label = f"count: {count}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                x = width - 10 - tw
                y = height - 10
                # Background box for readability
                cv2.rectangle(vis, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
                cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                writer.write(vis)
                idx += 1
        finally:
            cap.release()
            writer.release()

        return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="MMDetection bbox detector utilities")
    ap.add_argument("--config", default="configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py")
    ap.add_argument("--ckpt", default="checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--thr", type=float, default=0.5, help="score threshold")

    # Modes
    ap.add_argument("--img", help="Path to an image to annotate")
    ap.add_argument("--video", help="Path to a video to process")
    ap.add_argument("--out", help="Output path (image or video)")
    ap.add_argument("--csv", action="store_true", help="When --video is set, also write <video>.csv (timestamp,count)")
    ap.add_argument("--every-n", type=int, default=1, help="sample every N frames for video ops")

    args = ap.parse_args()

    det = MmdetBBoxDetector(args.config, args.ckpt, device=args.device, score_thr=args.thr)

    # Image mode
    if args.img:
        img = det.imread(args.img)
        vis = det.draw(img)
        out_img = args.out or "output.jpg"
        cv2.imwrite(out_img, vis)
        print("Wrote image:", out_img)

    # Video mode
    if args.video:
        # CSV output
        if args.csv:
            csv_dir = os.path.dirname(args.out) if args.out else None
            csv_path = det.video_to_csv(args.video, out_dir=csv_dir, every_n=args.every_n, score_thr=args.thr)
            print("Wrote CSV:", csv_path)
        # Annotated video
        if args.out:
            det.annotate_video(args.video, args.out, score_thr=args.thr, every_n=args.every_n)
            print("Wrote video:", args.out)
        elif not args.csv:
            print("[info] --video provided but neither --out nor --csv specified; nothing to write.")

if __name__ == "__main__":
    main()