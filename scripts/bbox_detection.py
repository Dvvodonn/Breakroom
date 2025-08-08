from __future__ import annotations
import numpy as np
import os
import csv
from typing import Union, Dict, Any, Optional, List

import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from mmengine.visualization import Visualizer
import cv2


class MmdetBBoxDetector:
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda:0",
        score_thr: float = 0.3,
    ):
        # Load model
        self.model = init_detector(config, checkpoint, device=device)
        self.score_thr = score_thr
        self.classes = getattr(self.model.dataset_meta, "classes", None)

        # Prepare visualizer
        self.visualizer = Visualizer(name="mmdet-vis", vis_backends=[], save_dir=None)
        if self.classes is not None:
            self.visualizer.set_dataset_meta(dict(classes=self.classes))

    @torch.inference_mode()
    def predict(self, img: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Run inference and return boxes, scores, labels."""
        result: DetDataSample = inference_detector(self.model, img)
        inst = result.pred_instances

        bboxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()

        keep = scores >= self.score_thr
        return {
            "boxes": bboxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "detsample": result,
        }

    def draw(
        self,
        img: np.ndarray,
        pred: Optional[Dict[str, Any]] = None,
        score_thr: Optional[float] = None,
    ) -> np.ndarray:
        """Draw bounding boxes on an image."""
        if pred is None:
            pred = self.predict(img)

        result = pred["detsample"].clone()
        thr = self.score_thr if score_thr is None else score_thr
        inst = result.pred_instances
        keep = inst.scores >= thr
        inst.bboxes = inst.bboxes[keep]
        inst.scores = inst.scores[keep]
        inst.labels = inst.labels[keep]

        self.visualizer.set_image(img[:, :, ::-1])  # BGR→RGB
        self.visualizer.draw_bboxes(
            inst.bboxes.numpy(),
            labels=[
                f"{self.classes[l]} {s:.2f}"
                for l, s in zip(inst.labels.numpy(), inst.scores.numpy())
            ],
        )
        out_rgb = self.visualizer.get_image()
        return out_rgb[:, :, ::-1]  # RGB→BGR

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
                writer.writerow([f"{timestamp:.6f}", str(count)])

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
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
                vis = self.draw(frame, pred=last_pred, score_thr=score_thr)

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


if __name__ == "__main__":
    # Example usage
    config = "configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py"
    checkpoint = "checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"

    det = MmdetBBoxDetector(config, checkpoint, device="cuda:0", score_thr=0.5)

    csv_path = det.video_to_csv(
    video_path="data/Breakroom_f.mp4",
    out_dir="data/csv",      # or None to write next to the video
    every_n=1,               # sample every frame
    score_thr=0.8            # optional override
    )
    print("Wrote:", csv_path)
    out_path = det.annotate_video(
    video_path="data/test.mp4",
    out_path="data/out/test_annotated.mp4",
    score_thr=0.4,    # optional
    every_n=1,        # detect every frame; increase to speed up
    codec="mp4v"      # or "avc1", "XVID" depending on your system
    )
    print("Wrote:", out_path)