from __future__ import annotations
import numpy as np
import os
import csv
from typing import Union, Dict, Any, Optional, List

import subprocess
import shutil
import tempfile

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
        force_transcode: bool = False,
        fast_transcode: bool = False,
        use_nvenc: Optional[bool] = None,
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
        self.force_transcode = force_transcode
        self.fast_transcode = fast_transcode
        self.use_nvenc = use_nvenc
        # Pull class names from model metainfo if available
        self.classes = []
        model = getattr(self.infer, 'model', None)
        if model is not None and getattr(model, 'dataset_meta', None):
            classes = model.dataset_meta.get('classes', None)
            if classes is not None:
                self.classes = list(classes)

    # --- Robust video preparation helpers ---------------------------------
    def _nvenc_available(self) -> bool:
        if not shutil.which('ffmpeg'):
            return False
        try:
            out = subprocess.check_output(['ffmpeg','-hide_banner','-encoders'], stderr=subprocess.STDOUT, text=True)
            return 'h264_nvenc' in out
        except Exception:
            return False
    def _probe_pix_fmt(self, path: str) -> Optional[str]:
        """Return pixel format via ffprobe if available, else None."""
        if not shutil.which('ffprobe'):
            return None
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=pix_fmt,width,height,r_frame_rate',
                '-of', 'default=noprint_wrappers=1', path
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            for line in out.splitlines():
                if line.startswith('pix_fmt='):
                    return line.split('=', 1)[1].strip()
        except Exception:
            return None
        return None
    def _ffprobe_stream_info(self, path: str) -> dict:
        """Return dict with pix_fmt, width, height, fps_str, fps (float) using ffprobe when available."""
        
        info = {'pix_fmt': None, 'width': None, 'height': None, 'fps_str': None, 'fps': None}
        if not shutil.which('ffprobe'):
            return info
        try:
            cmd = [
                'ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=pix_fmt,width,height,r_frame_rate',
                '-of','default=noprint_wrappers=1', path
            ]
            txt = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            for line in txt.splitlines():
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                v = v.strip()
                if k == 'pix_fmt':
                    info['pix_fmt'] = v
                elif k == 'width':
                    try: info['width'] = int(v)
                    except Exception: pass
                elif k == 'height':
                    try: info['height'] = int(v)
                    except Exception: pass
                elif k == 'r_frame_rate':
                    info['fps_str'] = v
            # parse fps
            fps_str = info['fps_str']
            if fps_str:
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    try:
                        num = float(num); den = float(den)
                        if den == 0: den = 1.0
                        info['fps'] = num/den
                    except Exception:
                        pass
                else:
                    try:
                        info['fps'] = float(fps_str)
                    except Exception:
                        pass
        except Exception:
            pass
        return info
    
    def _transcode_to_yuv420_ffmpeg(self, src_path: str) -> Optional[str]:
        """Use ffmpeg to transcode a video to yuv420p with even dims. Returns temp path or None on failure.
        Supports fast mode and NVENC acceleration with fallback."""
        if not shutil.which('ffmpeg'):
            print('[ffmpeg] not found on PATH')
            return None
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Try to keep fps; fall back to ffmpeg default if probe fails
        fps = None
        if shutil.which('ffprobe'):
            try:
                out = subprocess.check_output([
                    'ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=r_frame_rate',
                    '-of','default=nk=1:nw=1', src_path
                ], stderr=subprocess.STDOUT, text=True)
                fps = out.strip()
            except Exception:
                fps = None

        # Base command with two modes: fast vs robust
        analyzeduration = '2M' if getattr(self, 'fast_transcode', False) else '100M'
        probesize = '2M' if getattr(self, 'fast_transcode', False) else '100M'
        ff_base = [
            'ffmpeg', '-hide_banner', '-y',
            '-analyzeduration', analyzeduration, '-probesize', probesize,
            '-fflags', '+genpts+discardcorrupt',
            '-err_detect', 'ignore_err',
        ]

        # Choose encoder: nvenc if requested/available, else libx264
        want_nvenc = self.use_nvenc if self.use_nvenc is not None else True
        enc_is_nvenc = False
        if want_nvenc and self._nvenc_available():
            enc_is_nvenc = True

        vf_filter = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
        # Use NVIDIA scaler for speed if nvenc path
        if enc_is_nvenc:
            # scale_npp is fast; if missing it will error and we will fall back
            vf_nv = f'scale_npp=trunc(iw/2)*2:trunc(ih/2)*2:interp_algo=super;format=yuv420p'
            enc_opts = [
                '-hwaccel','cuda',
                '-i', src_path,
                '-vf', vf_nv,
                '-c:v','h264_nvenc',
                '-preset','p1',
                '-tune','ll',
                '-rc','constqp','-qp','28',
                '-movflags','+faststart',
            ]
        else:
            enc_opts = [
                '-i', src_path,
                '-vf', vf_filter,
                '-pix_fmt','yuv420p',
                '-vsync','1',
                '-c:v','libx264','-preset','veryfast','-crf','22',
                '-movflags','+faststart',
            ]

        cmd = ff_base + enc_opts
        if fps and fps != '0/0':
            cmd += ['-r', fps]
        cmd += ['-an', tmp_path]

        print('[ffmpeg] transcode start:\n  ' + ' '.join(cmd))
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0 and enc_is_nvenc:
                # Fall back to libx264 if nvenc path failed (e.g., no scale_npp)
                print('[ffmpeg] nvenc path failed; falling back to libx264. stderr head:\n' + (res.stderr or '')[:1000])
                cmd = ff_base + [
                    '-i', src_path,
                    '-vf', vf_filter,
                    '-pix_fmt','yuv420p',
                    '-vsync','1',
                    '-c:v','libx264','-preset','veryfast','-crf','22',
                    '-movflags','+faststart',
                ]
                if fps and fps != '0/0':
                    cmd += ['-r', fps]
                cmd += ['-an', tmp_path]
                print('[ffmpeg] retry transcode (libx264):\n  ' + ' '.join(cmd))
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print('[ffmpeg] transcode failed (returncode=%s). stderr:' % res.returncode)
                if res.stderr:
                    print(res.stderr.strip()[:4000])
                raise RuntimeError('ffmpeg returned non-zero')
        except Exception as e:
            print('[ffmpeg] exception during transcode:', e)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None

        # Verify the output really is decodable and yuv420p
        if shutil.which('ffprobe'):
            try:
                verify = subprocess.check_output([
                    'ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=pix_fmt,width,height',
                    '-of','default=noprint_wrappers=1', tmp_path
                ], stderr=subprocess.STDOUT, text=True)
                ok_pix = None
                for line in verify.splitlines():
                    if line.startswith('pix_fmt='):
                        ok_pix = line.split('=',1)[1].strip().lower()
                        break
                if ok_pix not in ('yuv420p', 'yuvj420p'):
                    print('[ffmpeg] verify failed: unexpected pix_fmt:', ok_pix)
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    return None
            except Exception as e:
                print('[ffmpeg] verify probe failed:', e)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return None

        print('[ffmpeg] transcoded to yuv420p with even dims:', tmp_path)
        return tmp_path


    def _prepare_video_path(self, video_path: str, force: Optional[bool] = None) -> str:
        """Ensure input video is decodable and yuv420p. If not, transcode to a temp file and return its path."""
        if force is None:
            force = getattr(self, 'force_transcode', False)

        # Probe with ffprobe only (no OpenCV). If ffprobe is missing, just force transcode when requested.
        info = self._ffprobe_stream_info(video_path)
        pix = (info.get('pix_fmt') or '').lower()
        w = info.get('width'); h = info.get('height')
        needs_yuv420 = (pix not in ('yuv420p', 'yuvj420p')) if pix else True
        need_even_fix = False
        if isinstance(w, int) and isinstance(h, int):
            need_even_fix = (w % 2 != 0) or (h % 2 != 0)

        print(f"[transcode] probe pix_fmt={pix or 'unknown'} size={w}x{h} force={force} needs_yuv420={needs_yuv420} need_even_fix={need_even_fix}")

        # Decide whether to transcode
        must_transcode = force or needs_yuv420 or need_even_fix
        if not must_transcode:
            print('[transcode] skip: already yuv420p with even dimensions')
            return video_path

        out = self._transcode_to_yuv420_ffmpeg(video_path)
        if out:
            return out

        # If transcode was requested/needed but failed, do NOT silently continue.
        # Raise so the caller sees a clear error instead of swscaler spam.
        if force:
            raise RuntimeError('ffmpeg transcode failed while --force-transcode is set')
        else:
            print('[transcode] WARNING: ffmpeg transcode failed; falling back to original input')
            return video_path


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
        safe_video = self._prepare_video_path(video_path, force=self.force_transcode)
        cap = cv2.VideoCapture(safe_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
        Note: If input video is transcoded, a temp file may be created (OS will clean up or you may remove).

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
        safe_video = self._prepare_video_path(video_path, force=self.force_transcode)
        cap = cv2.VideoCapture(safe_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {safe_video}")
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

        Note: If input video is transcoded, a temp file may be created (OS will clean up or you may remove).

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

        safe_video = self._prepare_video_path(video_path, force=self.force_transcode)
        cap = cv2.VideoCapture(safe_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {safe_video}")

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
    ap.add_argument("--force-transcode", action="store_true", default=False,
                    help="Force re-encode input video to yuv420p before processing (avoids swscale slice errors)")
    ap.add_argument('--fast-transcode', action='store_true', default=False,
                    help='Faster but less defensive transcode (smaller probe sizes).')
    ap.add_argument('--nvenc', dest='use_nvenc', action='store_true', help='Force NVENC if available')
    ap.add_argument('--no-nvenc', dest='use_nvenc', action='store_false', help='Disable NVENC even if available')
    ap.set_defaults(use_nvenc=None)

    # Modes
    ap.add_argument("--img", help="Path to an image to annotate")
    ap.add_argument("--video", help="Path to a video to process")
    ap.add_argument("--out", help="Output path (image or video)")
    ap.add_argument("--csv", action="store_true", help="When --video is set, also write <video>.csv (timestamp,count)")
    ap.add_argument("--every-n", type=int, default=1, help="sample every N frames for video ops")

    args = ap.parse_args()

    det = MmdetBBoxDetector(
        args.config, args.ckpt, device=args.device, score_thr=args.thr,
        force_transcode=bool(getattr(args, 'force_transcode', False)),
        fast_transcode=bool(getattr(args, 'fast_transcode', False)),
        use_nvenc=getattr(args, 'use_nvenc', None),
    )

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