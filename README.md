# YOLOE-11s-seg INT8 on phyBOARD-Pollux (i.MX8M Plus)

Real-time **text-prompted instance segmentation** (example prompt: `crocodile`) running on an embedded board with a quantized INT8 TFLite model, using a USB webcam and, optionally, the on-chip NPU through the VX delegate.

```
 PC (export)                                   Board (inference)
┌──────────────────────────┐   copy .tflite   ┌─────────────────────────────────────┐
│ yoloe-11s-seg.pt         │ ───────────────► │ USB cam → letterbox → INT8 quantize │
│ + text prompt(s)         │                  │ → TFLite (CPU / NPU) → NMS → masks  │
│ → INT8 TFLite export     │                  │ → overlay + FPS on screen           │
└──────────────────────────┘                  └─────────────────────────────────────┘
 convert_yolov8n_tflite.py                                   main.py
```

---

## 1. What is YOLOE?

**YOLOE** ("Real-Time Seeing Anything") is an open-vocabulary object detection and instance segmentation model from the YOLO family, available through the Ultralytics package. Classic YOLO models can only detect the fixed set of classes they were trained on (for example the 80 COCO classes). YOLOE can detect and segment **arbitrary categories that you describe**, while keeping YOLO-level speed.

It supports three ways of telling the model what to find:

| Mode | You provide | Typical use |
|------|-------------|-------------|
| **Text prompt** | Class names such as `["crocodile"]` | Used in this project |
| **Visual prompt** | A reference image / box of the object | When words are not enough |
| **Prompt-free** | Nothing | Model names objects from a built-in vocabulary |

### Why it suits edge deployment

- With text prompts, the text encoder (CLIP-style) is only needed **once, at export time**. The resulting text embeddings are folded into the detection head as fixed weights, so at runtime the model behaves like an ordinary YOLO segmentation network with **no text encoder and no extra cost**.
- That makes it exportable to TFLite and quantizable to INT8, which is what the NPU on the i.MX8M Plus needs.
- The `11s` variant is the small model (YOLO11-small backbone), a good size/accuracy trade-off for a board with 1.89 GB RAM.

### What the exported model outputs

Because only one class is baked in, the network has two outputs:

| Output | Shape | Meaning |
|--------|-------|---------|
| Detection | `[1, 37, 8400]` int8 | per anchor: 4 box values (`x, y, w, h`, normalized) + 1 class score + 32 mask coefficients |
| Mask prototypes | `[1, 160, 160, 32]` int8 | 32 shared mask "basis" images |

Input is `[1, 640, 640, 3]` int8.

A final mask for one object is computed as `sigmoid(prototypes · coefficients)`, which is the standard YOLO-seg approach.

---

## 2. Repository contents

| File | Role |
|------|------|
| `convert_yolov8n_tflite.py` | Runs on a **PC**. Loads `yoloe-11s-seg.pt`, bakes in the text prompt, exports a full-integer INT8 TFLite model. (The filename says yolov8n, but the script exports YOLOE-11s-seg.) |
| `main.py` | Runs on the **board**. Webcam capture, preprocessing, TFLite inference (CPU or NPU), post-processing, drawing. |

---

## 3. Step 1: Convert the model (on a PC)

```bash
pip install ultralytics tensorflow    # plus the export dependencies Ultralytics asks for
python3 convert_yolov8n_tflite.py
```

What the script does:

1. `YOLOE("yoloe-11s-seg.pt")` loads the pretrained weights.
2. `names = ["crocodile"]` is the class you want. Change this to detect something else.
3. `model.set_classes(names, model.get_text_pe(names))` computes the text embeddings and bakes them into the model. **This must happen before export.**
4. `model.export(format="tflite", int8=True, imgsz=640, data="coco128-seg.yaml")` exports and quantizes. The 128 COCO images are used as **calibration data** so the INT8 ranges are estimated well (a tiny set like coco8 gives noticeably worse quantization).

Output file:

```
yoloe-11s-seg_saved_model/yoloe-11s-seg_full_integer_quant.tflite
```

> If you change the prompt to more than one class, the detection output becomes `[1, 4 + N + 32, 8400]` and `main.py` must be adapted (it currently expects exactly `[1, 37, 8400]` and a single class).

---

## 4. Step 2: Deploy on the board

### Tested hardware and software

| Item | Value |
|------|-------|
| Board | phyBOARD-Pollux (NXP i.MX8M Plus) |
| RAM | 1.89 GB |
| Image | PHYTEC BSP-Yocto-CelebrityFaceMatch-i.MX8MP **v0.7** ([release notes](https://download.phytec.de/Software/Linux/BSP-Yocto-CelebrityFaceMatch/BSP-Yocto-CelebrityFaceMatch-i.MX8MP-v0.7/ReleaseNotes)) |
| Camera | USB webcam (V4L2, MJPG) |
| NPU delegate | `/usr/lib/libvx_delegate.so` (VeriSilicon VX delegate) |

### Prerequisites on the board

Python 3 with `numpy`, `opencv-python` (cv2), and `tflite_runtime`. Check what the image already provides:

```bash
python3 -c "import cv2, numpy, tflite_runtime; print('ok')"
ls /usr/lib/libvx_delegate.so
ls /dev/video*
```

If `tflite_runtime` is missing, `main.py` falls back to `tensorflow.lite` (much heavier, so prefer `tflite_runtime`). A display session is needed for `cv2.imshow` (for example Weston/Wayland, or a connected screen).

### Copy files to the board

```bash
scp -r yoloe-11s-seg_saved_model main.py root@<board-ip>:/home/root/
```

### Run

```bash
# CPU (4 threads)
python3 main.py

# NPU through the VX delegate
python3 main.py --npu

# NPU with custom delegate path
python3 main.py --npu --delegate /usr/lib/libvx_delegate.so

# Full example
python3 main.py \
    --model yoloe-11s-seg_saved_model/yoloe-11s-seg_full_integer_quant.tflite \
    --cam 0 --conf 0.30 --iou 0.45 --npu
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `yoloe-11s-seg_saved_model/yoloe-11s-seg_full_integer_quant.tflite` | INT8 TFLite model |
| `--cam` | `0` | Camera index (`/dev/videoN`) |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--npu` | off | Use the NPU via the VX delegate; falls back to CPU if loading fails |
| `--delegate` | `/usr/lib/libvx_delegate.so` | Delegate library path |

Keys: **q / ESC** quit, **s** save the current annotated frame as `crocodile_capture.jpg`.

---

## 5. Code workflow (`main.py`)

```
open camera ─► load model ─► warm-up ─► loop:
  capture → letterbox → quantize → invoke → decode boxes → NMS → masks → draw → show
```

1. **Model loading** (`create_interpreter`, `load_model`)
   Creates a `tflite.Interpreter`. With `--npu` it loads the VX delegate; if that fails it prints a message and falls back to CPU. It then finds the two outputs **by shape** (`[1,37,8400]` and `[1,160,160,32]`) and reads the quantization `(scale, zero_point)` of the input and both outputs.

2. **Warm-up**
   One dummy `invoke()` before the live loop. On the NPU the first call compiles the graph and can take several seconds, so it is done up front.

3. **Camera** (`open_camera`)
   Opens `/dev/videoN` through V4L2 with MJPG at 640x480 and a buffer size of 1 (low latency). If that gives no frames, it tries an explicit GStreamer pipeline.

4. **Preprocessing** (`letterbox`, `preprocess`)
   Resize keeping aspect ratio and pad to 640x640 with gray (114), convert BGR to RGB, scale to 0..1, then quantize with the model's input scale and zero point to int8.

5. **Inference**
   `set_tensor` → `invoke` → read the detection tensor and the prototype tensor.

6. **Detection post-processing** (`postprocess_detection`)
   Dequantize, split into boxes / score / 32 mask coefficients, convert normalized `xywh` to pixels, filter by confidence, convert to `xyxy`, undo the letterbox padding and scale back to the camera frame, clip, then run `cv2.dnn.NMSBoxes`.

7. **Mask generation** (`generate_mask`)
   For each kept detection: `sigmoid(prototypes @ coefficients)` → resize 160→640 → crop away the letterbox padding → resize to the camera frame → threshold at 0.5 → restrict to the bounding box.

8. **Drawing** (`draw_results`)
   Blends the mask color over the frame, draws boxes with `crocodile <score>` labels, and shows a smoothed FPS counter, the backend (CPU/NPU) and the object count.

---

## 6. Performance

Measured on the phyBOARD-Pollux with the setup above:

| Metric | Result |
|--------|--------|
| Throughput | **~3.5 FPS** |
| RAM usage | **~30 %** of 1.89 GB (about 0.57 GB) |

The FPS figure is the end-to-end loop (preprocess, inference, post-processing, mask generation, drawing), not the bare model latency. A large part of the per-frame time is the NumPy/OpenCV work on the CPU cores.

### Ideas to go faster

- Make sure `--npu` is active (the HUD shows `[NPU]` or `[CPU]`; look for `[NPU] Delegate loaded` in the log).
- Mask generation runs for each detection at full camera resolution. Limiting detections, using a higher `--conf`, or computing masks only inside the box region reduces cost.
- Lower the camera resolution or skip frames if you only need periodic results.
- Run camera capture in its own thread so it overlaps with inference.

---

## 7. Troubleshooting

| Problem | What to check |
|---------|---------------|
| `Failed to load delegate` | Does `/usr/lib/libvx_delegate.so` exist? Pass the correct path with `--delegate`. The script continues on CPU. |
| `Could not find YOLOE detection output` | The model was exported with a different number of classes or a different input size. Re-export with exactly one prompt and `imgsz=640`. |
| `Cannot read frames from /dev/videoN` | Try another `--cam` index; check `ls /dev/video*` and that the camera supports MJPG. |
| First frame is very slow with `--npu` | Normal: graph compilation during warm-up. |
| No window appears | A display server must be running and `DISPLAY` / Wayland variables set for the session. |
| Poor detections | Raise calibration quality (more images), tune `--conf`, or try a more descriptive prompt such as `"green crocodile"` at export. |

---

## 8. Credits

- YOLOE and the Ultralytics export tooling: [Ultralytics](https://docs.ultralytics.com/models/yoloe/)
- Board and BSP: [PHYTEC](https://www.phytec.com/)
- NPU delegate: VeriSilicon VX delegate for NXP i.MX8M Plus
