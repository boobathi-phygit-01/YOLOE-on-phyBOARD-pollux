"""
YOLOE-11s-seg INT8 TFLite Webcam Inference (tflite_runtime + optional NPU)

Model:
    yoloe-11s-seg_full_integer_quant.tflite

Expected model:
    Input:
        [1, 640, 640, 3] int8

    Detection output:
        [1, 37, 8400] int8

        0:4   -> bbox
        4     -> crocodile confidence
        5:37  -> 32 mask coefficients

    Mask prototype output:
        [1, 160, 160, 32] int8

Usage:
    # CPU
    python3 webcam_inference.py

    # NPU (VeriSilicon VX delegate)
    python3 main.py --npu

    # NPU with custom delegate path
    python3 main.py --npu --delegate /usr/lib/libvx_delegate.so

    python3 main.py \
        --model yoloe-11s-seg_saved_model/yoloe-11s-seg_full_integer_quant.tflite \
        --cam 0 --conf 0.30 --iou 0.45 --npu

Keys:
    q / ESC -> quit
    s       -> save current frame
"""

import argparse
import time

import cv2
import numpy as np

# ------------------------------------------------------------
# tflite_runtime (falls back to full TensorFlow if not installed)
# ------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
    print("[Warning] tflite_runtime not found, using tensorflow.lite")


# ============================================================
# Configuration
# ============================================================

CLASS_NAME = "crocodile"

DEFAULT_DELEGATE = "/usr/lib/libvx_delegate.so"

np.random.seed(42)

MASK_COLOR = (
    int(np.random.randint(0, 255)),
    int(np.random.randint(0, 255)),
    int(np.random.randint(0, 255)),
)


# ============================================================
# TFLite model loading
# ============================================================

def create_interpreter(model_path, use_npu=False,
                       delegate_path=DEFAULT_DELEGATE, num_threads=4):
    """
    Create a tflite_runtime Interpreter.
    If use_npu is True, load the VX delegate; on failure fall back to CPU.
    """

    if use_npu:
        try:
            print(f"[NPU] Loading delegate: {delegate_path}")
            delegate = tflite.load_delegate(delegate_path)

            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate]
            )

            print("[NPU] Delegate loaded, running on NPU")
            return interpreter, True

        except Exception as e:
            print(f"[NPU] Failed to load delegate: {e}")
            print("[NPU] Falling back to CPU")

    interpreter = tflite.Interpreter(
        model_path=model_path,
        num_threads=num_threads
    )

    print(f"[CPU] Running on CPU ({num_threads} threads)")

    return interpreter, False


def load_model(model_path, use_npu=False, delegate_path=DEFAULT_DELEGATE):

    print("\n========================================")
    print("Loading TFLite model")
    print("========================================")
    print(model_path)

    interpreter, npu_active = create_interpreter(
        model_path,
        use_npu,
        delegate_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise RuntimeError(
            f"Expected 1 input, found {len(input_details)}"
        )

    inp = input_details[0]

    print("\n[INPUT]")
    print(" index :", inp["index"])
    print(" name  :", inp["name"])
    print(" shape :", inp["shape"])
    print(" dtype :", inp["dtype"])
    print(" quant :", inp["quantization"])

    # --------------------------------------------------------
    # Find detection and prototype outputs by shape
    # --------------------------------------------------------

    det_out = None
    proto_out = None

    print("\n[OUTPUTS]")

    for out in output_details:

        print(
            f" index={out['index']} "
            f"name={out['name']} "
            f"shape={out['shape']} "
            f"dtype={out['dtype']} "
            f"quant={out['quantization']}"
        )

        shape = tuple(out["shape"])

        # Detection output [1, 37, 8400]
        if (
            len(shape) == 3
            and shape[0] == 1
            and shape[1] == 37
            and shape[2] == 8400
        ):
            det_out = out

        # Segmentation prototype output [1, 160, 160, 32]
        elif (
            len(shape) == 4
            and shape[0] == 1
            and shape[1] == 160
            and shape[2] == 160
            and shape[3] == 32
        ):
            proto_out = out

    if det_out is None:
        raise RuntimeError(
            "\nCould not find YOLOE detection output.\n"
            "Expected: [1, 37, 8400]"
        )

    if proto_out is None:
        raise RuntimeError(
            "\nCould not find YOLOE segmentation prototype output.\n"
            "Expected: [1, 160, 160, 32]"
        )

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    in_scale, in_zp = inp["quantization"]
    det_scale, det_zp = det_out["quantization"]
    proto_scale, proto_zp = proto_out["quantization"]

    input_size = int(inp["shape"][1])

    print("\n========================================")
    print("Model configuration")
    print("========================================")
    print(f"Backend           : {'NPU' if npu_active else 'CPU'}")
    print(f"Input size        : {input_size}")
    print(f"Input dtype       : {inp['dtype']}")
    print(f"Input quant       : {in_scale}, {in_zp}")
    print(f"Detection output  : {det_out['shape']}")
    print(f"Detection quant   : {det_scale}, {det_zp}")
    print(f"Prototype output  : {proto_out['shape']}")
    print(f"Prototype quant   : {proto_scale}, {proto_zp}")

    return (
        interpreter,
        inp,
        det_out,
        proto_out,
        input_size,
        (in_scale, in_zp),
        (det_scale, det_zp),
        (proto_scale, proto_zp),
        npu_active,
    )


# ============================================================
# Dequantization
# ============================================================

def dequantize(tensor, scale, zero_point):

    return (tensor.astype(np.float32) - zero_point) * scale


# ============================================================
# Letterbox
# ============================================================

def letterbox(image, size=640):

    h, w = image.shape[:2]

    scale = size / max(h, w)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2

    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    output = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return output, (pad_left, pad_top, scale, new_w, new_h)


# ============================================================
# Preprocessing
# ============================================================

def preprocess(frame, input_size, input_dtype, input_scale, input_zp):

    image, pad_info = letterbox(frame, input_size)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_float = image.astype(np.float32) / 255.0

    # Quantize according to actual model parameters
    if input_dtype == np.int8:

        tensor = np.clip(
            image_float / input_scale + input_zp,
            -128,
            127
        ).round().astype(np.int8)

    elif input_dtype == np.uint8:

        tensor = np.clip(
            image_float / input_scale + input_zp,
            0,
            255
        ).round().astype(np.uint8)

    else:

        tensor = image_float.astype(np.float32)

    tensor = np.expand_dims(tensor, axis=0)

    return tensor, pad_info


# ============================================================
# YOLOE detection post-processing
# ============================================================

def postprocess_detection(
    raw_detection,
    pad_info,
    original_shape,
    detection_scale,
    detection_zp,
    confidence_threshold,
    iou_threshold,
    input_size
):
    """
    Detection tensor: [1, 37, 8400]

        0:4   = x, y, w, h  (normalized)
        4     = class score
        5:37  = mask coefficients
    """

    empty = (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0, 32), dtype=np.float32)
    )

    output = dequantize(
        raw_detection,
        detection_scale,
        detection_zp
    )[0]

    # output: [37, 8400]

    boxes_xywh = output[0:4, :].T
    scores = output[4, :]
    mask_coefficients = output[5:37, :].T

    # This YOLOE export uses normalized coordinates
    boxes_xywh = boxes_xywh * input_size

    # Confidence filtering
    keep = scores >= confidence_threshold

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    mask_coefficients = mask_coefficients[keep]

    if len(scores) == 0:
        return empty

    # xywh -> xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Remove letterbox
    pad_left, pad_top, scale, _, _ = pad_info

    original_h, original_w = original_shape[:2]

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

    # Clip boxes
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_h)

    # NMS (cv2 expects x, y, w, h)
    nms_boxes = [
        [
            float(b[0]),
            float(b[1]),
            float(b[2] - b[0]),
            float(b[3] - b[1])
        ]
        for b in boxes_xyxy
    ]

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        scores.tolist(),
        confidence_threshold,
        iou_threshold
    )

    if len(indices) == 0:
        return empty

    indices = np.array(indices).reshape(-1)

    boxes_xyxy = boxes_xyxy[indices]
    scores = scores[indices]
    mask_coefficients = mask_coefficients[indices]

    class_ids = np.zeros(len(scores), dtype=np.int32)

    return boxes_xyxy, scores, class_ids, mask_coefficients


# ============================================================
# Sigmoid
# ============================================================

def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# Generate segmentation mask
# ============================================================

def generate_mask(
    mask_coefficients,
    prototypes,
    box,
    original_shape,
    pad_info,
    input_size
):
    """
    mask_coefficients: [32]
    prototypes:        [160, 160, 32]
    """

    # [160,160,32] @ [32] -> [160,160]
    mask = np.matmul(prototypes, mask_coefficients)

    mask = sigmoid(mask)

    # Prototype resolution -> input resolution
    mask = cv2.resize(
        mask,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Remove letterbox
    pad_left, pad_top, scale, new_w, new_h = pad_info

    mask = mask[
        pad_top:pad_top + new_h,
        pad_left:pad_left + new_w
    ]

    original_h, original_w = original_shape[:2]

    # Resize to original camera frame
    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = mask > 0.5

    # Restrict mask to bounding box
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, original_w))
    y1 = max(0, min(y1, original_h))
    x2 = max(0, min(x2, original_w))
    y2 = max(0, min(y2, original_h))

    box_mask = np.zeros((original_h, original_w), dtype=np.bool_)

    if x2 > x1 and y2 > y1:
        box_mask[y1:y2, x1:x2] = True

    mask &= box_mask

    return mask


# ============================================================
# Draw segmentation
# ============================================================

def draw_results(frame, boxes, scores, masks, fps, backend):

    output = frame.copy()

    # Mask overlay
    if len(masks) > 0:

        overlay = frame.copy()

        combined_mask = np.zeros(frame.shape[:2], dtype=np.bool_)

        for mask in masks:
            overlay[mask] = MASK_COLOR
            combined_mask |= mask

        output[combined_mask] = cv2.addWeighted(
            frame[combined_mask],
            0.55,
            overlay[combined_mask],
            0.45,
            0
        ).reshape(-1, 3)

    # Boxes + labels
    for box, score in zip(boxes, scores):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), MASK_COLOR, 2)

        label = f"{CLASS_NAME} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        label_y = max(y1, th + 8)

        cv2.rectangle(
            output,
            (x1, label_y - th - 8),
            (x1 + tw + 6, label_y),
            MASK_COLOR,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 3, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # HUD
    cv2.putText(
        output,
        f"FPS: {fps:.1f} [{backend}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Crocodiles: {len(boxes)}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return output


# ============================================================
# Camera
# ============================================================

def open_camera(cam_id=0, width=640, height=480):
    """
    Open a USB webcam via V4L2 with MJPG (bypasses OpenCV's GStreamer
    backend, which fails to negotiate 1280x720 on raw formats).
    Falls back to an explicit GStreamer pipeline if V4L2 gives no frames.
    """

    device = f"/dev/video{cam_id}"

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = cap.read()
        if ok:
            print(f"[Camera] V4L2 MJPG {width}x{height} on {device}")
            return cap

        print("[Camera] V4L2 gave no frames, trying GStreamer pipeline")
        cap.release()

    pipeline = (
        f"v4l2src device={device} ! "
        f"image/jpeg,width={width},height={height},framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            print(f"[Camera] GStreamer MJPG {width}x{height} on {device}")
            return cap
        cap.release()

    raise RuntimeError(f"Cannot read frames from {device}")


# ============================================================
# Main inference loop
# ============================================================

def run(
    model_path,
    cam_id=0,
    confidence_threshold=0.25,
    iou_threshold=0.45,
    use_npu=False,
    delegate_path=DEFAULT_DELEGATE
):

    (
        interpreter,
        inp,
        det_out,
        proto_out,
        input_size,
        (input_scale, input_zp),
        (det_scale, det_zp),
        (proto_scale, proto_zp),
        npu_active
    ) = load_model(model_path, use_npu, delegate_path)

    backend = "NPU" if npu_active else "CPU"

    # --------------------------------------------------------
    # Warm-up (NPU compiles its graph on first invoke, which
    # can take several seconds; do it before the live loop)
    # --------------------------------------------------------

    print("\n[Warm-up] running dummy inference...")

    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])

    interpreter.set_tensor(inp["index"], dummy)

    t0 = time.perf_counter()
    interpreter.invoke()
    print(f"[Warm-up] done in {time.perf_counter() - t0:.2f}s")

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = open_camera(cam_id)

    print(f"\n[Camera] opened (id={cam_id})")
    print("Press q / ESC to quit")
    print("Press s to save frame")

    # FPS smoothing
    fps_avg = 0.0
    alpha = 0.1

    read_fails = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            read_fails += 1
            if read_fails > 30:
                raise RuntimeError("Camera stopped delivering frames")
            continue

        read_fails = 0

        start_time = time.perf_counter()

        # Preprocess
        tensor, pad_info = preprocess(
            frame,
            input_size,
            inp["dtype"],
            input_scale,
            input_zp
        )

        # Inference
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()

        raw_detection = interpreter.get_tensor(det_out["index"])
        raw_prototypes = interpreter.get_tensor(proto_out["index"])

        # Detection post-processing
        (
            boxes,
            scores,
            class_ids,
            mask_coefficients
        ) = postprocess_detection(
            raw_detection,
            pad_info,
            frame.shape,
            det_scale,
            det_zp,
            confidence_threshold,
            iou_threshold,
            input_size
        )

        # Prototypes: [160,160,32]
        prototypes = dequantize(
            raw_prototypes,
            proto_scale,
            proto_zp
        )[0]

        # Masks
        masks = [
            generate_mask(
                coeff,
                prototypes,
                box,
                frame.shape,
                pad_info,
                input_size
            )
            for coeff, box in zip(mask_coefficients, boxes)
        ]

        # FPS
        elapsed = time.perf_counter() - start_time

        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        fps_avg = fps if fps_avg == 0 else (
            alpha * fps + (1.0 - alpha) * fps_avg
        )

        # Draw + display
        output = draw_results(frame, boxes, scores, masks, fps_avg, backend)

        cv2.imshow("YOLOE-11s Seg INT8 TFLite", output)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("s"):
            filename = "crocodile_capture.jpg"
            cv2.imwrite(filename, output)
            print(f"[Saved] {filename}")

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLOE-11s Segmentation INT8 TFLite Webcam Inference"
    )

    parser.add_argument(
        "--model",
        default=(
            "yoloe-11s-seg_saved_model/"
            "yoloe-11s-seg_full_integer_quant.tflite"
        ),
        help="Path to INT8 TFLite model"
    )

    parser.add_argument("--cam", type=int, default=0,
                        help="Camera device ID")

    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")

    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")

    parser.add_argument("--npu", action="store_true",
                        help="Run on NPU using the VX delegate")

    parser.add_argument("--delegate", default=DEFAULT_DELEGATE,
                        help=f"Path to NPU delegate (default: {DEFAULT_DELEGATE})")

    args = parser.parse_args()

    run(
        args.model,
        args.cam,
        args.conf,
        args.iou,
        args.npu,
        args.delegate
    )
