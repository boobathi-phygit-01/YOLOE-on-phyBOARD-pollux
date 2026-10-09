
from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt")

# Define your specific prompts that you want to bake into the model
names = ["crocodile"]

# Set the classes and get text embeddings BEFORE export
model.set_classes(names, model.get_text_pe(names))

print("Re-exporting with coco128 calibration (128 images)...")
export_path = model.export(
    format="tflite",
    int8=True,
    imgsz=640,
    data="coco128-seg.yaml",   # 128 calibration images — much better than coco8 (4 images)
)

print(f"\nExported to: {export_path}")
