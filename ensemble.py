from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import cv2
import os
import numpy as np


BASE = r"C:\EnsembleProject"

MODEL_DIR = os.path.join(BASE, "Models")
TEST_DIR = os.path.join(BASE, "test_images")
OUT_DIR = os.path.join(BASE, "Output")



CONF = 0.25
IOU = 0.55

os.makedirs(OUT_DIR, exist_ok=True)

# Load all models
models = []
for f in os.listdir(MODEL_DIR):
    if f.endswith(".pt"):
        models.append(YOLO(os.path.join(MODEL_DIR, f)))
        print("Loaded model:", f)

def xyxy_to_norm(box, w, h):
    x1,y1,x2,y2 = box
    return [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]

def norm_to_xyxy(box, w, h):
    cx,cy,bw,bh = box
    cx*=w; cy*=h; bw*=w; bh*=h
    return [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]

# Process images
for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(TEST_DIR, img_name)
    img0 = cv2.imread(img_path)
    H, W = img0.shape[:2]

    boxes_all, scores_all, labels_all = [], [], []

    for m in models:
        r = m.predict(img_path, conf=CONF, verbose=False)[0]

        if len(r.boxes) == 0:
            boxes_all.append([])
            scores_all.append([])
            labels_all.append([])
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)

        norm_boxes = [xyxy_to_norm(b, W, H) for b in boxes]

        boxes_all.append(norm_boxes)
        scores_all.append(scores.tolist())
        labels_all.append(labels.tolist())

    if sum(len(x) for x in boxes_all) == 0:
        cv2.imwrite(os.path.join(OUT_DIR, img_name), img0)
        continue

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_all, scores_all, labels_all,
        iou_thr=IOU, skip_box_thr=CONF
    )

    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1,y1,x2,y2 = map(int, norm_to_xyxy(box, W, H))
        cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img0, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(os.path.join(OUT_DIR, img_name), img0)

print("\n🎉 Ensemble complete! Check the output folder.")