from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(0)

# -------- TRACKING --------
prev_tracked = {}
prev_masks_smooth = {}
last_seen = {}  # object_id -> frames since last seen
next_object_id = 0

iou_threshold = 0.5
max_missing_frames = 10   # persistence window
smooth_alpha = 0.7

selected_objects = {}
background_selected = False
current_filter = "blur"

frame_count = 0
skip_frames = 2

masks = {}

# -------- IoU --------
def compute_iou(mask1, mask2):
    m1 = (mask1 > 0.5).astype(np.uint8)
    m2 = (mask2 > 0.5).astype(np.uint8)
    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return intersection / union if union != 0 else 0

# -------- MOUSE --------
def mouse_callback(event, x, y, flags, param):
    global selected_objects, background_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = False

        for obj_id, mask in masks.items():
            if mask[int(y), int(x)] > 0.5:
                clicked = True
                if obj_id in selected_objects:
                    del selected_objects[obj_id]
                else:
                    selected_objects[obj_id] = current_filter
                break

        if not clicked:
            background_selected = not background_selected

cv2.namedWindow("Segmentation")
cv2.setMouseCallback("Segmentation", mouse_callback)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # -------- FILTERS --------
    blurred = cv2.GaussianBlur(original, (15,15), 0)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(cv2.GaussianBlur(original, (5,5), 0), 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # -------- MODEL --------
    if frame_count % skip_frames == 0:
        results = model(frame, imgsz=320, verbose=False)

        current_masks = []

        if results[0].masks is not None:
            raw_masks = results[0].masks.data.cpu().numpy()

            for m in raw_masks:
                m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
                m = cv2.GaussianBlur(m, (5,5), 0)
                current_masks.append(m)

        new_tracked = {}
        matched_ids = set()

        for curr_mask in current_masks:
            best_iou = 0
            best_id = None

            for obj_id, prev_mask in prev_tracked.items():
                iou = compute_iou(curr_mask, prev_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id

            if best_iou > iou_threshold:
                new_tracked[best_id] = curr_mask
                matched_ids.add(best_id)
                last_seen[best_id] = 0
            else:
                new_tracked[next_object_id] = curr_mask
                last_seen[next_object_id] = 0
                next_object_id += 1

        # -------- PERSISTENCE --------
        for obj_id, prev_mask in prev_tracked.items():
            if obj_id not in matched_ids:
                last_seen[obj_id] += 1
                if last_seen[obj_id] < max_missing_frames:
                    new_tracked[obj_id] = prev_mask

        # -------- SMOOTHING --------
        smoothed = {}
        for obj_id, curr_mask in new_tracked.items():
            if obj_id in prev_masks_smooth:
                smooth = smooth_alpha * curr_mask + (1 - smooth_alpha) * prev_masks_smooth[obj_id]
            else:
                smooth = curr_mask
            smoothed[obj_id] = smooth

        prev_masks_smooth = smoothed.copy()
        prev_tracked = smoothed.copy()
        masks = smoothed.copy()

    frame_count += 1

    output = original.copy()

    # -------- BACKGROUND MASK --------
    if masks:
        combined = np.zeros_like(next(iter(masks.values())))
        for m in masks.values():
            combined = np.maximum(combined, m)
        background_mask = combined
    else:
        background_mask = np.zeros(frame.shape[:2], dtype=float)

    # -------- OBJECT FILTERS --------
    for obj_id, filt in selected_objects.items():
        if obj_id not in masks:
            continue

        mask = np.clip(masks[obj_id], 0, 1)
        mask = cv2.GaussianBlur(mask, (11,11), 0)
        mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

        if filt == "blur":
            output = (1 - mask_3) * output + mask_3 * blurred
        elif filt == "gray":
            output = (1 - mask_3) * output + mask_3 * gray
        elif filt == "edge":
            output = (1 - mask_3) * output + mask_3 * edges

    # -------- BACKGROUND FILTER --------
    if background_selected:
        bg_mask = np.clip(1 - background_mask, 0, 1)
        bg_mask = cv2.GaussianBlur(bg_mask, (11,11), 0)
        bg_mask_3 = np.repeat(bg_mask[:, :, None], 3, axis=2)

        if current_filter == "blur":
            output = (1 - bg_mask_3) * output + bg_mask_3 * blurred
        elif current_filter == "gray":
            output = (1 - bg_mask_3) * output + bg_mask_3 * gray
        elif current_filter == "edge":
            output = (1 - bg_mask_3) * output + bg_mask_3 * edges

    output = output.astype(np.uint8)

    # -------- FPS --------
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time

    cv2.putText(output, f"FPS:{int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Segmentation", output)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('1'):
        current_filter = "blur"
    if key == ord('2'):
        current_filter = "gray"
    if key == ord('3'):
        current_filter = "edge"
    if key == ord('c'):
        selected_objects.clear()
        background_selected = False

cap.release()
cv2.destroyAllWindows()