# Road Surface Obstacle and Traffic Warning Object Segmentation System

# Import the necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO
import config

# Model and inference configuration (loaded from config.py)
MODEL_PATH           = config.MODEL_PATH
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
IOU_THRESHOLD        = config.IOU_THRESHOLD
MASK_ALPHA           = config.MASK_ALPHA

# Class metadata (loaded from config.py; must match the dataset's training-time class indices)
CLASS_NAMES = config.CLASS_NAMES

# Colors (BGR) — matched to the dataset's platform visualization (loaded from config.py)
CLASS_COLORS = config.CLASS_COLORS

# Visualization parameters (loaded from config.py; font kept here as a cv2 constant)
CONTOUR_THICKNESS = config.CONTOUR_THICKNESS
LABEL_FONT        = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE       = config.LABEL_SCALE
LABEL_THICKNESS   = config.LABEL_THICKNESS
LABEL_PADDING     = config.LABEL_PADDING


# Function to pick black or white text color for best legibility against a BGR background
def get_text_color_for_background(bg_color):
    b, g, r = bg_color
    luma = 0.114 * b + 0.587 * g + 0.299 * r  # perceived brightness
    return (0, 0, 0) if luma > 160 else (255, 255, 255)


# Function to resize the model's binary masks to match the original frame dimensions
def resize_masks_to_frame(masks, frame_height, frame_width):
    if masks.shape[1] == frame_height and masks.shape[2] == frame_width:
        return masks

    resized = np.zeros((masks.shape[0], frame_height, frame_width), dtype=masks.dtype)
    for i in range(masks.shape[0]):
        resized[i] = cv2.resize(masks[i], (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    return resized


# Function to blend per-class colored masks onto the frame, painting larger masks first
def render_masks(frame, masks, classes):
    if len(masks) == 0:
        return frame

    overlay = frame.copy()

    # Paint largest masks first so smaller (closer) ones stay visible on top
    order = np.argsort([m.sum() for m in masks])[::-1]

    for i in order:
        mask   = masks[i].astype(bool)
        cls_id = int(classes[i])
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        overlay[mask] = color

    return cv2.addWeighted(overlay, 1.0 - MASK_ALPHA, frame, MASK_ALPHA, 0)


# Function to outline each instance and place a class + confidence label at its anchor
def render_contours_and_labels(frame, masks, classes, confidences):
    if len(masks) == 0:
        return frame

    # Largest masks first so labels for smaller masks remain on top
    order = np.argsort([m.sum() for m in masks])[::-1]

    for i in order:
        mask_u8 = (masks[i] > 0).astype(np.uint8) * 255
        cls_id  = int(classes[i])
        conf    = float(confidences[i])
        color   = CLASS_COLORS.get(cls_id, (200, 200, 200))
        name    = CLASS_NAMES.get(cls_id, "unknown")

        # Smooth outline of the segmentation mask
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cv2.drawContours(frame, contours, -1, color, CONTOUR_THICKNESS, lineType=cv2.LINE_AA)

        # Anchor the label at the top-left of the largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, _, _ = cv2.boundingRect(largest)

        label = f"{name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)

        label_x = max(x, 0)
        label_y = max(y - 6, text_h + LABEL_PADDING)

        # Filled background for legibility
        cv2.rectangle(frame,
                      (label_x - 2, label_y - text_h - LABEL_PADDING),
                      (label_x + text_w + LABEL_PADDING, label_y + 2),
                      color, -1)

        cv2.putText(frame, label,
                    (label_x + 1, label_y - 2),
                    LABEL_FONT, LABEL_SCALE,
                    get_text_color_for_background(color),
                    LABEL_THICKNESS, cv2.LINE_AA)

    return frame


# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video (paths loaded from config.py)
video_capture = cv2.VideoCapture(config.INPUT_VIDEO)  # write the name of the video file here
output_file = config.OUTPUT_VIDEO  # write the name and extension of the video file to be recorded

# Video features
frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object to save the video file
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()  # read video frame
    if not ret:
        break

    # Run YOLO segmentation inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    if results.masks is not None and len(results.masks) > 0:
        masks       = results.masks.data.cpu().numpy()
        classes     = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # Ensure masks are at frame resolution
        masks = resize_masks_to_frame(masks, frame_height, frame_width)

        # Visualization stack: filled masks -> contours + labels
        frame = render_masks(frame, masks, classes)
        frame = render_contours_and_labels(frame, masks, classes, confidences)

    # Write the drawn frame to the video file to be saved
    output_video.write(frame)

    # Show video with road surface obstacle and traffic warning object segmentation
    cv2.imshow("Road Surface Obstacle and Traffic Warning Object Segmentation", frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()