# Road Surface Obstacle and Traffic Warning Object Segmentation System — Configuration

# Model and inference constraints
MODEL_PATH           = "road-surface-obstacle-and-traffic-warning-object-segmentation.pt"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD        = 0.9
MASK_ALPHA           = 0.4

# Class metadata (must match the dataset's training-time class indices)
CLASS_NAMES = {
    0: "pothole",
    1: "traffic-cone",
    2: "road-barrier",
    3: "patch",
    4: "crack",
}

# Colors (BGR) — matched to the dataset's platform visualization
CLASS_COLORS = {
    0: (0, 60, 255),     # pothole
    1: (220, 230, 20),   # traffic-cone
    2: (235, 235, 235),  # road-barrier
    3: (170, 235, 35),   # patch
    4: (10, 35, 10),     # crack
}

# Visualization parameters
CONTOUR_THICKNESS = 2
LABEL_SCALE       = 0.6
LABEL_THICKNESS   = 1
LABEL_PADDING     = 4

# Video files
INPUT_VIDEO  = "Road-Surface-Obstacle-and-Traffic-Warning-Object.mp4"                # input video file name
OUTPUT_VIDEO = "Road-Surface-Obstacle-and-Traffic-Warning-Object-Segmentation.mp4"   # output video file name
