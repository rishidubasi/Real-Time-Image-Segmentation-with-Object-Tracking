# Real-Time Image Segmentation with Object Tracking

## Overview
This project performs real-time image segmentation using YOLOv8 and applies filters independently to different objects and the background in a live webcam feed.

It includes object tracking, smoothing, and persistence logic to improve stability and reduce flickering.

## Features
- Real-time instance segmentation using YOLOv8
- IoU-based object tracking across frames
- Temporal smoothing to reduce flickering
- Object persistence (handles temporary disappearance)
- Apply filters (blur, grayscale, edge) to selected objects
- Background filtering separately from objects
- Interactive mouse-based object selection

## Tech Stack
- Python
- OpenCV
- NumPy
- YOLOv8 (Ultralytics)

## How It Works
1. Captures live video from webcam
2. Performs segmentation using YOLOv8
3. Tracks objects using IoU matching
4. Applies smoothing and persistence across frames
5. Allows user to click and apply filters on objects/background

## Controls
- Mouse Click → Select/Deselect objects
- Click on background → Toggle background filter
- Press `1` → Blur filter
- Press `2` → Grayscale filter
- Press `3` → Edge detection
- Press `c` → Clear all selections
- Press `ESC` → Exit

## How to Run

1. Install dependencies:
2. Run the script:
