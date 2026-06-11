#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"  # download from MediaPipe (Pose Landmarker model)
INPUT_VIDEO = "video/GX010235.MP4"
OUTPUT_CSV = "video/GX010235_pose.csv"


def process_video_to_csv(video_path, output_csv, model_path=MODEL_PATH):
    """Process video with MediaPipe and save pose landmarks to CSV."""
    # Build landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize CSV
    csv_file = open(output_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    # Create Header: 33 landmarks * (World X,Y,Z + Norm X,Y,Z + Visibility)
    header = ["frame_idx", "timestamp_ms"]
    for i in range(33):
        header.extend([
            f"lm{i}_world_x", f"lm{i}_world_y", f"lm{i}_world_z",
            f"lm{i}_norm_x", f"lm{i}_norm_y", f"lm{i}_norm_z",
            f"lm{i}_visibility"
        ])
    csv_writer.writerow(header)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # timestamp must increase (ms)
        timestamp_ms = int((frame_idx / fps) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_world_landmarks and result.pose_landmarks:
            # We assume the first detected person [0]
            world_landmarks = result.pose_world_landmarks[0]
            norm_landmarks = result.pose_landmarks[0]

            row = [frame_idx, timestamp_ms]

            # Loop through all 33 landmarks
            for i in range(33):
                # World coordinates (meters)
                row.append(world_landmarks[i].x)
                row.append(world_landmarks[i].y)
                row.append(world_landmarks[i].z)

                # Normalized coordinates (0.0 - 1.0)
                row.append(norm_landmarks[i].x)
                row.append(norm_landmarks[i].y)
                row.append(norm_landmarks[i].z)

                # Visibility score
                row.append(norm_landmarks[i].visibility)

            csv_writer.writerow(row)

        frame_idx += 1

    cap.release()
    landmarker.close()
    csv_file.close()


def main():
    """Main function for command-line usage."""
    process_video_to_csv(INPUT_VIDEO, OUTPUT_CSV)
    print("Wrote CSV:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
