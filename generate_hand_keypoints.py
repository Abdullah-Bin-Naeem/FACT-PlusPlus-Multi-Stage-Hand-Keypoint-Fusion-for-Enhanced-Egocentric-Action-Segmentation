#!/usr/bin/env python3
"""
Hand Keypoint Generation Script using MediaPipe

This script processes video frames and extracts hand keypoints using MediaPipe Hands.
The keypoints are saved as flattened NumPy arrays (.npy files) for each frame.

Output format:
- Left hand: 21 keypoints × 3 coordinates (x, y, z) = 63 values
- Right hand: 21 keypoints × 3 coordinates (x, y, z) = 63 values
- Total: 126 values per frame (flattened array)

If a hand is not detected, zeros are used for that hand's keypoints.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def generate_hand_keypoints(frames_root, keypoints_root, use_cuda=False):
    """
    Generate hand keypoints for all frames in the dataset.
    
    Args:
        frames_root (str): Path to the root directory containing video frame folders
        keypoints_root (str): Path to save the generated keypoint .npy files
        use_cuda (bool): Whether to attempt using CUDA acceleration (if available)
    """
    # Check OpenCV GPU support
    if use_cuda:
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"OpenCV CUDA Support: {cuda_available}")
        if not cuda_available:
            print("Warning: CUDA requested but not available. Using CPU.")
    
    # Create output directory
    os.makedirs(keypoints_root, exist_ok=True)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.1
    )
    
    # Get list of video folders
    video_folders = sorted([
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ])
    
    print(f"Processing {len(video_folders)} video folders...")
    
    # Process each video folder
    for video_folder in tqdm(video_folders, desc="Videos"):
        video_frames_path = os.path.join(frames_root, video_folder)
        video_keypoints_path = os.path.join(keypoints_root, video_folder)
        os.makedirs(video_keypoints_path, exist_ok=True)
        
        # Get all frame files
        frame_files = sorted([
            f for f in os.listdir(video_frames_path)
            if f.endswith(".png")
        ], key=lambda x: int(os.path.splitext(x)[0]))
        
        # Process each frame
        for frame_file in frame_files:
            frame_path = os.path.join(video_frames_path, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue
            
            # Process with MediaPipe
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Initialize keypoint arrays (zeros if hand not detected)
            left_hand_keypoints = np.zeros((21, 3))
            right_hand_keypoints = np.zeros((21, 3))
            
            # Extract keypoints if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_classification in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    hand_type = hand_classification.classification[0].label
                    keypoints = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ])
                    
                    if hand_type == "Left":
                        left_hand_keypoints = keypoints
                    elif hand_type == "Right":
                        right_hand_keypoints = keypoints
            
            # Flatten and concatenate: [left_hand (63), right_hand (63)] = 126 values
            flattened_keypoints = np.concatenate([
                left_hand_keypoints.flatten(),
                right_hand_keypoints.flatten()
            ])
            
            # Save as .npy file
            npy_filename = os.path.join(
                video_keypoints_path,
                f"{os.path.splitext(frame_file)[0]}.npy"
            )
            np.save(npy_filename, flattened_keypoints)
    
    hands.close()
    print("Processing complete. Flattened keypoints saved as NumPy arrays.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate hand keypoints from video frames using MediaPipe"
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="Path to the root directory containing video frame folders"
    )
    parser.add_argument(
        "--keypoints_root",
        type=str,
        default="keypoints_output_arr",
        help="Path to save the generated keypoint .npy files (default: keypoints_output_arr)"
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Attempt to use CUDA acceleration if available"
    )
    
    args = parser.parse_args()
    
    generate_hand_keypoints(
        frames_root=args.frames_root,
        keypoints_root=args.keypoints_root,
        use_cuda=args.use_cuda
    )
