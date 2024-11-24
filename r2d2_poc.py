import sys
sys.path.append('/home/rrrschuetz/r2d2')  # Update to the path of your R2D2 repository

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from r2d2.extract import load_model, extract_keypoints

# Paths to images
large_image_path = "karlsdorf_highres2.jpg"  # Path to large satellite image
small_image_path = "normal-120.JPG"         # Path to smaller drone image

# Load images
large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Preprocessing function
def preprocess(image):
    """
    Normalize the image to improve feature detection.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

large_image_preprocessed = preprocess(large_image)
small_image_preprocessed = preprocess(small_image)

# Load the R2D2 model
model_path = "/path/to/r2d2/models/r2d2_WASF_N16.pt"  # Update this path
model = load_model(model_path)
model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

# Extract keypoints and descriptors using R2D2
def extract_features(image, model):
    """
    Extract keypoints and descriptors using R2D2.
    """
    with torch.no_grad():
        # Convert image to PyTorch tensor
        tensor_image = torch.tensor(image[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0)
        tensor_image = tensor_image.cuda() if torch.cuda.is_available() else tensor_image

        # Extract keypoints and descriptors
        keypoints, descriptors = extract_keypoints(model, tensor_image)
    return keypoints, descriptors

keypoints_large, descriptors_large = extract_features(large_image_preprocessed, model)
keypoints_small, descriptors_small = extract_features(small_image_preprocessed, model)

# Match descriptors using BFMatcher
def match_features(desc1, desc2):
    """
    Match descriptors using BFMatcher with Lowe's ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

good_matches = match_features(descriptors_small, descriptors_large)

# Visualize matches
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(
        img1, [cv2.KeyPoint(*kp[:2], 1) for kp in kp1],
        img2, [cv2.KeyPoint(*kp[:2], 1) for kp in kp2],
        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_matches(small_image_preprocessed, keypoints_small, large_image_preprocessed, keypoints_large, good_matches, "R2D2 Matches")

# Check for enough matches and compute homography
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_small[m.queryIdx][:2] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx][:2] for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Draw detected region
    h, w = small_image_preprocessed.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    large_image_color = cv2.cvtColor(large_image_preprocessed, cv2.COLOR_GRAY2BGR)
    detected_image = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=5
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(detected_image)
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()
else:
    print(f"Not enough matches found: {len(good_matches)}/{MIN_MATCH_COUNT}")
