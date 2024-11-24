import sys
sys.path.append('/home/rrrschuetz/d2-net')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from d2_net.models.d2_net import D2Net
from d2_net.utils.pyramid import process_multiscale
import torch

# Load images
large_image_path = "karlsdorf_highres2.jpg"
small_image_path = "normal-120.JPG"

large_image = cv2.imread(large_image_path, cv2.COLOR_BGR2GRAY)
small_image = cv2.imread(small_image_path, cv2.COLOR_BGR2GRAY)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Preprocessing function
def preprocess(image, is_large=False):
    """
    Preprocessing tailored for D2-Net inputs.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16) if is_large else (8, 8))
    enhanced = clahe.apply(image)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

# Preprocess both images
large_image_preprocessed = preprocess(large_image, is_large=True)
small_image_preprocessed = preprocess(small_image, is_large=False)

# Resize images
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image_preprocessed, 1024)
small_image_resized = resize_with_aspect_ratio(small_image_preprocessed, 1024)

# D2-Net Model Initialization
model = D2Net(model_file='models/d2_tf.pth', use_relu=True)  # Use the path to your pretrained model
model = model.cuda() if torch.cuda.is_available() else model

# Extract features with D2-Net
def extract_features(image, model):
    with torch.no_grad():
        input_image = torch.tensor(image[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0)
        input_image = input_image.cuda() if torch.cuda.is_available() else input_image
        features = process_multiscale(input_image, model)  # Extract multiscale features
    return features

large_features = extract_features(large_image_resized, model)
small_features = extract_features(small_image_resized, model)

# Match descriptors using BFMatcher
def match_features(features1, features2):
    kp1, desc1 = features1['keypoints'], features1['descriptors']
    kp2, desc2 = features2['keypoints'], features2['descriptors']
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches

keypoints1, keypoints2, good_matches = match_features(small_features, large_features)

# Visualize Matches
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

visualize_matches(small_image_resized, keypoints1, large_image_resized, keypoints2, good_matches, "D2-Net Matches")

# Check for homography if enough matches are found
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Draw detected region
    h, w = small_image_resized.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    large_image_color = cv2.cvtColor(large_image_resized, cv2.COLOR_GRAY2BGR)
    detected_image = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=5
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(detected_image)
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()
else:
    print(f"Not enough matches: {len(good_matches)}")
