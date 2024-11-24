import sys
sys.path.append('/home/rrrschuetz/superpoint')  # Update to the path of your R2D2 repository

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from superpoint import SuperPointNet  # Ensure this path matches your repository
from scipy.ndimage import maximum_filter
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter, sobel
import scipy.ndimage

# Paths to images
#large_image_path = "karlsdorf_highres2.jpg"  # Path to large satellite image
large_image_path = "48MP-200.JPG"  # Path to large satellite image
small_image_path = "normal-120.JPG"         # Path to smaller drone image

# Load images
large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Preprocessing function
def preprocess(image):
    """
    Normalize and enhance the image for better feature detection.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

large_image_preprocessed = preprocess(large_image)
small_image_preprocessed = preprocess(small_image)

# Resize images for consistent processing
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image_preprocessed, 4096)
small_image_resized = resize_with_aspect_ratio(small_image_preprocessed, 4096)

# Initialize SuperPoint Model
model_path = "/home/rrrschuetz/SuperPointPretrainedNetwork/models/superpoint_v1.pth"  # Update this path
superpoint = SuperPointNet()
superpoint.load_state_dict(torch.load(model_path,weights_only=True))
superpoint.eval()
if torch.cuda.is_available():
    superpoint = superpoint.cuda()

def preprocess(image):
    """
    Normalize and enhance the image for better feature detection.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

def calculate_orientations(image, keypoints):
    """
    Calculate the dominant orientation for each keypoint using image gradients.
    """
    # Compute image gradients
    grad_x = sobel(image, axis=1)  # Horizontal gradients
    grad_y = sobel(image, axis=0)  # Vertical gradients

    orientations = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < grad_x.shape[1] and 0 <= y < grad_x.shape[0]:
            # Compute the gradient direction at the keypoint
            angle = np.arctan2(grad_y[y, x], grad_x[y, x])
            orientations.append(angle)
        else:
            orientations.append(0.0)  # Default orientation for out-of-bounds keypoints
    return np.array(orientations)

def rotate_descriptors(descriptors, orientations):
    """
    Rotate descriptors to align with the canonical orientation.
    This modifies descriptors to be rotation invariant.
    """
    rotated_descriptors = []
    for desc, angle in zip(descriptors, orientations):
        # Compute the rotation matrix
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])

        # Assuming descriptors are high-dimensional (e.g., 256),
        # split into 2D components for rotation.
        reshaped_desc = desc.reshape(-1, 2)  # Reshape into pairs
        rotated = np.dot(reshaped_desc, rotation_matrix.T)
        rotated_descriptors.append(rotated.flatten())  # Flatten back to original shape

    return np.array(rotated_descriptors)


def extract_features(image, model, confidence_threshold=0.01):
    """
    Extract keypoints and descriptors using SuperPoint with rotation invariance.
    """
    with torch.no_grad():
        # Prepare the image for SuperPoint
        tensor_image = torch.tensor(image[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0)
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()

        # Forward pass through the SuperPoint model
        outputs = model(tensor_image)
        heatmap = outputs[0].squeeze(0).cpu().numpy()[0]  # First channel, ensure 2D
        dense_descriptors = outputs[1].squeeze().cpu().numpy()

        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Dense descriptors shape: {dense_descriptors.shape}")

        # Normalize the heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Apply non-maximum suppression
        footprint = np.ones((3, 3), dtype=bool)  # Kernel size for suppression
        local_maxima = (heatmap == maximum_filter(heatmap, footprint=footprint))
        keypoints = np.argwhere(local_maxima & (heatmap > confidence_threshold))

        print(f"Detected {len(keypoints)} keypoints with confidence threshold {confidence_threshold}")

        if len(keypoints) == 0:
            return np.array([]), np.array([])

        # Compute scaling factors
        scale_x = image.shape[1] / heatmap.shape[1]
        scale_y = image.shape[0] / heatmap.shape[0]

        # Scale and convert (row, col) to (x, y)
        keypoints = [(kp[1] * scale_x, kp[0] * scale_y) for kp in keypoints]

        # Extract descriptors at keypoint locations
        descriptors = []
        for kp in keypoints:
            x, y = int(kp[0] / scale_x), int(kp[1] / scale_y)  # Scale back to heatmap coordinates
            if 0 <= x < dense_descriptors.shape[2] and 0 <= y < dense_descriptors.shape[1]:
                descriptors.append(dense_descriptors[:, y, x])
            else:
                descriptors.append(np.zeros(dense_descriptors.shape[0]))  # Default descriptor for out-of-bounds

        descriptors = np.array(descriptors)

        # Compute keypoint orientations
        orientations = calculate_orientations(image, keypoints)

        # Rotate descriptors for rotation invariance
        aligned_descriptors = rotate_descriptors(descriptors, orientations)

    return keypoints, aligned_descriptors, orientations


# Replace calls to `extract_features` with `extract_features_rotation_invariant`
keypoints_large, descriptors_large, orientations_large = extract_features(large_image_resized, superpoint)
keypoints_small, descriptors_small, orientations_small  = extract_features(small_image_resized, superpoint)

print(f"Keypoints (Sample): {keypoints_large[:5]}")  # Print the first few keypoints
print(f"Image Dimensions: {large_image_resized.shape}")     # Print the current image dimensions


def visualize_keypoints_with_orientations(image, keypoints, orientations, title="Keypoints with Orientations"):
    """
    Visualize keypoints and their orientations.
    :param image: Grayscale image.
    :param keypoints: List of keypoints [(x, y), ...].
    :param orientations: List of orientations (in radians, as floats).
    :param title: Title for the plot.
    """
    if not all(isinstance(angle, (float, np.float32, np.float64)) for angle in orientations):
        orientations = [float(angle) for angle in orientations]  # Ensure numeric type

    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    for (x, y), angle in zip(keypoints, orientations):
        dx = 10 * np.cos(angle)  # Horizontal component of the arrow
        dy = 10 * np.sin(angle)  # Vertical component of the arrow
        plt.arrow(x, y, dx, dy, color='red', head_width=3, head_length=5, linewidth=1.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Visualize keypoints
visualize_keypoints_with_orientations(large_image_resized, keypoints_large, orientations_large,"Large Image Keypoints")
visualize_keypoints_with_orientations(small_image_resized, keypoints_small, orientations_small,"Small Image Keypoints")

# Match descriptors using BFMatcher
def match_features(desc1, desc2, ratio_threshold=0.75):
    """
    Match descriptors using BFMatcher with Lowe's ratio test.
    """
    if desc1.size == 0 or desc2.size == 0:
        raise ValueError("One or both descriptor arrays are empty. Cannot perform matching.")

    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    #matches = bf.knnMatch(desc1, desc2, k=2)

    # FLANN Parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_small, descriptors_large, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:  # Relax the ratio threshold
            good_matches.append(m)

    return good_matches

good_matches = match_features(descriptors_small, descriptors_large)

# Visualize matches
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    """
    Visualize matches between two sets of keypoints with thicker lines and custom styling.
    """
    # Convert keypoints to OpenCV KeyPoint objects
    kp1_converted = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in kp1]
    kp2_converted = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in kp2]

    # Draw matches with customized line thickness
    img_matches = cv2.drawMatches(
        img1, kp1_converted, img2, kp2_converted, matches, None,
        matchColor=(0, 255, 0),  # Green lines
        singlePointColor=(255, 0, 0),  # Blue keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add thicker lines to highlight matches
    for match in matches:
        pt1 = tuple(map(int, kp1_converted[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2_converted[match.trainIdx].pt))
        pt2 = (pt2[0] + img1.shape[1], pt2[1])  # Adjust for second image position
        cv2.line(img_matches, pt1, pt2, (0, 255, 255), 5)  # Yellow thick lines

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_matches(small_image_resized, keypoints_small, large_image_resized, keypoints_large, good_matches, "SuperPoint Matches")

print (f"Number of matches: {len(good_matches)}")

# Calculate distances for matched pairs
matched_distances = [np.linalg.norm(descriptors_small[m.queryIdx] - descriptors_large[m.trainIdx]) for m in good_matches]

# Plot histogram
plt.hist(matched_distances, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Descriptor Distances for Matches")
plt.xlabel("Descriptor Distance")
plt.ylabel("Frequency")
plt.show()

# Check for enough matches and compute homography
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_small[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC
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
    print(f"Not enough matches found: {len(good_matches)}/{MIN_MATCH_COUNT}")
