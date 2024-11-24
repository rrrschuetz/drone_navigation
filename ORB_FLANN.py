import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
large_image_path = "karlsdorf_highres4.jpg"
small_image_path = "normal-120.JPG"

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Preprocessing function
def preprocess(image, is_large=False):
    """
    Preprocessing pipeline for better feature matching.
    """
    tile_size = (16, 16) if is_large else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
    enhanced = clahe.apply(image)

    # Edge Detection and Filtering
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(enhanced, -1, kernel)
    edge_boosted = cv2.addWeighted(enhanced, 0.9, filtered, 0.1, 0)

    # Normalize intensities
    normalized = cv2.normalize(edge_boosted, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

# Preprocess both images
large_image_preprocessed = preprocess(large_image, is_large=True)
small_image_preprocessed = preprocess(small_image, is_large=False)

# Resize images to manage scale differences
def resize_with_aspect_ratio(image, scale):
    h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Multiscale matching
scales = [1.0, 0.75, 0.5]  # Different scales for satellite image
best_matches = []
best_homography = None

for scale in scales:
    # Resize large image
    large_image_scaled = resize_with_aspect_ratio(large_image_preprocessed, scale)

    # ORB Feature Detector
    orb = cv2.ORB_create(nfeatures=10000)
    keypoints_large, descriptors_large = orb.detectAndCompute(large_image_scaled, None)
    keypoints_small, descriptors_small = orb.detectAndCompute(small_image_preprocessed, None)

    # FLANN Matcher
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # LSH
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_small, descriptors_large, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # If enough matches are found, estimate homography
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        if M is not None:
            best_matches = good_matches
            best_homography = M
            break

# Visualize Results
if best_homography is not None:
    h, w = small_image_preprocessed.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, best_homography)

    large_image_color = cv2.cvtColor(large_image_scaled, cv2.COLOR_GRAY2BGR)
    large_image_with_box = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(large_image_with_box)
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()
else:
    print("No sufficient matches found.")
