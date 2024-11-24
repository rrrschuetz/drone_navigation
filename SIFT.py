import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the large (satellite) and small (drone) images
large_image_path = "karlsdorf_highres2.jpg"  # Path to the large satellite image
#large_image_path = "48MP-200.JPG"  # Path to the large satellite image
small_image_path = "normal-120.JPG"         # Path to the smaller drone image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Preprocessing function
def preprocess(image, is_large=False):
    """
    Preprocessing pipeline tailored for drone and satellite images.
    """
    # Step 1: Adjust CLAHE for satellite image scale
    tile_size = (16, 16) if is_large else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_size)
    enhanced = clahe.apply(image)

    # Step 2: Edge Detection and Filtering
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(enhanced, -1, kernel)

    # Blend edges less aggressively
    edge_boosted = cv2.addWeighted(enhanced, 1.0, filtered, 0.1, 0)

    # Step 3: Normalize intensities
    normalized = cv2.normalize(edge_boosted, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

# Preprocess both images
large_image_enhanced = preprocess(large_image, is_large=True)
small_image_enhanced = preprocess(small_image, is_large=False)

# Resize images to manage scale differences
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image_enhanced, 4096)
small_image_resized = resize_with_aspect_ratio(small_image_enhanced, 1024)

# Detect and compute keypoints and descriptors using SIFT
sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=20)  # Tuned for larger-scale features
keypoints_large, descriptors_large = sift.detectAndCompute(large_image_resized, None)
keypoints_small, descriptors_small = sift.detectAndCompute(small_image_resized, None)

# Visualize keypoints
def visualize_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_keypoints, cmap="gray")
    plt.title(f"{title} - Keypoints: {len(keypoints)}")
    plt.axis("off")
    plt.show()

visualize_keypoints(large_image_resized, keypoints_large, "Large Image Keypoints")
visualize_keypoints(small_image_resized, keypoints_small, "Small Image Keypoints")

# Match descriptors using BFMatcher with Lowe's Ratio Test
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Use L2 norm for SIFT
matches = bf.knnMatch(descriptors_small, descriptors_large, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:  # Stricter ratio for satellite images
        good_matches.append(m)

# Visualize matches
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_matches(small_image_resized, keypoints_small, large_image_resized, keypoints_large, good_matches, "Good Matches")

# Check for enough matches and estimate homography
MIN_MATCH_COUNT = 8
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography using RANSAC
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = small_image_resized.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw detected region on the large image
        large_image_color = cv2.cvtColor(large_image_resized, cv2.COLOR_GRAY2BGR)
        large_image_with_box = cv2.polylines(
            large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=5
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(large_image_with_box)
        plt.title("Detected Region")
        plt.axis("off")
        plt.show()
    except cv2.error as e:
        print("Homography estimation failed:", e)
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
