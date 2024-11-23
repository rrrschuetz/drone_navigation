import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the large (satellite) and small images
large_image_path = "bruchsal_highres.jpg"  # Replace with path to the large satellite image
small_image_path = "luftbild6.jpg"         # Replace with path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

def preprocess(image):
    """
    Enhanced preprocessing pipeline with improved edge sharpening and watermark removal.
    """
    # Step 1: CLAHE (Local Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Step 2: Noise Reduction
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Step 3: Watermark Removal
    # Threshold to create a mask for the watermark
    _, watermark_mask = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(denoised, watermark_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Step 4: Edge Sharpening (Unsharp Masking)
    gaussian_blurred = cv2.GaussianBlur(inpainted, (9, 9), 2)
    sharpened = cv2.addWeighted(inpainted, 1.5, gaussian_blurred, -0.5, 0)

    # Step 5: Normalize Pixel Intensities
    normalized = cv2.normalize(sharpened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return normalized


large_image_enhanced = preprocess(large_image)
small_image_enhanced = preprocess(small_image)

# Resize images while preserving aspect ratio
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image_enhanced, 4096)
small_image_resized = resize_with_aspect_ratio(small_image_enhanced,  2048)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
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
    if m.distance < 0.7 * n.distance:  # Lowe's ratio
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
MIN_MATCH_COUNT = 10
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
