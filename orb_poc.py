import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the large (satellite) and small images
large_image_path = 'bruchsal_highres.jpg'  # Replace with path to the large satellite image
small_image_path = 'luftbild2.jpg'      # Replace with path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Enhance and preprocess the large image
large_image_enhanced = cv2.equalizeHist(large_image)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
large_image_enhanced = cv2.filter2D(large_image_enhanced, -1, kernel)

# Resize large image for faster processing
scale_factor = 0.5
large_image_resized = cv2.resize(large_image_enhanced, None, fx=scale_factor, fy=scale_factor)

# Initialize the AKAZE detector
detector = cv2.AKAZE_create()
detector.setThreshold(0.001)  # Lower threshold for more keypoints
keypoints_large, descriptors_large = detector.detectAndCompute(large_image_resized, None)
keypoints_small, descriptors_small = detector.detectAndCompute(small_image, None)

# Fallback to ORB if AKAZE fails
if len(keypoints_large) == 0:
    detector = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
    keypoints_large, descriptors_large = detector.detectAndCompute(large_image_resized, None)

if len(keypoints_large) == 0:
    raise ValueError("No keypoints detected in the large image, even after fallback.")

# Visualize keypoints for debugging
def visualize_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

#visualize_keypoints(large_image_resized, keypoints_large, "Keypoints in Large Image")
#visualize_keypoints(small_image, keypoints_small, "Keypoints in Small Image")

# Use appropriate norm based on detector type
if "ORB" in type(detector).__name__:
    norm_type = cv2.NORM_HAMMING
else:  # AKAZE or SIFT
    norm_type = cv2.NORM_L2

# Use BFMatcher for matching
bf = cv2.BFMatcher(norm_type, crossCheck=False)

# Match descriptors between small and large images
matches = bf.knnMatch(descriptors_small, descriptors_large, k=2)

# Apply a looser Lowe's ratio test
good_matches = []
for m, n in matches:
    if (m.distance < 0.85 * n.distance):
        good_matches.append(m)

# Visualize matches
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_matches(small_image, keypoints_small, large_image_resized, keypoints_large, good_matches, "Good Matches")

# Check for enough matches
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = small_image.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    large_image_with_box = cv2.polylines(
        large_image_resized, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=10
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(large_image_with_box, cmap='gray')
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
