import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the large (satellite) and small images
large_image_path = 'satellite_image.jpg'  # Replace with path to the large satellite image
small_image_path = 'small_image.jpg'      # Replace with path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Enhance the large image to improve feature detection
#large_image_enhanced = cv2.equalizeHist(large_image)

# Downscale large image for efficient processing
scale_factor = 0.5  # Adjust as needed
large_image_resized = cv2.resize(large_image, None, fx=scale_factor, fy=scale_factor)


# Initialize the SIFT detector with adjusted thresholds
detector = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)

# Detect and compute descriptors for both images
keypoints_large, descriptors_large = detector.detectAndCompute(large_image_resized, None)
keypoints_small, descriptors_small = detector.detectAndCompute(small_image, None)

# Function to visualize keypoints
def visualize_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

# Visualize keypoints in both images for debugging
visualize_keypoints(large_image_resized, keypoints_large, "Keypoints in Enhanced Large Image")
visualize_keypoints(small_image, keypoints_small, "Keypoints in Small Image")

# Check if keypoints are detected
if len(keypoints_large) == 0:
    raise ValueError("No keypoints detected in the large image.")
if len(keypoints_small) == 0:
    raise ValueError("No keypoints detected in the small image.")

# Use the BFMatcher for SIFT (L2 norm for floating-point descriptors)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Match descriptors between small and large images
matches = bf.knnMatch(descriptors_small, descriptors_large, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Lowe's ratio test
        good_matches.append(m)

# Visualize matches for debugging
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_matches(small_image, keypoints_small, large_image_resized, keypoints_large, good_matches, "Good Matches")

# Check if there are enough matches to proceed
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    # Extract location of good matches
    src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix with RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use the homography matrix to find the position in the large image
    h, w = small_image.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw detected region
    large_image_color = cv2.cvtColor(large_image_enhanced, cv2.COLOR_GRAY2BGR)
    large_image_with_box = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=10, lineType=cv2.LINE_AA
    )

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(large_image_with_box, cv2.COLOR_BGR2RGB))
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()

    # Calculate the approximate center of the detected region
    location_center = np.mean(dst, axis=0)
    print("Estimated location of small image center:", location_center[0])
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
