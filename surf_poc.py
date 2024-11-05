import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure you have opencv-contrib-python installed:
# pip install opencv-contrib-python

# Load the large (satellite) and small images
large_image_path = 'satellite_image.jpg'  # Replace with path to the large satellite image
small_image_path = 'small_image.jpg'      # Replace with path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Initialize the SURF detector with a Hessian threshold (e.g., 400)
surf = cv2.xfeatures2d.SURF_create(400)

# Detect keypoints and descriptors in both images
keypoints_large, descriptors_large = surf.detectAndCompute(large_image, None)
keypoints_small, descriptors_small = surf.detectAndCompute(small_image, None)

# Use the FLANN-based matcher to match descriptors between the two images
# FLANN parameters
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors between small and large images
matches = flann.knnMatch(descriptors_small, descriptors_large, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Check if there are enough matches to proceed
MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    # Extract location of good matches
    src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use the homography matrix to find the position in the large image
    h, w = small_image.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw the detected region in the large image
    large_image_with_box = cv2.polylines(large_image.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(large_image_with_box, cmap='gray')
    plt.title("Detected Location of Small Image in Large Satellite Image")
    plt.axis('off')
    plt.show()

    # Optionally, calculate the approximate location coordinates within the large image
    location_center = np.mean(dst, axis=0)
    print("Estimated location of small image center:", location_center[0])
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
