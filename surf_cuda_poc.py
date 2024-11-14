import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the large (satellite) and small images
large_image_path = 'satellite_image.jpg'  # Replace with path to the large satellite image
small_image_path = 'small_imag3.jpg'      # Replace with path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Initialize the CUDA-enabled SURF detector with a Hessian threshold (e.g., 400)
try:
    surf = cv2.cuda.SURF_CUDA_create(400)
except AttributeError:
    print("CUDA SURF not available. Switching to ORB.")
    surf = cv2.cuda.ORB_create(400)

# Upload images to GPU memory
gpu_large_image = cv2.cuda_GpuMat()
gpu_small_image = cv2.cuda_GpuMat()
gpu_large_image.upload(large_image)
gpu_small_image.upload(small_image)

# Detect keypoints and compute descriptors in both images
keypoints_large, descriptors_large = surf.detectWithDescriptors(gpu_large_image, None)
keypoints_small, descriptors_small = surf.detectWithDescriptors(gpu_small_image, None)

# Convert GPU keypoints and descriptors to CPU for matching
keypoints_large = surf.downloadKeypoints(keypoints_large)
keypoints_small = surf.downloadKeypoints(keypoints_small)
descriptors_large = descriptors_large.download()
descriptors_small = descriptors_small.download()

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

    # Find the homography matrix (no CUDA support, so this remains on CPU)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use the homography matrix to find the position in the large image
    h, w = small_image.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Convert the large image to color to enable color drawing
    large_image_color = cv2.cvtColor(large_image, cv2.COLOR_GRAY2BGR)
    # Draw a thick yellow line around the detected region in the large image
    large_image_with_box = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=10, lineType=cv2.LINE_AA
    )

    # Mouse callback to create a magnifier
    def magnifier(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Define the size of the magnifier window
            mag_size = 200  # Size of the region to magnify
            zoom_factor = 3  # Magnification factor

            # Calculate coordinates of the zoomed-in area
            x1 = max(0, x - mag_size // 2)
            y1 = max(0, y - mag_size // 2)
            x2 = min(large_image_with_box.shape[1], x + mag_size // 2)
            y2 = min(large_image_with_box.shape[0], y + mag_size // 2)

            # Crop and zoom
            magnifier_region = large_image_with_box[y1:y2, x1:x2]
            magnified_view = cv2.resize(magnifier_region, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

            # Display the magnified region in a separate window
            cv2.imshow("Magnifier", magnified_view)

    # Display the main image with the magnifier
    cv2.namedWindow("Detected Location", cv2.WINDOW_NORMAL)  # Allows resizing
    cv2.setMouseCallback("Detected Location", magnifier)  # Set the mouse callback for the magnifier
    cv2.imshow("Detected Location", large_image_with_box)
    cv2.resizeWindow("Detected Location", 1200, 800)  # Adjust the size as needed

    # Keep the windows open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, calculate the approximate location coordinates within the large image
    location_center = np.mean(dst, axis=0)
    print("Estimated location of small image center:", location_center[0])
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
