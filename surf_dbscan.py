import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def crop_to_middle_80_percent(image):
    """
    Crop the top 10% and bottom 10% of the image, keeping the middle 80%.
    """
    height, width = image.shape
    top = int(height * 0.1)
    bottom = int(height * 0.9)
    return image[top:bottom, :], top


def extract_relevant_keypoints(image_gpu, hessian_threshold=1000):
    """
    Extract keypoints and descriptors from the image using CUDA SURF or ORB.
    """
    try:
        # Try to create a CUDA SURF detector
        surf = cv2.cuda.SURF_CUDA_create(hessian_threshold)
        # Flag indicating the use of SURF
        is_surf = True
    except AttributeError:
        print("CUDA SURF not available. Switching to CUDA ORB.")
        # Use CUDA ORB detector as a fallback
        surf = cv2.cuda_ORB.create()
        is_surf = False

    # Detect keypoints and compute descriptors
    keypoints_gpu, descriptors_gpu = surf.detectWithDescriptors(image_gpu, None)

    # Download keypoints from GPU to CPU
    if is_surf:
        keypoints = surf.downloadKeypoints(keypoints_gpu)
    else:
        keypoints = surf.convert(keypoints_gpu)

    return keypoints, descriptors_gpu, is_surf


def match_keypoints(descriptors_small_gpu, descriptors_large_gpu, is_surf, ratio=0.7):
    """
    Match descriptors between small and large images using CPU-based matcher.
    """
    # Download descriptors from GPU to CPU
    descriptors_small = descriptors_small_gpu.download()
    descriptors_large = descriptors_large_gpu.download()

    # Determine norm type based on descriptor type
    if is_surf:
        # SURF descriptors are float32
        norm_type = cv2.NORM_L2
        index_params = dict(algorithm=1, trees=5)
    else:
        # ORB descriptors are uint8
        norm_type = cv2.NORM_HAMMING
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN parameters for ORB

    # Use FLANN-based matcher
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(descriptors_small, descriptors_large, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def spatial_filter_matches(matches, keypoints_large, eps=50, min_samples=3):
    """
    Filter matches based on spatial consistency using DBSCAN clustering.
    """
    # Extract coordinates of matched points in the larger image
    points = np.array([keypoints_large[m.trainIdx].pt for m in matches])

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Identify the largest cluster
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    if largest_cluster_label == -1:
        raise ValueError("No valid clusters found in matches.")

    # Filter matches belonging to the largest cluster
    filtered_matches = [m for m, label in zip(matches, labels) if label == largest_cluster_label]
    return filtered_matches


def crop_to_matches(image, keypoints, matches, margin=20):
    """
    Crop the larger image to the region containing all matched keypoints.
    """
    matched_coords = np.array([keypoints[m.trainIdx].pt for m in matches])

    x_min, y_min = np.min(matched_coords, axis=0).astype(int)
    x_max, y_max = np.max(matched_coords, axis=0).astype(int)

    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)

    # Ensure the crop region is valid
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid crop region. Check matched keypoints.")

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, x_min, y_min, (x_max - x_min), (y_max - y_min)


def adjust_keypoints_for_crop_and_resize(keypoints, x_min, y_min, crop_width, crop_height, output_width, output_height):
    """
    Adjust keypoint coordinates for cropping and resizing.
    """
    x_scale = output_width / crop_width
    y_scale = output_height / crop_height

    adjusted_keypoints = []
    for kp in keypoints:
        x, y = kp.pt
        adjusted_x = (x - x_min) * x_scale
        adjusted_y = (y - y_min) * y_scale
        adjusted_keypoints.append(cv2.KeyPoint(adjusted_x, adjusted_y, kp.size))
    return adjusted_keypoints


def ensure_minimum_matches(small_image_gpu, large_image_gpu, min_matches=10, max_attempts=10):
    """
    Dynamically adjust parameters to ensure a minimum number of matches.
    """
    hessian_threshold = 1000
    ratio_test = 0.6
    attempts = 0

    while attempts < max_attempts:
        # Extract keypoints and descriptors
        keypoints_small, descriptors_small_gpu, is_surf = extract_relevant_keypoints(
            small_image_gpu, hessian_threshold=hessian_threshold
        )
        keypoints_large, descriptors_large_gpu, _ = extract_relevant_keypoints(
            large_image_gpu, hessian_threshold=hessian_threshold
        )

        # Match keypoints
        good_matches = match_keypoints(descriptors_small_gpu, descriptors_large_gpu, is_surf, ratio=ratio_test)

        # Apply spatial filtering to matches
        try:
            filtered_matches = spatial_filter_matches(good_matches, keypoints_large, eps=100, min_samples=20)
        except ValueError:
            filtered_matches = []

        print(f"Attempt {attempts + 1}: Found {len(filtered_matches)} spatially consistent matches "
              f"with Hessian={hessian_threshold}, Ratio={ratio_test}")

        if len(filtered_matches) >= min_matches:
            return keypoints_small, keypoints_large, filtered_matches

        # Adjust parameters
        hessian_threshold = max(100, hessian_threshold - 100)  # Lower Hessian threshold
        ratio_test = min(0.9, ratio_test + 0.05)  # Relax Lowe's ratio test
        attempts += 1

    return keypoints_small, keypoints_large, None


def draw_keypoints_and_matches(image, keypoints, matched_keypoints, kp_color=(0, 255, 0),
                               match_color=(255, 0, 0)):
    """
    Draw all keypoints and matched keypoints on the image.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw all keypoints
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output_image, (x, y), 8, kp_color, 2)

    # Highlight matched keypoints
    if matched_keypoints is not None:
        for match_kp in matched_keypoints:
            x, y = int(match_kp.pt[0]), int(match_kp.pt[1])
            cv2.circle(output_image, (x, y), 12, match_color, 3)

    return output_image


# Load the images
#large_image = cv2.imread('satellite_image.jpg', cv2.IMREAD_GRAYSCALE)
#small_image = cv2.imread('small_imag3.jpg', cv2.IMREAD_GRAYSCALE)

large_image = cv2.imread('karlsdorf_highres4.jpg', cv2.IMREAD_GRAYSCALE)
#small_image = cv2.imread('karlsdorf_highres2.jpg', cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread('normal-120.JPG', cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

large_image = cv2.equalizeHist(large_image)
small_image = cv2.equalizeHist(small_image)

# Remove upper 10% and lower 10% from the images
large_image, large_top_offset = crop_to_middle_80_percent(large_image)
small_image, small_top_offset = crop_to_middle_80_percent(small_image)

# Upload images to GPU
large_image_gpu = cv2.cuda_GpuMat()
large_image_gpu.upload(large_image)

small_image_gpu = cv2.cuda_GpuMat()
small_image_gpu.upload(small_image)

try:
    # Ensure minimum number of spatially consistent matches
    keypoints_small, keypoints_large, filtered_matches = ensure_minimum_matches(
        small_image_gpu, large_image_gpu, min_matches=20, max_attempts=10
    )

    if filtered_matches is not None:
        # Crop the large image to the relevant region
        cropped_image, x_min, y_min, crop_width, crop_height = crop_to_matches(
            large_image, keypoints_large, filtered_matches, margin=0
        )

        # Resize the cropped image to match the small image's height
        aspect_ratio = crop_width / crop_height
        output_height = small_image.shape[0]
        output_width = int(output_height * aspect_ratio)
        resized_cropped_image = cv2.resize(cropped_image, (output_width, output_height))

        # Adjust keypoints for the cropped and resized image
        adjusted_keypoints = adjust_keypoints_for_crop_and_resize(
            keypoints_large, x_min, y_min, crop_width, crop_height, output_width, output_height
        )

        # Draw keypoints and matches on the small image
        matched_keypoints_small = [keypoints_small[m.queryIdx] for m in filtered_matches]
        small_image_with_matches = draw_keypoints_and_matches(
            small_image, keypoints_small, matched_keypoints_small, kp_color=(0, 255, 0), match_color=(255, 0, 0)
        )

        # Draw keypoints and matches on the resized cropped large image
        matched_keypoints_large = [adjusted_keypoints[m.trainIdx] for m in filtered_matches]
        resized_cropped_with_matches = draw_keypoints_and_matches(
            resized_cropped_image, adjusted_keypoints, matched_keypoints_large, kp_color=(0, 255, 0),
            match_color=(255, 0, 0)
        )
    else:
        small_image_with_matches = draw_keypoints_and_matches(
            small_image, keypoints_small, None, kp_color=(0, 255, 0), match_color=(255, 0, 0)
        )

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Small Image with Matches")
    plt.imshow(cv2.cvtColor(small_image_with_matches, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    if filtered_matches is not None:
        plt.subplot(1, 2, 2)
        plt.title("Cropped and Resized Large Image with Matches")
        plt.imshow(cv2.cvtColor(resized_cropped_with_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.show()

except RuntimeError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")
