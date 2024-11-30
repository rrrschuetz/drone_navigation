import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_to_middle_80_percent(image):
    """
    Crop the top 10% and bottom 10% of the image, keeping the middle 80%.
    """
    height, width = image.shape
    top = int(height * 0.1)
    bottom = int(height * 0.9)
    return image[top:bottom, :]


def extract_relevant_keypoints(image, max_keypoints=50, hessian_threshold=1000):
    """
    Extract the most relevant keypoints from the image.
    """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)

    # Detect keypoints and descriptors
    keypoints, descriptors = surf.detectAndCompute(image, None)

    # Sort keypoints by response strength
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)

    # Keep only the top keypoints
    keypoints = keypoints[:max_keypoints]
    descriptors = descriptors[:max_keypoints] if descriptors is not None else None

    return keypoints, descriptors


def match_keypoints(descriptors_small, descriptors_large, ratio=0.7):
    """
    Match descriptors between small and large images.
    """
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_small, descriptors_large, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


def filter_matches_with_ransac(keypoints_small, keypoints_large, matches):
    """
    Filter matches using RANSAC to identify inliers.
    """
    if len(matches) < 4:  # At least 4 matches are required for homography
        return []

    # Extract points from the matches
    src_pts = np.float32([keypoints_small[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_large[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography with RANSAC
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Filter inlier matches
    inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
    return inlier_matches


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

    return image[y_min:y_max, x_min:x_max]


def draw_keypoints_and_matches(image, keypoints, matches, match_keypoints, match_color=(255, 0, 0), kp_color=(0, 255, 0)):
    """
    Draw all keypoints and matched keypoints on the image.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Draw all keypoints
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        #cv2.circle(output_image, (x, y), 8, kp_color, 2)

    # Highlight matched keypoints
    for match_kp in match_keypoints:
        x, y = int(match_kp.pt[0]), int(match_kp.pt[1])
        cv2.circle(output_image, (x, y), 12, match_color, 3)

    return output_image


def ensure_minimum_matches(small_image, large_image, min_matches=10, max_attempts=20):
    """
    Dynamically adjust parameters to ensure a minimum number of matches.
    """
    hessian_threshold = 1000
    ratio_test = 0.7
    attempts = 0

    while attempts < max_attempts:
        keypoints_small, descriptors_small = extract_relevant_keypoints(
            small_image, max_keypoints=100, hessian_threshold=hessian_threshold
        )
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        keypoints_large, descriptors_large = surf.detectAndCompute(large_image, None)

        good_matches = match_keypoints(descriptors_small, descriptors_large, ratio=ratio_test)

        # Filter matches using RANSAC
        inlier_matches = filter_matches_with_ransac(keypoints_small, keypoints_large, good_matches)

        print(f"Attempt {attempts + 1}: Found {len(inlier_matches)} inlier matches with Hessian={hessian_threshold}, Ratio={ratio_test}")

        if len(inlier_matches) >= min_matches:
            return keypoints_small, keypoints_large, inlier_matches

        # Adjust parameters
        hessian_threshold = max(100, hessian_threshold - 100)  # Lower Hessian threshold
        ratio_test = min(0.9, ratio_test + 0.05)  # Relax Lowe's ratio test
        attempts += 1

    raise RuntimeError(f"Unable to find at least {min_matches} inlier matches after {max_attempts} attempts.")


# Load the images
large_image = cv2.imread('satellite_image.jpg', cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread('small_image.jpg', cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Remove upper 10% and lower 10% from the images
large_image = crop_to_middle_80_percent(large_image)
small_image = crop_to_middle_80_percent(small_image)

try:
    # Ensure minimum number of matches
    keypoints_small, keypoints_large, inlier_matches = ensure_minimum_matches(
        small_image, large_image, min_matches=10
    )

    # Extract matched keypoints
    matched_keypoints_small = [keypoints_small[m.queryIdx] for m in inlier_matches]
    matched_keypoints_large = [keypoints_large[m.trainIdx] for m in inlier_matches]

    # Draw keypoints and matches on the small image
    small_image_with_matches = draw_keypoints_and_matches(
        small_image, keypoints_small, inlier_matches, matched_keypoints_small
    )

    # Crop the large image to the relevant region and resize
    cropped_image = crop_to_matches(large_image, keypoints_large, inlier_matches, margin=20)
    resized_cropped_image = cv2.resize(cropped_image, (small_image.shape[1], small_image.shape[0]))

    # Draw matches on the resized cropped image
    resized_cropped_with_matches = draw_keypoints_and_matches(
        resized_cropped_image, keypoints_large, inlier_matches, matched_keypoints_large
    )

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Small Image: Keypoints and Matches")
    plt.imshow(cv2.cvtColor(small_image_with_matches, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Cropped Large Image: Keypoints and Matches")
    plt.imshow(cv2.cvtColor(resized_cropped_with_matches, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

except RuntimeError as e:
    print(e)
except ValueError as e:
    print(f"Error during cropping: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
