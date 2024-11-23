
import cv2
import numpy as np
import matplotlib.pyplot as plt

# File paths for the images
luftbild5_path = "luftbild6.jpg"
bruchsal_highres_path = "bruchsal_highres.jpg"
small_image8_path = "small_imag8.jpg"

# Load images
luftbild5 = cv2.imread(luftbild5_path, cv2.IMREAD_GRAYSCALE)
bruchsal_highres = cv2.imread(bruchsal_highres_path, cv2.IMREAD_GRAYSCALE)
small_image8 = cv2.imread(small_image8_path, cv2.IMREAD_GRAYSCALE)

# Check if images loaded successfully
if luftbild5 is None or bruchsal_highres is None or small_image8 is None:
    raise FileNotFoundError("One or more images could not be loaded.")

# Define preprocessing function
def preprocess_image(image):
    enhanced = cv2.equalizeHist(image)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# Preprocess images
luftbild5_preprocessed = preprocess_image(luftbild5)
bruchsal_highres_preprocessed = preprocess_image(bruchsal_highres)
small_image8_preprocessed = preprocess_image(small_image8)

# Resize function
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Resize images
luftbild5_resized = resize_with_aspect_ratio(luftbild5_preprocessed, 1024)
bruchsal_highres_resized = resize_with_aspect_ratio(bruchsal_highres_preprocessed, 4096)
small_image8_resized = resize_with_aspect_ratio(small_image8_preprocessed, 1024)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints_luftbild5, descriptors_luftbild5 = sift.detectAndCompute(luftbild5_resized, None)
keypoints_bruchsal, descriptors_bruchsal = sift.detectAndCompute(bruchsal_highres_resized, None)
keypoints_small_image8, descriptors_small_image8 = sift.detectAndCompute(small_image8_resized, None)

# Visualization function for keypoints
def visualize_keypoints(image, keypoints, title):
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_keypoints, cmap="gray")
    plt.title(f"{title} - Keypoints: {len(keypoints)}")
    plt.axis("off")
    plt.show()

# Visualize keypoints for all images
visualize_keypoints(luftbild5_resized, keypoints_luftbild5, "Luftbild5 Keypoints")
visualize_keypoints(bruchsal_highres_resized, keypoints_bruchsal, "Bruchsal Highres Keypoints")
visualize_keypoints(small_image8_resized, keypoints_small_image8, "Small Image8 Keypoints")

# Match descriptors and visualize matches
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Match descriptors between luftbild5 and the large image
matches_luftbild5 = bf.knnMatch(descriptors_luftbild5, descriptors_bruchsal, k=2)

# Apply Lowe's ratio test for luftbild5
good_matches_luftbild5 = [m for m, n in matches_luftbild5 if m.distance < 0.7 * n.distance]

# Match descriptors between small_image8 and the large image
matches_small_image8 = bf.knnMatch(descriptors_small_image8, descriptors_bruchsal, k=2)

# Apply Lowe's ratio test for small_image8
good_matches_small_image8 = [m for m, n in matches_small_image8 if m.distance < 0.7 * n.distance]

# Visualization function for matches
def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Visualize matches
visualize_matches(luftbild5_resized, keypoints_luftbild5, bruchsal_highres_resized, keypoints_bruchsal, good_matches_luftbild5, "Luftbild5 Matches")
visualize_matches(small_image8_resized, keypoints_small_image8, bruchsal_highres_resized, keypoints_bruchsal, good_matches_small_image8, "Small Image8 Matches")

# Output match counts
print(f"Luftbild5 matches: {len(good_matches_luftbild5)}")
print(f"Small Image8 matches: {len(good_matches_small_image8)}")