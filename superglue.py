import sys
sys.path.append('/home/rrrschuetz/SuperGluePretrainedNetwork')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from models.matching import Matching
from models.utils import frame2tensor

# Set matplotlib backend
try:
    matplotlib.use('TkAgg')  # Interactive backend
except ImportError:
    matplotlib.use('Agg')  # Headless fallback

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load SuperGlue configuration
superglue_config = {
    'superpoint': {
        'nms_radius': 3,                # Adjusted for finer detection
        'keypoint_threshold': 0.005,    # Lower threshold for more keypoints
        'max_keypoints': 2048,          # Allow more keypoints
    },
    'superglue': {
        'weights': 'outdoor',           # Experiment with 'indoor' as needed
    },
}
matching = Matching(superglue_config).eval().to(device)

# Load images
large_image_path = 'satellite_image.jpg'  # Replace with the path to the large satellite image
small_image_path = 'small_image.jpg'      # Replace with the path to the smaller image

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect or the images could not be loaded.")

# Resize images proportionally
def resize_image(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_image(large_image, 1024)
small_image_resized = resize_image(small_image, 512)

# Resize images to manageable dimensions
resize_large = (2048, 2048)  # Resize large image to a square resolution
resize_small = (1024, 1024)    # Resize small image to a smaller resolution

large_image_resized = cv2.resize(large_image, resize_large, interpolation=cv2.INTER_AREA)
small_image_resized = cv2.resize(small_image, resize_small, interpolation=cv2.INTER_AREA)

# Convert images to tensors
large_tensor = frame2tensor(large_image_resized, device)
small_tensor = frame2tensor(small_image_resized, device)

# Perform matching
with torch.no_grad():
    pred = matching({'image0': small_tensor, 'image1': large_tensor})

# Extract matches and keypoints
keypoints0 = pred['keypoints0'][0].cpu().numpy()
keypoints1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
matches_conf = pred['matching_scores0'][0].cpu().numpy()

# Print the number of keypoints and matches
print(f"Keypoints in small image: {len(keypoints0)}")
print(f"Keypoints in large image: {len(keypoints1)}")
print(f"Matches: {np.sum(matches > -1)}")

# Filter matches based on confidence
confidence_threshold = 0.01
valid_matches = (matches > -1) & (matches_conf > confidence_threshold)
keypoints0_valid = keypoints0[valid_matches]
keypoints1_valid = keypoints1[matches[valid_matches]]

if len(keypoints0_valid) == 0 or len(keypoints1_valid) == 0:
    print("No valid matches found. Try tuning the parameters.")
    exit()

# Visualization function for matches
def draw_custom_matches(img1, kpts1, img2, kpts2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    output[:h2, w1:w1 + w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for pt1, pt2 in zip(kpts1, kpts2):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2 + np.array([w1, 0])))  # Adjust for second image offset
        color = (0, 255, 0)
        cv2.line(output, pt1, pt2, color, 2)
        cv2.circle(output, pt1, 5, (255, 0, 0), -1)
        cv2.circle(output, pt2, 5, (255, 0, 0), -1)

    return output

# Call custom visualization
output_image = draw_custom_matches(small_image_resized, keypoints0_valid, large_image_resized, keypoints1_valid)

# Save and display the image
plt.figure(figsize=(15, 15))
plt.imshow(output_image)
plt.axis("off")
plt.title("Custom Matches")
plt.savefig("custom_matches.png")
plt.show()

# Estimate homography
src_pts = np.float32(keypoints0_valid).reshape(-1, 1, 2)
dst_pts = np.float32(keypoints1_valid).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw the detected region
h, w = small_image_resized.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

large_image_color = cv2.cvtColor(large_image_resized, cv2.COLOR_GRAY2BGR)
detected_image = cv2.polylines(
    large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=10
)

# Save and display the detected region
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Region with SuperGlue")
plt.savefig("detected_region.png")
plt.show()

# Print the center of the detected region
location_center = np.mean(dst, axis=0)
print("Estimated location of small image center:", location_center[0])
