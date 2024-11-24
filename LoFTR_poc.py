from kornia.feature import LoFTR
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

# Initialize LoFTR model with outdoor weights
loftr = LoFTR(pretrained="outdoor")
loftr.config['match_coarse']['thr'] = 0.2

# Load checkpoint
checkpoint = torch.hub.load_state_dict_from_url(
    "https://github.com/zju3dv/LoFTR/releases/download/1.0.0/loftr_outdoor.ckpt",
    map_location="cpu"
)
loftr.load_state_dict(checkpoint["state_dict"])
loftr.eval()

# Load images (Ensure grayscale conversion)
large_image_path = "48MP-200.JPG"
#large_image_path = "karlsdorf_highres2.jpg"
#large_image_path = "bruchsal.jpg"
#small_image_path = "small_imag5.jpg"
small_image_path = "normal-120.JPG"

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Resize images for processing (optional but recommended)
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image, 1024)
small_image_resized = resize_with_aspect_ratio(small_image, 1024)

# Convert images to tensors
large_tensor = torch.from_numpy(large_image_resized / 255.0).float().unsqueeze(0).unsqueeze(0)
small_tensor = torch.from_numpy(small_image_resized / 255.0).float().unsqueeze(0).unsqueeze(0)

# Move tensors to GPU if available
if torch.cuda.is_available():
    loftr = loftr.cuda()
    large_tensor = large_tensor.cuda()
    small_tensor = small_tensor.cuda()

# Forward pass through LoFTR
with torch.no_grad():
    input_dict = {"image0": large_tensor, "image1": small_tensor}
    correspondences = loftr(input_dict)

# Extract matches
keypoints_large = correspondences["keypoints0"].cpu().numpy()
keypoints_small = correspondences["keypoints1"].cpu().numpy()
matches = correspondences["confidence"].cpu().numpy()

# Apply RANSAC to filter matches
matrix, mask = cv2.findHomography(keypoints_large, keypoints_small, cv2.RANSAC, 5.0)
inliers_mask = mask.ravel() == 1

# Filter inlier keypoints
keypoints_large_inliers = keypoints_large[inliers_mask]
keypoints_small_inliers = keypoints_small[inliers_mask]
matches_inliers = matches[inliers_mask]

print(f"Number of matches: {len(matches)}")
print(f"Number of inliers: {len(matches_inliers)}")

def visualize_inlier_matches(img1, img2, kp1, kp2):
    # Calculate resizing ratios
    height = min(img1.shape[0], img2.shape[0])  # Target height
    scale1 = height / img1.shape[0]
    scale2 = height / img2.shape[0]

    # Resize images
    img1_resized = cv2.resize(img1, (int(img1.shape[1] * scale1), height))
    img2_resized = cv2.resize(img2, (int(img2.shape[1] * scale2), height))

    # Rescale keypoints
    kp1_rescaled = [(x * scale1, y * scale1) for (x, y) in kp1]
    kp2_rescaled = [(x * scale2, y * scale2) for (x, y) in kp2]

    # Combine images side-by-side
    img_combined = cv2.hconcat([img1_resized, img2_resized])
    h1, w1 = img1_resized.shape[:2]

    # Adjust second keypoints' x-coordinates for concatenated image
    kp2_adjusted = [(x + w1, y) for (x, y) in kp2_rescaled]

    # Plot the matches
    plt.figure(figsize=(12, 8))
    plt.imshow(img_combined, cmap="gray")
    plt.axis("off")

    # Draw inlier matches
    for (x1, y1), (x2, y2) in zip(kp1_rescaled, kp2_adjusted):
        plt.plot([x1, x2], [y1, y2], "lime", linewidth=1.0)  # Line between inlier matches
        plt.scatter([x1, x2], [y1, y2], color="red", s=5)  # Keypoints

    plt.title("Inlier Matches After RANSAC")
    plt.show()

# Visualize the inliers
visualize_inlier_matches(large_image_resized, small_image_resized, keypoints_large_inliers, keypoints_small_inliers)

