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
config = {
    "backbone_type": "ResNetFPN",  # Backbone type
    "resnetfpn": {
        "initial_dim": 128,  # Initial feature map dimension
        "block_dims": [128, 196, 256],  # ResNet block output dimensions
    },
    "resolution": (8, 2),  # Output resolution (down-sampling factor, typical for LoFTR)
    "coarse": {
        "d_model": 256,  # Model dimension for coarse-level transformer
        "nhead": 8,  # Number of attention heads
        "layer_names": ["self", "cross"] * 6,  # Transformer layer structure
        "attention": "linear",  # Type of attention (e.g., 'linear' or 'softmax')
        "temp_bug_fix": False,  # Temporary bug fix
    },
    "match_coarse": {
        "border_rm": 2,  # Pixels to remove near the border
        "match_type": "dual_softmax",  # Matching method
        "dsmax_temperature": 0.1,  # Softmax temperature
        "train_coarse_percent": 0.4,  # Percentage of coarse matches used in training
        "train_pad_num_gt_min": 200,  # Minimum ground-truth matches during training
    },
    "fine_concat_coarse_feat": True,  # Whether to concatenate coarse features
    "fine_window_size": 5,  # Fine-level
    "fine": {
        "d_model": 128,  # Model dimension for fine-level transformer
        "nhead": 4,  # Number of attention heads
        "layer_names": ["self", "cross"] * 4,  # Transformer layer structure
        "attention": "softmax",  # Type of attention
    },
    "fine_matching": {
        "match_threshold": 0.5,  # Matching threshold
    },
    "match_coarse": {
        "border_rm": 4,  # Border pixels to remove
        "match_type": "dual_softmax",  # Matching method
        "dsmax_temperature": 0.1,  # Temperature for dual softmax
        "train_coarse_percent": 0.2,  # Fraction of coarse matches used in training
        "train_pad_num_gt_min": 200,  # Minimum ground-truth matches during training
        "train_pad_num_gt_max": 300,  # Maximum ground-truth matches during training
        "train_temp_bug_fix": False,  # Fix a known bug in training
        "thr": 0.2,  # Placed correctly within match_coarse
    },
    "fine_matching": {
        "threshold": 0.2,  # Threshold for fine matching
    },
}
#loftr = LoFTR(config=config)

# Load Kornia LoFTR model
loftr = LoFTR(pretrained="outdoor")

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

print(f"Number of matches: {len(matches)}")

def visualize_matches(img1, img2, kp1, kp2):
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

    # Combine keypoints from both images for clustering
    combined_keypoints = np.array(kp1_rescaled + kp2_adjusted)

    # Perform clustering (DBSCAN)
    clustering = DBSCAN(eps=30, min_samples=5).fit(combined_keypoints)
    labels = clustering.labels_

    # Identify the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    if largest_cluster_label == -1:  # If largest cluster is noise, ignore
        largest_cluster_label = unique_labels[np.argsort(counts)[-2]]

    # Filter keypoints in the largest cluster
    largest_cluster_points = combined_keypoints[labels == largest_cluster_label]

    # Create polygon around the largest cluster
    from scipy.spatial import ConvexHull
    hull = ConvexHull(largest_cluster_points)
    polygon = largest_cluster_points[hull.vertices]

    # Plot the matches and polygon
    plt.figure(figsize=(12, 8))
    plt.imshow(img_combined, cmap="gray")
    plt.axis("off")

    # Draw matches
    for (x1, y1), (x2, y2) in zip(kp1_rescaled, kp2_adjusted):
        plt.plot([x1, x2], [y1, y2], "r", linewidth=0.8)  # Line between matches
        plt.scatter([x1, x2], [y1, y2], color="cyan", s=5)  # Keypoints

    # Draw polygon
    polygon_patch = Polygon(polygon, closed=True, fill=False, edgecolor='yellow', linewidth=2)
    plt.gca().add_patch(polygon_patch)

    plt.title("Matches with Polygon Around Largest Cluster")
    plt.show()

visualize_matches(large_image_resized, small_image_resized, keypoints_large, keypoints_small)

