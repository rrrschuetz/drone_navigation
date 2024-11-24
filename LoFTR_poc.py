import sys
sys.path.append('/home/rrrschuetz/LoFTR')  # Update to the path of your R2D2 repository

import cv2
import torch
import numpy as np
from src.loftr import LoFTR
from src.utils import plotting as plot_matches

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

loftr = LoFTR(config=config)
loftr.load_state_dict(torch.load("outdoor.ckpt")["state_dict"])
loftr.eval()

# Load images (Ensure grayscale conversion)
large_image_path = "48MP-200.JPG"
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

# Visualize matches
def visualize_matches(image1, image2, keypoints1, keypoints2, matches, title="LoFTR Matches"):
    """Visualize the matched keypoints between two images."""
    matched_image = plot_matches(
        image1, image2, keypoints1, keypoints2, matches
    )
    cv2.imshow(title, matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

visualize_matches(
    large_image_resized, small_image_resized, keypoints_large, keypoints_small, matches
)
