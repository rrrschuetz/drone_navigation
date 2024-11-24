import sys
sys.path.append('/home/rrrschuetz/d2_net/')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
import torch

# Load images
#large_image_path = "karlsdorf_highres2.jpg"
large_image_path = "48MP-200.JPG"
small_image_path = "normal-120.JPG"

large_image = cv2.imread(large_image_path, cv2.COLOR_BGR2GRAY)
small_image = cv2.imread(small_image_path, cv2.COLOR_BGR2GRAY)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")
print("Images loaded.")

# Preprocessing function
def preprocess(image, is_large):
    # Convert to grayscale
    if len(image.shape) == 3:  # Check if image is multi-channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Normalize if needed
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return enhanced


# Preprocess both images
large_image_preprocessed = preprocess(large_image, is_large=True)
small_image_preprocessed = preprocess(small_image, is_large=False)

# Resize images
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image_preprocessed, 1024)
small_image_resized = resize_with_aspect_ratio(small_image_preprocessed, 1024)
print("Images loaded and preprocessed.")

# D2-Net Model Initialization
model = D2Net(model_file='/home/rrrschuetz/d2_net/models/d2_tf.pth')  # Use the path to your pretrained model
model = model.cuda() if torch.cuda.is_available() else model
print("Model loaded.")

def extract(input_image, model):
    # Ensure input image has 3 channels (convert grayscale to RGB)
    if len(input_image.shape) == 2:  # Grayscale image
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    # Convert NumPy array to PyTorch tensor
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

    # Normalize and add batch dimension
    input_tensor = input_tensor / 255.0  # Normalize to [0, 1]
    input_tensor = input_tensor.unsqueeze(0)  # [B, C, H, W]

    # Move to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Extract features using the model
    with torch.no_grad():
        # Ensure the model is on the desired device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Move input image to the same device as the model
        input_tensor = input_tensor.to(device)
        features = process_multiscale(input_tensor, model)  # Ensure `process_multiscale` handles tensors correctly

    return features

large_features = extract(large_image_resized, model)
small_features = extract(small_image_resized, model)

# Match descriptors using BFMatcher
def match_features(features1, features2):
    # Unpack the tuples
    kp1, scores1, desc1 = features1
    kp2, scores2, desc2 = features2

    # Use a nearest-neighbor search to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # Sort matches by distance (optional)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract keypoints for visualization
    keypoints1 = kp1[:, :2]  # Only (x, y) coordinates
    keypoints2 = kp2[:, :2]
    good_matches = matches  # You can filter matches further if needed

    return keypoints1, keypoints2, good_matches

keypoints1, keypoints2, good_matches = match_features(small_features, large_features)

def filter_good_matches(matches, threshold=0.75):
    return [m for m in matches if m.distance < threshold]

good_matches = filter_good_matches(good_matches, threshold=0.75)  # Adjust threshold as needed

def resize_with_fixed_height(image, target_height):
    h, w = image.shape[:2]
    print(target_height, h)
    scale = target_height / h
    new_width = int(w * scale)
    return scale, cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)

def visualize_matches(img1, img2, kp1, kp2, matches):
    # Resize images to the same height
    target_height = min(img1.shape[0], img2.shape[0])
    scale1, img1_resized = resize_with_fixed_height(img1, target_height)
    scale2, img2_resized = resize_with_fixed_height(img2, target_height)

    kp1_rescaled = [(kp[0] * scale1 , kp[1] * scale1 *0.5) for kp in kp1]
    kp2_rescaled = [(kp[0] * scale2 , kp[1] * scale2 *0.5) for kp in kp2]
    kp1 = kp1_rescaled
    kp2 = kp2_rescaled

    # Ensure data types match
    img1_resized = img1_resized.astype('uint8')
    img2_resized = img2_resized.astype('uint8')

    # Concatenate images horizontally
    combined_image = cv2.hconcat([img1_resized, img2_resized])

    # Adjust second image keypoints
    offset_width = img1_resized.shape[1]
    kp2_adjusted = [(x + offset_width, y) for (x, y) in kp2]

    # Plot matches
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    for match in matches:
        x1, y1 = kp1[match.queryIdx]
        x2, y2 = kp2_adjusted[match.trainIdx]
        plt.plot([x1, x2], [y1, y2], "r", linewidth=0.8)
        plt.scatter([x1, x2], [y1, y2], color="cyan", s=5)

    plt.title("Feature Matches")
    plt.show()


print(f"Large image original: {large_image.shape}, resized: {large_image_resized.shape}")
print(f"Small image original: {small_image.shape}, resized: {small_image_resized.shape}")

visualize_matches(small_image_resized, large_image_resized, keypoints1, keypoints2, good_matches)

# Check for homography if enough matches are found
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Draw detected region
    h, w = small_image_resized.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    large_image_color = cv2.cvtColor(large_image_resized, cv2.COLOR_GRAY2BGR)
    detected_image = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=5
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(detected_image)
    plt.title("Detected Region")
    plt.axis("off")
    plt.show()
else:
    print(f"Not enough matches: {len(good_matches)}")
