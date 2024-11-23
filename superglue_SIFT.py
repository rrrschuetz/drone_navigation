import sys
sys.path.append('/home/rrrschuetz/SuperGluePretrainedNetwork')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.matching import Matching
from models.utils import frame2tensor

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load SuperGlue configuration with updated parameters
superglue_config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.001,  # Increased for better filtering
        'max_keypoints': 4096,
    },
    'superglue': {
        'weights': 'outdoor',
    },
}
matching = Matching(superglue_config).eval().to(device)

# Load images
large_image_path = 'bruchsal_highres.jpg'
small_image_path = ('luftbild2.jpg')

large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

if large_image is None or small_image is None:
    raise FileNotFoundError("One or both image paths are incorrect or the images could not be loaded.")

# Resize images while preserving aspect ratio
def resize_with_aspect_ratio(image, max_dim):
    h, w = image.shape
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

large_image_resized = resize_with_aspect_ratio(large_image, 4096)
small_image_resized = resize_with_aspect_ratio(small_image, 2048)

# Convert large image to tensor
large_tensor = frame2tensor(large_image_resized, device)

# Define a function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

# Initialize variables for the best match
angles = range(0, 360, 15)  # Test rotations at 15° intervals
best_homography = None
best_angle = 0
best_matches = 0
best_keypoints0 = None
best_keypoints1 = None

# Iterate over rotations of the small image
for angle in angles:
    print(f"Testing rotation angle: {angle}°")
    rotated_small = rotate_image(small_image_resized, angle)

    # Convert to tensor and perform SuperGlue matching
    small_tensor_rotated = frame2tensor(rotated_small, device)
    with torch.no_grad():
        pred = matching({'image0': small_tensor_rotated, 'image1': large_tensor})

    # Extract matches
    keypoints0 = pred['keypoints0'][0].cpu().numpy()
    keypoints1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    matches_conf = pred['matching_scores0'][0].cpu().numpy()

    # Filter matches based on confidence
    confidence_threshold = 0.05
    valid_matches = (matches > -1) & (matches_conf > confidence_threshold)
    keypoints0_valid = keypoints0[valid_matches]
    keypoints1_valid = keypoints1[matches[valid_matches]]

    # Check if this angle yields more matches
    if len(keypoints0_valid) > best_matches:
        best_matches = len(keypoints0_valid)
        best_angle = angle
        best_keypoints0 = keypoints0_valid
        best_keypoints1 = keypoints1_valid

        # Estimate homography if sufficient matches
        if len(best_keypoints0) >= 4:
            src_pts = np.float32(best_keypoints0).reshape(-1, 1, 2)
            dst_pts = np.float32(best_keypoints1).reshape(-1, 1, 2)
            best_homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Print results
if best_homography is not None:
    print(f"Best angle: {best_angle}°, Matches: {best_matches}")
else:
    print("Homography could not be estimated. Try adjusting parameters or preprocessing.")

# Draw the detected region if homography was found
if best_homography is not None:
    h, w = small_image_resized.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, best_homography)

    large_image_color = cv2.cvtColor(large_image_resized, cv2.COLOR_GRAY2BGR)
    detected_image = cv2.polylines(
        large_image_color, [np.int32(dst)], isClosed=True, color=(0, 255, 255), thickness=10
    )

    # Save and display the detected region
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Detected Region (Rotation: {best_angle}°)")
    plt.savefig("detected_region_rotation.png")
    plt.show()
else:
    print("No region detected due to insufficient matches.")

# Save keypoints visualization
if best_keypoints0 is not None and best_keypoints1 is not None:
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

    output_image = draw_custom_matches(small_image_resized, best_keypoints0, large_image_resized, best_keypoints1)

    # Save and display matches
    plt.figure(figsize=(15, 15))
    plt.imshow(output_image)
    plt.axis("off")
    plt.title("Best Matches")
    plt.savefig("best_matches.png")
    #plt.show()
