# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
import torch
from scipy.spatial.distance import cdist

# Load images
# Replace 'satellite_image.jpg' and 'drone_image.jpg' with your image file paths
satellite_image_path = 'satellite_image.jpg'
drone_image_path = 'small_image.jpg'

satellite_image = cv2.imread(satellite_image_path)
drone_image = cv2.imread(drone_image_path)

# Check if images are loaded correctly
if satellite_image is None or drone_image is None:
    raise FileNotFoundError("One or both image paths are incorrect.")

# Convert images from BGR to RGB
satellite_image_rgb = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2RGB)
drone_image_rgb = cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB)

# Initialize the model
# Specify the configuration and checkpoint files
config_file = 'https://raw.githubusercontent.com/open-mmlab/mmsegmentation/master/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210630_110128-2a2dd774.pth'

# Initialize the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_segmentor(config_file, checkpoint_file, device=device)

# Run inference
result_satellite = inference_segmentor(model, satellite_image_rgb)
result_drone = inference_segmentor(model, drone_image_rgb)

# Get class names from the model
class_names = model.CLASSES

# Find class indices for 'building' and 'road'
building_class_name = 'building'
road_class_name = 'road'

# Ensure class names are in the class_names list
if building_class_name in class_names:
    building_class_index = class_names.index(building_class_name)
else:
    raise ValueError(f"Class '{building_class_name}' not found in model classes.")

if road_class_name in class_names:
    road_class_index = class_names.index(road_class_name)
else:
    raise ValueError(f"Class '{road_class_name}' not found in model classes.")

print(f"'Building' class index: {building_class_index}")
print(f"'Road' class index: {road_class_index}")

# Classes of interest
classes_of_interest = [building_class_index, road_class_index]

# For satellite image
satellite_prediction = result_satellite[0]  # The segmentation map
satellite_mask = np.isin(satellite_prediction, classes_of_interest).astype(np.uint8)

# For drone image
drone_prediction = result_drone[0]
drone_mask = np.isin(drone_prediction, classes_of_interest).astype(np.uint8)

# Apply morphological operations to clean up the masks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
satellite_mask = cv2.morphologyEx(satellite_mask, cv2.MORPH_CLOSE, kernel)
drone_mask = cv2.morphologyEx(drone_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in satellite mask
contours_satellite, _ = cv2.findContours(
    satellite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Find contours in drone mask
contours_drone, _ = cv2.findContours(
    drone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Draw contours on the images
satellite_image_contours = satellite_image_rgb.copy()
drone_image_contours = drone_image_rgb.copy()

cv2.drawContours(satellite_image_contours, contours_satellite, -1, (255, 0, 0), 2)
cv2.drawContours(drone_image_contours, contours_drone, -1, (255, 0, 0), 2)

# Display the images with contours
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(satellite_image_contours)
plt.title('Satellite Image with Contours')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(drone_image_contours)
plt.title('Drone Image with Contours')
plt.axis('off')

plt.show()

# Function to compute centroids of contours
def compute_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids

# Compute centroids
centroids_satellite = compute_centroids(contours_satellite)
centroids_drone = compute_centroids(contours_drone)

# Convert centroids to NumPy arrays
centroids_satellite_np = np.array(centroids_satellite)
centroids_drone_np = np.array(centroids_drone)

# Check if centroids are available
if len(centroids_satellite_np) == 0 or len(centroids_drone_np) == 0:
    print("No centroids found in one or both images.")
else:
    # Compute distance matrix between centroids
    distance_matrix = cdist(centroids_drone_np, centroids_satellite_np)

    # Match centroids based on minimum distance
    matches = []
    for i in range(len(centroids_drone_np)):
        min_idx = np.argmin(distance_matrix[i])
        min_distance = distance_matrix[i][min_idx]
        matches.append((i, min_idx, min_distance))

    # Set a distance threshold for matching (adjust as needed)
    distance_threshold = 100  # Pixels
    good_matches = [m for m in matches if m[2] < distance_threshold]

    # Create a combined image for visualization
    combined_image = np.hstack((drone_image_rgb, satellite_image_rgb))
    offset = drone_image_rgb.shape[1]  # Offset for the satellite image in the combined image

    # Draw lines between matched centroids
    for match in good_matches:
        idx_drone = match[0]
        idx_satellite = match[1]

        cX_drone, cY_drone = centroids_drone_np[idx_drone]
        cX_satellite, cY_satellite = centroids_satellite_np[idx_satellite]
        cX_satellite += offset  # Adjust x-coordinate for combined image

        # Draw a line between the matched centroids
        cv2.line(
            combined_image,
            (cX_drone, cY_drone),
            (cX_satellite, cY_satellite),
            (0, 255, 0),
            2
        )

    # Display the matched centroids
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image)
    plt.title('Matched Structures Between Drone and Satellite Images')
    plt.axis('off')
    plt.show()

    # Save the combined image with matches (Optional)
    cv2.imwrite('matched_structures.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

# Save the images with contours (Optional)
cv2.imwrite('satellite_with_contours.jpg', cv2.cvtColor(satellite_image_contours, cv2.COLOR_RGB2BGR))
cv2.imwrite('drone_with_contours.jpg', cv2.cvtColor(drone_image_contours, cv2.COLOR_RGB2BGR))
