# Importieren der erforderlichen Bibliotheken
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import warnings
import os

# Warnungen unterdrücken
warnings.filterwarnings('ignore')

# Gerät auswählen (CPU oder GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Verwende Gerät: {device}')

# Laden der Bilder
satellite_image_path = 'satellite_image.jpg'  # Pfad zum Satellitenbild
drone_image_path = 'small_image.jpg'          # Pfad zum Drohnenbild

satellite_image = cv2.imread(satellite_image_path)
drone_image = cv2.imread(drone_image_path)

# Überprüfen, ob die Bilder korrekt geladen wurden
if satellite_image is None or drone_image is None:
    raise FileNotFoundError("Eines oder beide Bilder konnten nicht geladen werden.")

# Konvertieren von BGR zu RGB
satellite_image_rgb = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2RGB)
drone_image_rgb = cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB)

# Verwenden der Originalbildgröße
# Wenn die Bilder zu groß sind, können Sie sie auf eine höhere Größe wie (1024, 1024) skalieren
# Achten Sie darauf, dass genügend Speicher verfügbar ist

# Initialisieren des Feature Extractors und des Modells
feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')

# Modell auf das Gerät verschieben
model.to(device)
model.eval()

# Vorverarbeitung der Bilder
inputs_satellite = feature_extractor(images=satellite_image_rgb, return_tensors="pt").to(device)
inputs_drone = feature_extractor(images=drone_image_rgb, return_tensors="pt").to(device)

# Inferenz durchführen
with torch.no_grad():
    outputs_satellite = model(**inputs_satellite)
    outputs_drone = model(**inputs_drone)

# Abrufen der Logits
logits_satellite = outputs_satellite.logits  # Form [1, num_classes, H, W]
logits_drone = outputs_drone.logits          # Form [1, num_classes, H, W]

# Wahrscheinlichkeiten berechnen
probs_satellite = torch.nn.functional.softmax(logits_satellite, dim=1)[0].cpu().numpy()
probs_drone = torch.nn.functional.softmax(logits_drone, dim=1)[0].cpu().numpy()

# Klassenlabels abrufen
id2label = model.config.id2label

# Finden der Klassenindizes für 'building' und 'road'
building_class_name = 'building'
road_class_name = 'road'

building_class_index = None
road_class_index = None

for idx, label in id2label.items():
    if label.lower() == building_class_name:
        building_class_index = idx
    if label.lower() == road_class_name:
        road_class_index = idx

if building_class_index is None or road_class_index is None:
    raise ValueError("Klassenindizes für 'building' und/oder 'road' nicht gefunden.")

print(f"'Building' Klassenindex: {building_class_index}")
print(f"'Road' Klassenindex: {road_class_index}")

# Schwellenwert setzen, um binäre Masken zu erstellen
threshold = 0.5  # Kann angepasst werden

satellite_mask = ((probs_satellite[building_class_index] > threshold) |
                  (probs_satellite[road_class_index] > threshold)).astype(np.uint8)
drone_mask = ((probs_drone[building_class_index] > threshold) |
              (probs_drone[road_class_index] > threshold)).astype(np.uint8)

# Optional: CRF-Nachbearbeitung anwenden, um die Masken zu verfeinern
# Installieren Sie pydensecrf mit: pip install pydensecrf

# CRF-Code hier einfügen, wenn gewünscht

# Morphologische Operationen anpassen, um feinere Details zu erhalten
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Kleinere Kernelgröße
satellite_mask_cleaned = cv2.morphologyEx(satellite_mask, cv2.MORPH_OPEN, kernel)
drone_mask_cleaned = cv2.morphologyEx(drone_mask, cv2.MORPH_OPEN, kernel)

# Konturen in den Masken finden
contours_satellite, _ = cv2.findContours(
    satellite_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
contours_drone, _ = cv2.findContours(
    drone_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

# Konturen auf die Bilder zeichnen
satellite_image_contours = satellite_image_rgb.copy()
drone_image_contours = drone_image_rgb.copy()

cv2.drawContours(satellite_image_contours, contours_satellite, -1, (255, 0, 0), 2)
cv2.drawContours(drone_image_contours, contours_drone, -1, (255, 0, 0), 2)

# Bilder mit Konturen anzeigen
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(satellite_image_contours)
plt.title('Satellitenbild mit Konturen')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(drone_image_contours)
plt.title('Drohnenbild mit Konturen')
plt.axis('off')

plt.show()

# Funktion zum Berechnen der Konturzentroide
def compute_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids

# Zentren berechnen
centroids_satellite = compute_centroids(contours_satellite)
centroids_drone = compute_centroids(contours_drone)

# Zentren in NumPy-Arrays umwandeln
centroids_satellite_np = np.array(centroids_satellite)
centroids_drone_np = np.array(centroids_drone)

# Überprüfen, ob Zentren vorhanden sind
if len(centroids_satellite_np) == 0 or len(centroids_drone_np) == 0:
    print("Keine Zentren in einem oder beiden Bildern gefunden.")
else:
    # Distanzmatrix zwischen den Zentren berechnen
    distance_matrix = cdist(centroids_drone_np, centroids_satellite_np)

    # Zentren basierend auf minimaler Distanz zuordnen
    matches = []
    for i in range(len(centroids_drone_np)):
        min_idx = np.argmin(distance_matrix[i])
        min_distance = distance_matrix[i][min_idx]
        matches.append((i, min_idx, min_distance))

    # Distanzschwellenwert für die Zuordnung setzen
    distance_threshold = 200  # An Bildgröße anpassen
    good_matches = [m for m in matches if m[2] < distance_threshold]

    # Kombiniertes Bild zur Visualisierung erstellen
    combined_image = np.hstack((drone_image_rgb, satellite_image_rgb))
    offset = drone_image_rgb.shape[1]  # Offset für das Satellitenbild im kombinierten Bild

    # Linien zwischen zugeordneten Zentren zeichnen
    for match in good_matches:
        idx_drone = match[0]
        idx_satellite = match[1]

        cX_drone, cY_drone = centroids_drone_np[idx_drone]
        cX_satellite, cY_satellite = centroids_satellite_np[idx_satellite]
        cX_satellite += offset  # x-Koordinate für kombiniertes Bild anpassen

        # Linie zwischen den zugeordneten Zentren zeichnen
        cv2.line(
            combined_image,
            (cX_drone, cY_drone),
            (cX_satellite, cY_satellite),
            (0, 255, 0),
            2
        )

    # Zuordnung anzeigen
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image)
    plt.title('Zugeordnete Strukturen zwischen Drohnen- und Satellitenbild')
    plt.axis('off')
    plt.show()

    # Optional: Kombiniertes Bild speichern
    cv2.imwrite('matched_structures.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

# Optional: Bilder mit Konturen speichern
cv2.imwrite('satellite_with_contours.jpg', cv2.cvtColor(satellite_image_contours, cv2.COLOR_RGB2BGR))
cv2.imwrite('drone_with_contours.jpg', cv2.cvtColor(drone_image_contours, cv2.COLOR_RGB2BGR))
