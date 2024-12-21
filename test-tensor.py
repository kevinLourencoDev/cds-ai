import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pickle

# Charger le modèle CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # Passer en mode évaluation

# Charger les embeddings sauvegardés
with open('embeddings.pkl', 'rb') as f:
    image_embeddings, text_embeddings, codes = pickle.load(f)

# Initialiser le processeur CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Charger une image
image_path = "image-to-test.png"  # Remplace par l'image que tu veux tester
image = Image.open(image_path).convert("RGB")

# Prétraiter l'image
image_inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_output = model.get_image_features(**image_inputs)

# Vérification de la dimension de l'output
print(f"Dimension de l'output de l'image entière : {image_output.shape}")

# S'assurer que l'embedding est bien un vecteur de taille [512]
image_embedding = image_output.squeeze(0)
print(f"Dimension après squeeze pour l'image : {image_embedding.shape}")

# Définir un seuil de similarité
threshold = 0.7  # Ajustez ce seuil si nécessaire

# Trouver toutes les correspondances
matched_codes = []
for idx, stored_image_embeds in enumerate(image_embeddings):
    # Suppression de dimensions inutiles
    stored_image_embeds = stored_image_embeds.squeeze(0)

    # Calcul de la similarité de cosinus
    similarity = torch.cosine_similarity(image_embedding, stored_image_embeds, dim=-1).item()
    print(f"Similarité avec l'embedding {idx} : {similarity}")

    # Ajouter le code correspondant si la similarité dépasse le seuil
    if similarity >= threshold:
        matched_codes.append((similarity, codes[idx]))

# Afficher les correspondances trouvées
if matched_codes:
    print(f"\nCorrespondances trouvées (seuil {threshold}):")
    for similarity, code in matched_codes:
        print(f"\nSimilitude : {similarity}\nCode :\n{code}")
else:
    print(f"Aucune correspondance dépassant le seuil {threshold} n'a été trouvée.")