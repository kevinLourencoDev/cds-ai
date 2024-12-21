import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pickle
import torch.nn.functional as F

# Charger le modèle entraîné
model = CLIPModel.from_pretrained("models")  # Le modèle entraîné
model.eval()  # Passer en mode évaluation

# Charger les embeddings sauvegardés
with open('embeddings.pkl', 'rb') as f:
    image_embeddings, text_embeddings, codes = pickle.load(f)

# Initialiser le processeur
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Charger une image
image_path = "image-to-test.png"  # Remplace par l'image que tu veux tester
image = Image.open(image_path).convert("RGB")

# Prétraiter l'image
image_inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_outputs = model.get_image_features(**image_inputs)

# Afficher la taille des embeddings d'image et de texte pour déboguer
print(f"Dimension de l'embedding de l'image : {image_outputs.shape}")
text_embeddings_tensor = torch.stack(text_embeddings)  # Assure-toi que text_embeddings est bien une liste de tensors
print(f"Dimension des embeddings des textes : {text_embeddings_tensor.shape}")

# Si l'embedding des textes contient plusieurs éléments, on les réduit à un seul (en supposant que le bon texte est le premier)
if text_embeddings_tensor.ndimension() > 2:
    text_embeddings_tensor = text_embeddings_tensor[0]  # On sélectionne le premier texte (s'il y en a plusieurs)

# Calculer la similarité entre l'image et le texte
similarities = torch.matmul(image_outputs, text_embeddings_tensor.T)  # Produit scalaire

# Trouver le texte le plus similaire
most_similar_idx = torch.argmax(similarities).item()
most_similar_code = codes[most_similar_idx]

# Afficher le code le plus similaire
print(f"Le code le plus similaire à l'image est :\n{most_similar_code}")