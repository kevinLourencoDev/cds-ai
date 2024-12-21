import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

# Dataset pour l'entraînement
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        image = Image.open(image_path).convert("RGB")
        text = item['code']
        # Prétraitement de l'image et du texte
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return inputs['pixel_values'].squeeze(0), inputs['input_ids'].squeeze(0), text

# Préparer les données
data = [
    {
        "image": "images/input-no-label.png",
        "code": """
        <mat-form-field data-test="mat-form-field-with-no-label">
            <mat-label class="cdk-visually-hidden">Field no label</mat-label>
            <input matInput placeholder="Input with no" />
        </mat-form-field>
        """
    },
    {
        "image": "images/input-x-small.png",
        "code": """
        <mat-form-field cds-size="x-small">
            <mat-label>Input x-small</mat-label>
            <input matInput placeholder="Placeholder value" />
            <mat-hint align="start">Hint message</mat-hint>
     </mat-form-field>
        """
    },
    {
        "image": "images/button.png",
        "code": """
         <button mat-button color="primary" cds-type="very-strong">Very strong</button>
         xxxx xxxx xxxxxx xxxxxx xxxxx xxxxx xxxxxxx xxxxxxxx xxxxxxxxxx xxxxxxxxxxx   xxxxxxxx   xxxxxxxx  xxxxxxxx xxxxxxxx xxxxx xxxxxxx xxxxxxxx xxxxxxxxxx xxxxxxxxxxx xxxxxxx xxxxxxxx xxxxxxxx
        """
    },
]

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=1)  # Utilisation d'un batch_size de 1

# Charger le modèle
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.train()

# Sauvegarder les embeddings image et texte
image_embeddings = []
text_embeddings = []
codes = []

# Traiter chaque lot d'images et de textes
for batch in dataloader:
    pixel_values, input_ids, text = batch

    # Passer dans le modèle CLIP pour obtenir les embeddings
    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    image_embeds = outputs.image_embeds  # Embeddings des images
    text_embeds = outputs.text_embeds    # Embeddings des textes

    # S'assurer que les dimensions des embeddings sont correctes (512)
    print(f"Dimension de l'embedding des images : {image_embeds.shape}")
    print(f"Dimension de l'embedding des textes : {text_embeds.shape}")

    # Ajouter les embeddings aux listes
    # Chaque image_embeds est maintenant un lot de taille 1 avec un embedding de forme [512]
    for img_emb in image_embeds:
        image_embeddings.append(img_emb.detach().cpu().squeeze())  # Squeeze pour enlever les dimensions inutiles

    for txt_emb in text_embeds:
        text_embeddings.append(txt_emb.detach().cpu().squeeze())  # Squeeze pour enlever les dimensions inutiles

    codes.extend(text)  # Sauvegarder les codes de texte correspondants

# Sauvegarder les embeddings dans un fichier
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((image_embeddings, text_embeddings, codes), f)