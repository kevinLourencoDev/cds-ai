import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# Étape 1 : Préparer vos données
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger l'image avec PIL
        image_path = item['image']
        image = Image.open(image_path).convert("RGB")  # S'assurer que l'image est en RGB

        # Traiter l'image et le texte
        text = item['code']
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        return inputs['pixel_values'].squeeze(0), inputs['input_ids'].squeeze(0)  # On renvoie directement les valeurs attendues par le modèle

data = [
    {
        "image": "images/input-no-label.png",
        "code": """
        <mat-form-field data-test="mat-form-field-with-no-label">
            <mat-label class="cdk-visually-hidden">Field no label</mat-label>
            <input matInput placeholder="Input with no Label" />
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
]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=4)

# Étape 2 : Charger le modèle CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Étape 3 : Entraîner le modèle
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):  # Nombre d'époques
    model.train()  # Mettre le modèle en mode entraînement
    for batch in dataloader:
        pixel_values, input_ids = batch

        # Passer les données dans le modèle
        outputs = model(pixel_values=pixel_values, input_ids=input_ids)

        # Obtenez les embeddings
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        # Normalisation des embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Calcul des similarités entre les embeddings image et texte (produit scalaire)
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T)  # Produit scalaire
        logits_per_text = logits_per_image.T  # Transposé pour la perte

        # Création des labels (les indices de correspondance pour chaque paire image-texte)
        labels = torch.arange(image_embeddings.size(0)).to(image_embeddings.device)

        # Calcul de la perte contrastive
        loss_image = loss_fn(logits_per_image, labels)
        loss_text = loss_fn(logits_per_text, labels)

        # Moyenne des deux pertes
        loss = (loss_image + loss_text) / 2

        # Affichage de la perte pour le débogage
        print(f"Epoch {epoch+1} - Loss: {loss.item()}")

        # Backpropagation et mise à jour des gradients
        optimizer.zero_grad()  # Remise à zéro des gradients
        loss.backward()        # Rétropropagation
        optimizer.step()       # Mise à jour des poids

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# Étape 4 : Sauvegarder le modèle
model.save_pretrained("models")