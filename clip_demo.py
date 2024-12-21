import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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

        return inputs['pixel_values'], inputs['input_ids']  # On renvoie directement les valeurs attendues par le modèle

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
]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=4)

# Étape 2 : Charger le modèle CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Étape 3 : Entraîner le modèle
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):  # Nombre d'époques
    for batch in dataloader:
        pixel_values, input_ids = batch

        # Passer les données dans le modèle
        outputs = model(pixel_values=pixel_values, input_ids=input_ids)

        # Obtenez les embeddings
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

        # Calcul de la perte
        loss = loss_fn(image_embeddings, text_embeddings)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# Étape 4 : Sauvegarder le modèle
model.save_pretrained("models")