import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
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
dataloader = DataLoader(dataset, batch_size=4)

# Charger le modèle
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.train()

# Sauvegarder les embeddings image et texte
image_embeddings = []
text_embeddings = []
codes = []

for batch in dataloader:
    pixel_values, input_ids, text = batch

    # Passer dans le modèle
    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Ajouter aux listes
    image_embeddings.append(image_embeds.detach().cpu())
    text_embeddings.append(text_embeds.detach().cpu())
    codes.extend(text)  # Sauvegarder le texte correspondant

# Sauvegarder les embeddings dans un fichier
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((image_embeddings, text_embeddings, codes), f)