import sys
import torch
import torchvision.transforms as T
from PIL import Image
import json

from models.models import Swin


with open('files/style2idx.json', 'r') as f:
    style2idx = json.load(f)

with open('files/artist2idx.json', 'r') as f:
    artist2idx = json.load(f)

idx2style = {v:k for k, v in style2idx.items()}
idx2artist = {v:k for k, v in artist2idx.items()}


model = Swin(len(style2idx), len(artist2idx))
model.load_state_dict(torch.load('files/model_weights.pth', map_location='cpu'))
model.eval()

def predict(image_path):
    image = Image.open(image_path)
    transforms = T.Compose([
        T.Resize((384,384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        style_pred = torch.argmax(outputs[0], dim=1)
        artist_pred = torch.argmax(outputs[1], dim=1)

    return idx2style[style_pred.item()], idx2artist[artist_pred.item()]
