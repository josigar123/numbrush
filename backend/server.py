from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import io
from model import NumBrush
from pydantic import BaseModel, Field

class Candidate(BaseModel):
    digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    predictions: list[Candidate]

app = FastAPI()

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
model = NumBrush()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def crop_and_center(image: Image.Image) -> Image.Image:
    image = image.convert('L')
    arr = np.array(image)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)
    if not rows.any():
        return image
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = max((rmax - rmin), (cmax - cmin)) // 4
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0], rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1], cmax + pad)
    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cropped
    return Image.fromarray(square)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = crop_and_center(image)
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top2 = torch.topk(probs, 2)

    predictions = [
        Candidate(digit=top2.indices[i].item(), confidence=round(top2.values[i].item(), 4))
        for i in range(2)
    ]
    return PredictionResponse(predictions=predictions)