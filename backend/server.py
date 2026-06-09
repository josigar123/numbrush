from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from model import NumBrush
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)

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
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        predicted = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    
    return JSONResponse(content=PredictionResponse(digit=predicted, confidence=round(confidence, 4)))