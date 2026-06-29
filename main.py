import os
import io
import json
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

app.mount("/webapp", StaticFiles(directory="webapp"), name="webapp")

# 12 Classes mapping
CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL", "Acne", "Alopecia", "Nail", "Fungal", "Vascular", "Healthy"]
CLASS_FULL_NAMES = {
    "BCC": "Basal Cell Carcinoma",
    "ACK": "Actinic Keratosis",
    "NEV": "Nevus (Mole)",
    "SEK": "Seborrheic Keratosis",
    "SCC": "Squamous Cell Carcinoma",
    "MEL": "Melanoma",
    "Acne": "Acne and Rosacea",
    "Alopecia": "Alopecia and Hair Loss",
    "Nail": "Nail Fungus and Disease",
    "Fungal": "Fungal Infections",
    "Vascular": "Vascular Tumors",
    "Healthy": "Healthy Skin"
}
SEVERITY_MAP = {
    "MEL": "High · Malignant",
    "BCC": "High · Malignant",
    "SCC": "High · Malignant",
    "ACK": "Medium · Pre-cancerous",
    "NEV": "Low · Benign",
    "SEK": "Low · Benign",
    "Acne": "Low · Benign",
    "Alopecia": "Low · Benign",
    "Nail": "Low · Benign",
    "Fungal": "Low · Benign",
    "Vascular": "Low · Benign",
    "Healthy": "Low · Benign"
}
FALLBACK_CKPT = r"C:\Users\Ishan\Desktop\hybrid_model_12_class_29.pth"
MEL_IDX = 5  # class index for melanoma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = convnext_base(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        # New heads for Multi-Task and SupCon (Strat 1 & 3)
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.binary_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )

    def load_state_dict(self, state_dict, strict=True):
        has_seq = any(k.startswith("backbone.classifier.2.1.") for k in state_dict.keys())
        current_is_seq = isinstance(self.backbone.classifier[2], nn.Sequential)
        
        if has_seq != current_is_seq:
            in_features = self.backbone.classifier[2][1].in_features if current_is_seq else self.backbone.classifier[2].in_features
            num_classes = self.backbone.classifier[2][4].out_features if current_is_seq else self.backbone.classifier[2].out_features
            
            if has_seq:
                self.backbone.classifier[2] = nn.Sequential(
                    nn.Dropout(p=0.4),
                    nn.Linear(in_features, 512),
                    nn.GELU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(512, num_classes)
                )
            else:
                self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
                
        # Use strict=False to bypass missing binary_head/projection_head in older checkpoints
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # Extract features up to LayerNorm + Flatten manually
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        features = self.backbone.classifier[0](x)
        features = self.backbone.classifier[1](features)
        
        class_logits = self.backbone.classifier[2](features)
        binary_logits = self.binary_head(features)
        embeddings = self.projection_head(features)
        return class_logits, binary_logits, embeddings

# ---------------------------------------------------------------------------
#  Load best checkpoint — try trained result first, fall back to original
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_OUTPUT = os.path.join(BASE_DIR, "model_weights")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "results_summary.json")
MEL_BIAS = 0.0

print("Loading model...", flush=True)
model = SkinCancerModel(num_classes=12)

CE_LS_PATH = os.path.join(TRAIN_OUTPUT, "ce_ls_best.pth")

if os.path.exists(CE_LS_PATH):
    print(f"  Found ce_ls_best.pth! Loading directly...", flush=True)
    state = torch.load(CE_LS_PATH, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    
    # Read bias if results_summary exists, else default to 0.0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH) as f:
                summary = json.load(f)
            if "bias_sweep" in summary:
                MEL_BIAS = summary["bias_sweep"]["selected_bias"]
            else:
                MEL_BIAS = 0.0
        except Exception:
            MEL_BIAS = 0.0
    else:
        MEL_BIAS = 0.0
    print(f"  Model loaded. MEL bias: {MEL_BIAS:.1f}", flush=True)
elif os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        summary = json.load(f)
    # Determine best checkpoint and bias
    if "bias_sweep" in summary:
        MEL_BIAS = summary["bias_sweep"]["selected_bias"]
        exp_tag = None
        # Find which experiment produced the bias checkpoint
        for tag, exp in summary.get("experiments", {}).items():
            ckpt = os.path.join(TRAIN_OUTPUT, f"{tag}_best.pth")
            if os.path.exists(ckpt):
                exp_tag = tag
                break
    else:
        MEL_BIAS = 0.0
        exp_tag = None

    if exp_tag:
        ckpt_file = os.path.join(TRAIN_OUTPUT, f"{exp_tag}_best.pth")
        print(f"  Using trained checkpoint: {ckpt_file}", flush=True)
        print(f"  MEL bias: {MEL_BIAS:.1f}", flush=True)
        state = torch.load(ckpt_file, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd)
    else:
        print("  No trained experiment. Using fallback.", flush=True)
        model.load_state_dict(torch.load(FALLBACK_CKPT, map_location=device, weights_only=True))
else:
    print("  No results_summary.json, using fallback.", flush=True)
    model.load_state_dict(torch.load(FALLBACK_CKPT, map_location=device, weights_only=True))

model.to(device)
model.eval()
print("Model loaded.", flush=True)

transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_index():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        # Apply MEL bias at inference time
        if MEL_BIAS > 0:
            outputs[0, MEL_IDX] += MEL_BIAS
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    top_probs, top_idxs = torch.topk(probs, 4)
    
    predictions = []
    for i in range(4):
        code = CLASSES[top_idxs[i].item()]
        predictions.append({
            "code": code,
            "name": CLASS_FULL_NAMES[code],
            "prob": round(top_probs[i].item() * 100, 1),
            "severity": SEVERITY_MAP[code]
        })

    return JSONResponse(content={
        "top_prediction": predictions[0],
        "alternatives": predictions[1:]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
