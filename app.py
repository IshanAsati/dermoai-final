from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision.models import convnext_base
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
import io
import os
import json

# ---------------------------------------------------------------------------
#  SkinCancerModel definition (Self-contained, strict=False fallback weights)
# ---------------------------------------------------------------------------
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
                
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        features = self.backbone.classifier[0](x)
        features = self.backbone.classifier[1](features)
        
        class_logits = self.backbone.classifier[2](features)
        binary_logits = self.binary_head(features)
        embeddings = self.projection_head(features)
        return class_logits, binary_logits, embeddings

# ---------------------------------------------------------------------------
#  Application configuration
# ---------------------------------------------------------------------------
MEL_IDX = 5

app = Flask(__name__, static_folder='webapp')
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Booting DermoAI 12-Class Flask backend on {device}...", flush=True)

model = SkinCancerModel(num_classes=12)

# Load checkpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_weights", "ce_ls_best.pth")

if os.path.exists(model_path):
    print(f"Loading neural network weights from {model_path}...", flush=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("DermoAI 12-Class Engine Online!", flush=True)
else:
    print(f"CRITICAL WARNING: Checkpoint '{model_path}' not found.", flush=True)

# Melanoma logit bias
MEL_LOGIT_BIAS = 0.0
bias_config_path = os.path.join(BASE_DIR, "results_summary.json")
if os.path.exists(bias_config_path):
    try:
        with open(bias_config_path, 'r') as f:
            config = json.load(f)
        if "bias_sweep" in config:
            MEL_LOGIT_BIAS = config["bias_sweep"].get("selected_bias", 0.0)
        else:
            MEL_LOGIT_BIAS = config.get("mel_logit_bias", 0.0)
        print(f"Loaded melanoma logit bias: {MEL_LOGIT_BIAS}", flush=True)
    except Exception:
        MEL_LOGIT_BIAS = 0.0

# Image transforms (3x TTA)
tta_transforms = [
    v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.RandomHorizontalFlip(p=1.0),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomVerticalFlip(p=1.0),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

# Labels list
idx_to_label = {
    0: 'Basal Cell Carcinoma (BCC)', 
    1: 'Actinic Keratosis (ACK)', 
    2: 'Nevus / Mole (NEV)', 
    3: 'Seborrheic Keratosis (SEK)', 
    4: 'Squamous Cell Carcinoma (SCC)', 
    5: 'Melanoma (MEL)',
    6: 'Acne and Rosacea',
    7: 'Hair Loss (Alopecia)',
    8: 'Nail Fungus / Disease',
    9: 'Fungal Infections (Tinea / Ringworm)',
    10: 'Vascular Tumors',
    11: 'Healthy / Normal Skin'
}

# ---------------------------------------------------------------------------
#  Web API Routes
# ---------------------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded to the AI'}), 400
        
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Corrupt image file: {e}'}), 400
    
    with torch.no_grad():
        img1 = tta_transforms[0](img).unsqueeze(0).to(device)
        img2 = tta_transforms[1](img).unsqueeze(0).to(device)
        img3 = tta_transforms[2](img).unsqueeze(0).to(device)
        
        out1, _, _ = model(img1)
        out2, _, _ = model(img2)
        out3, _, _ = model(img3)

        if MEL_LOGIT_BIAS != 0.0:
            out1[0][MEL_IDX] += MEL_LOGIT_BIAS
            out2[0][MEL_IDX] += MEL_LOGIT_BIAS
            out3[0][MEL_IDX] += MEL_LOGIT_BIAS

        prob1 = torch.nn.functional.softmax(out1[0], dim=0)
        prob2 = torch.nn.functional.softmax(out2[0], dim=0)
        prob3 = torch.nn.functional.softmax(out3[0], dim=0)
        
        probabilities = (prob1 + prob2 + prob3) / 3.0
            
    all_probs = probabilities.cpu().numpy()
    ranked_indices = all_probs.argsort()[::-1]

    results = []
    for class_idx in ranked_indices:
        prob = float(all_probs[class_idx]) * 100.0
        class_name = idx_to_label.get(int(class_idx), f"Unknown Class {class_idx}")

        if class_idx == 5:
            severity = 'critical'
        elif class_idx in [0, 4]:
            severity = 'high'
        elif class_idx in [1, 3]:
            severity = 'medium'
        else:
            severity = 'low'

        results.append({
            'class': class_name,
            'class_code': ['BCC','ACK','NEV','SEK','SCC','MEL','Acne','Hair Loss','Nail Fungus','Fungal','Vascular','Healthy'][int(class_idx)],
            'probability': prob,
            'confidence': prob,
            'severity': severity,
            'risk': 'high' if severity in ('high','critical') else ('medium' if severity == 'medium' else 'low')
        })

    top = results[0]
    return jsonify({
        'predictions': results,
        'top_class': top['class'],
        'top_class_code': top['class_code'],
        'top_confidence': top['confidence'],
        'top_severity': top['severity'],
    })

if __name__ == '__main__':
    print("\nStarting DermoAI 12-Class Flask Server at http://localhost:5000 ...", flush=True)
    app.run(host='0.0.0.0', port=5000)
