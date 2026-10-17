# DermoAI V2 Local Model Tester

A local-only web interface for the selected DermoAI V2 checkpoint. It accepts one image at a time and displays the complete ranked output for all 12 canonical classes.

## Start

Create an environment and install the requirements:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\model_tester\requirements.txt"
```

Then start the app from PowerShell:

```powershell
& "C:\Users\Ishan\Documents\Codex\2026-10-15\referenced-chatgpt-conversation-this-is-an\outputs\dermoai\model_tester\start.ps1"
```

Then open `http://127.0.0.1:7861/`.

To stop a background copy of the tester:

```powershell
& "C:\Users\Ishan\Documents\Codex\2026-10-15\referenced-chatgpt-conversation-this-is-an\outputs\dermoai\model_tester\stop.ps1"
```

The server binds only to the local computer. Uploaded images are decoded in memory and are not written to disk.

## Model

- Architecture: ConvNeXt V2 Base
- Checkpoint: `model_weights/dermoai_v2_convnextv2_base.pth`
- Input: one JPG, PNG, WebP, or BMP image up to 16 MB
- Output: all 12 class scores, exact class indices, and the top model match

The displayed scores are uncalibrated softmax outputs. They are not medical probabilities, and the app is not a diagnostic tool.
