# DermoAI — Clinical Skin Lesion Diagnostic Triage Engine

Welcome to the official submission repository for DermoAI, an AI-powered diagnostic and clinical triage support system. 

DermoAI fine-tunes a **ConvNeXt-Base** deep neural network (88M parameters) to classify **12 distinct skin conditions** from clinical dermoscopy and consumer smartphone photography, bridging the gap between clean research datasets and real-world frontend clinical usage.

---

## 📂 Directory Structure

The submission folder is structured as a fully portable, self-contained project:

```text
DermoAI_Submission/
├── app.py                  # Standalone Flask server (Port 5000) - supports 3x TTA and premium UI
├── main.py                 # FastAPI backend server (Port 8000)
├── results_summary.json    # JSON summary of final training runs & evaluations
├── README.md               # This documentation file
├── webapp/
│   └── index.html          # Premium, glassmorphic visual interface (Tailwind & Inter font)
└── model_weights/
    └── ce_ls_best.pth      # Optimized, fine-tuned model checkpoint (~354.8 MB)
```

---

## 🚀 How to Run the Application

Follow these simple steps to run the server and UI locally:

### 1. Pre-requisites (Python Packages)
Ensure you have the required libraries installed in your Python environment:
```bash
pip install flask flask-cors fastapi uvicorn torch torchvision pillow
```

### 2. Launch the Server

#### Method A: Flask Server (Recommended — Runs Port 5000)
To run the server with the premium UI on port 5000:
```bash
python app.py
```
Open your browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

#### Method B: FastAPI Server (Runs Port 8000)
To run the alternate server on port 8000:
```bash
python main.py
```
Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

You can drag and drop any skin lesion image (or click to browse). The UI displays an instant image preview in the **Image Acquisition** box and returns a 12-class ranked differential and calibrated confidence intervals upon running analysis.

---

## 📊 Final Optimized Performance Metrics

Our final model ensembled a label-smoothed Cross-Entropy model (`ce_ls_best.pth`) and a Focal Loss model (`focal_g1_best.pth`) alongside 5-view Test-Time Augmentation (TTA).

*   **Overall Multi-Class Accuracy**: **82.79%** (an absolute +4.07% increase over baseline).
*   **Clinical Triage Sensitivity**: **87.30%** (calibrated with a conservative 15% malignancy threshold).
*   **Clinical Triage Specificity**: **92.27%** (skyrocketed from 76.00% previously to prevent false-alarm clinic referrals).
*   **Melanoma Recall**: **78.62%** (critical safety-first improvement).
*   **Smartphone-Captured Photo Accuracy**: **35.58%** (successfully adapted from a 27.59% baseline).

---

## 🛠️ The 11 Model Engineering Hardships We Overcame

Our team navigated multiple critical bottlenecks to shape DermoAI into a medical-grade decision support tool:

### 1. The 100x VRAM Shared Memory Slowdown
*   *Problem*: Training a ConvNeXt-Base model (88M params) on 384x384 images locally exceeded the 6GB VRAM of our laptop RTX 4050 GPU. Windows dynamically swapped VRAM overflow into standard system RAM, bottlenecking training speed by 100x (54s per batch).
*   *Solution*: Migrated the main training loop to a Google Colab T4 GPU (16GB VRAM), and configured local validation using batch sizes of 4 and 8-step gradient accumulation.

### 2. The Windows Multiprocessing Loader Bug
*   *Problem*: PyTorch dataloaders with `num_workers > 0` spawned child worker threads recursively on Windows, flooding the console with print statements and halting training.
*   *Solution*: Locked `num_workers = 0` on Windows and utilized memory pinning (`pin_memory=True`) to maintain high data loading throughput.

### 3. Kaggle Network Storage Deadlocks
*   *Problem*: Using Kaggle multi-GPU runtimes with high worker counts caused the Kagglehub network storage drive to deadlock, freezing execution.
*   *Solution*: Restrained worker counts to exactly 2 and adjusted pre-fetching queues.

### 4. Legacy Keras-to-PyTorch Translation Conflicts
*   *Problem*: The legacy codebase attempted to run a Keras/TensorFlow model ensemble alongside PyTorch weights inside the Flask server, leading to severe CUDA device lockups.
*   *Solution*: Completely discarded the legacy Keras modules and refactored the entire model architecture into a unified native PyTorch `SkinCancerModel` class.

### 5. Classifier Output Weight Dimension Mismatches
*   *Problem*: Transferring weights from a 26-class pre-trained checkpoint (`hybrid_model_26_class.pth`) to our new clinical 12-class head threw dimension mismatch errors.
*   *Solution*: Wrote a custom `load_state_dict` method in the model definition that dynamically intercepts the final linear classification layer, resizing shapes and loading weights with `strict=False`.

### 6. StreamReader ZipArchive Checkpoint Corruption
*   *Problem*: Model weights occasionally became corrupted during saving or network transfers, causing backend crashes.
*   *Solution*: Added an integrity checking script that scans the central directory metadata of the `.pth` file on startup and falls back to a verified baseline checkpoint if corrupted.

### 7. Windows Console Emoji Encoding Crashes
*   *Problem*: Status emojis (e.g. `✅`, `✗`) printed by logging modules threw `UnicodeEncodeError` exceptions on Windows consoles.
*   *Solution*: Reconfigured Python's stdout stream to use UTF-8 and stripped console emojis.

### 8. Resolution Scaling & Interpolation Distortion
*   *Problem*: Training at 224x224 was fast but missed microscopic cellular borders, while starting training at 384x384 destabilized convergence.
*   *Solution*: Implemented Progressive Resolution Scaling: trained at 224x224 for the first 50 epochs, and transitioned weights to 384x384 for final fine-tuning.

### 9. Dataset Class Imbalance & Minority Class Amnesia
*   *Problem*: Benign Nevi heavily outnumbered malignant Melanoma, causing the baseline model to collapse into predicting only the majority classes.
*   *Solution*: Deployed class-balanced dynamic resampling in the data loader, and trained using **Focal Loss** to force gradient updates to focus on hard, misclassified examples.

### 10. Out-of-Distribution Defocus (The 27% Smartphone Collapse)
*   *Problem*: The model achieved ~78% accuracy on clean clinical images but crashed to 27.59% on low-quality smartphone captures due to blur and lighting shifts.
*   *Solution*: Unfroze stages 6 & 7 of the ConvNeXt-Base backbone to learn camera-invariant features, applied **Supervised Contrastive Learning (SupConLoss)** clustering, and used defocus/shadow data augmentations.

### 11. The Logit Bias False-Alarm Crisis
*   *Problem*: Adding positive logit biases (+1.3 for Melanoma) boosted recall but spiked false-positives (dropping Melanoma precision below 20%), making the UI clinically useless.
*   *Solution*: Removed output logit biases and utilized model ensembling at zero bias, achieving a clean **92.27% specificity** and **87.30% sensitivity**.
