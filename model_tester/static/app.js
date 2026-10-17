const elements = {
  form: document.querySelector("#predictionForm"),
  input: document.querySelector("#imageInput"),
  dropZone: document.querySelector("#dropZone"),
  dropPrompt: document.querySelector("#dropPrompt"),
  previewWrap: document.querySelector("#previewWrap"),
  preview: document.querySelector("#imagePreview"),
  fileRow: document.querySelector("#fileRow"),
  fileName: document.querySelector("#fileName"),
  fileMeta: document.querySelector("#fileMeta"),
  clearButton: document.querySelector("#clearButton"),
  analyzeButton: document.querySelector("#analyzeButton"),
  emptyState: document.querySelector("#emptyState"),
  loadingState: document.querySelector("#loadingState"),
  errorState: document.querySelector("#errorState"),
  errorMessage: document.querySelector("#errorMessage"),
  resultContent: document.querySelector("#resultContent"),
  resultState: document.querySelector("#resultState"),
  modelStatus: document.querySelector("#modelStatus"),
  modelStatusText: document.querySelector("#modelStatusText"),
  deviceLabel: document.querySelector("#deviceLabel"),
  topIndex: document.querySelector("#topIndex"),
  topCode: document.querySelector("#topCode"),
  topName: document.querySelector("#topName"),
  topGroup: document.querySelector("#topGroup"),
  topConfidence: document.querySelector("#topConfidence"),
  predictionList: document.querySelector("#predictionList"),
  runMeta: document.querySelector("#runMeta"),
};

let selectedFile = null;
let previewUrl = null;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setVisible(active) {
  ["emptyState", "loadingState", "errorState", "resultContent"].forEach((key) => {
    elements[key].hidden = key !== active;
  });
}

function clearSelection() {
  selectedFile = null;
  elements.input.value = "";
  elements.dropPrompt.hidden = false;
  elements.previewWrap.hidden = true;
  elements.fileRow.hidden = true;
  elements.analyzeButton.disabled = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function selectFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Choose a JPG, PNG, WebP, or BMP image.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("That image is larger than the 16 MB limit.");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.dropPrompt.hidden = true;
  elements.previewWrap.hidden = false;
  elements.fileRow.hidden = false;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${formatBytes(file.size)} · ${file.type.replace("image/", "").toUpperCase()}`;
  elements.analyzeButton.disabled = false;
  elements.resultState.textContent = "Ready to analyze";
  if (!elements.resultContent.hidden || !elements.errorState.hidden) setVisible("emptyState");
}

function showError(message) {
  setVisible("errorState");
  elements.errorMessage.textContent = message;
  elements.resultState.textContent = "Needs attention";
}

function groupClass(group) {
  if (group === "cancer-associated") return "cancer";
  if (group === "precancer-associated") return "precancer";
  return "other";
}

function renderPredictionRow(prediction) {
  const item = document.createElement("li");
  item.className = `prediction-row ${prediction.group === "cancer-associated" ? "cancer-row" : ""}`;

  const rank = document.createElement("span");
  rank.className = "rank-index";
  rank.textContent = `${String(prediction.rank).padStart(2, "0")} · #${prediction.class_index}`;

  const label = document.createElement("div");
  label.className = "prediction-label";
  const code = document.createElement("strong");
  code.textContent = prediction.code;
  const name = document.createElement("span");
  name.textContent = prediction.name;
  label.append(code, name);

  const bar = document.createElement("div");
  bar.className = "probability-bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(prediction.percentage, 0.35)}%`;
  bar.append(fill);

  const value = document.createElement("span");
  value.className = "probability-value";
  value.textContent = `${prediction.percentage.toFixed(2)}%`;
  item.append(rank, label, bar, value);
  return item;
}

function renderResult(data) {
  const top = data.prediction;
  elements.topIndex.textContent = top.class_index;
  elements.topCode.textContent = top.code;
  elements.topName.textContent = top.name;
  elements.topGroup.textContent = top.group;
  elements.topGroup.className = `group-tag ${groupClass(top.group)}`;
  elements.topConfidence.textContent = `${top.percentage.toFixed(2)}%`;

  elements.predictionList.replaceChildren(...data.predictions.map(renderPredictionRow));
  const metadata = [
    `${data.model.architecture}`,
    `${data.model.device}`,
    `${data.model.inference_ms} ms model pass`,
    `${data.image.width} × ${data.image.height} ${data.image.format}`,
    `${data.model.version}`,
  ];
  elements.runMeta.replaceChildren(...metadata.map((text) => {
    const tag = document.createElement("span");
    tag.textContent = text;
    return tag;
  }));
  elements.deviceLabel.textContent = data.model.device;
  elements.resultState.textContent = "Analysis complete";
  setVisible("resultContent");
}

async function runPrediction(event) {
  event.preventDefault();
  if (!selectedFile) return;

  setVisible("loadingState");
  elements.resultState.textContent = "Analyzing";
  elements.analyzeButton.disabled = true;
  const body = new FormData();
  body.append("image", selectedFile, selectedFile.name);

  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    renderResult(data);
  } catch (error) {
    showError(error.message || "Prediction failed.");
  } finally {
    elements.analyzeButton.disabled = !selectedFile;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error("Model unavailable");
    elements.modelStatus.classList.add("ready");
    elements.modelStatusText.textContent = data.model.loaded
      ? `${data.model.architecture} · ${data.model.device}`
      : `${data.model.architecture} · ready`;
    if (data.model.device) elements.deviceLabel.textContent = data.model.device;
  } catch (_error) {
    elements.modelStatus.classList.add("error");
    elements.modelStatusText.textContent = "Local model unavailable";
  }
}

elements.input.addEventListener("change", () => selectFile(elements.input.files[0]));
elements.clearButton.addEventListener("click", clearSelection);
elements.form.addEventListener("submit", runPrediction);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
});

elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
checkHealth();
