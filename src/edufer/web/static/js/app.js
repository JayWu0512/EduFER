const videoElement = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay");
const pipelineStatusElement = document.getElementById("pipeline-status");
const predictedLabelElement = document.getElementById("predicted-label");
const educationalSignalElement = document.getElementById("educational-signal");
const recommendationElement = document.getElementById("recommendation");
const runtimeNotesElement = document.getElementById("runtime-notes");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");

const captureCanvas = document.createElement("canvas");

let mediaStream = null;
let analysisTimer = null;
let analysisInFlight = false;

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const detectorStatus = data.detector_ready ? "ready" : "waiting for weights";
    pipelineStatusElement.textContent = `${data.detector_name} / ${detectorStatus}`;
    renderNotes([
      `Detector: ${data.detector_name}`,
      `Detector ready: ${String(data.detector_ready)}`,
      `Classifier: ${data.classifier_name}`,
      `Placeholder classifier: ${String(data.classifier_placeholder)}`,
    ]);
  } catch (error) {
    pipelineStatusElement.textContent = "Backend unavailable";
    renderNotes(["Could not reach /api/health. Start the FastAPI server first."]);
  }
}

async function startWebcam() {
  if (mediaStream) {
    return;
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  videoElement.srcObject = mediaStream;
  await videoElement.play();
  startButton.disabled = true;
  stopButton.disabled = false;
  pipelineStatusElement.textContent = "Webcam live, waiting for first analysis";
  startAnalysisLoop();
}

function stopAnalysis() {
  if (analysisTimer) {
    clearInterval(analysisTimer);
    analysisTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  analysisInFlight = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  videoElement.srcObject = null;
  clearOverlay();
  pipelineStatusElement.textContent = "Analysis paused";
}

function startAnalysisLoop() {
  if (analysisTimer) {
    clearInterval(analysisTimer);
  }

  analysisTimer = window.setInterval(() => {
    void analyzeCurrentFrame();
  }, 1200);

  void analyzeCurrentFrame();
}

async function analyzeCurrentFrame() {
  if (!mediaStream || analysisInFlight || videoElement.readyState < 2) {
    return;
  }

  analysisInFlight = true;
  try {
    captureCanvas.width = videoElement.videoWidth;
    captureCanvas.height = videoElement.videoHeight;
    const context = captureCanvas.getContext("2d");
    context.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
    const imageData = captureCanvas.toDataURL("image/jpeg", 0.85);

    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: imageData }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText);
    }

    const result = await response.json();
    updateDashboard(result);
  } catch (error) {
    pipelineStatusElement.textContent = "Analysis error";
    renderNotes([`Request failed: ${String(error)}`]);
  } finally {
    analysisInFlight = false;
  }
}

function updateDashboard(result) {
  pipelineStatusElement.textContent = result.face_detected
    ? "Face detected and analyzed"
    : "Webcam live, but no face is currently detected";

  predictedLabelElement.textContent = result.prediction
    ? `${result.prediction.label} (${(result.prediction.confidence * 100).toFixed(0)}%)`
    : "No face detected";

  educationalSignalElement.textContent = result.prediction
    ? result.prediction.educational_signal
    : "Move closer to the camera or improve lighting so the detector can find your face.";

  recommendationElement.textContent = result.prediction
    ? result.prediction.recommendation
    : "The system needs a visible face before it can surface a classroom intervention suggestion.";

  renderNotes(result.notes.length ? result.notes : ["No warnings from the backend."]);
  drawOverlay(result);
}

function drawOverlay(result) {
  overlayCanvas.width = videoElement.videoWidth || 1280;
  overlayCanvas.height = videoElement.videoHeight || 720;

  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!result.primary_face) {
    return;
  }

  const { x1, y1, x2, y2, confidence } = result.primary_face;
  const width = x2 - x1;
  const height = y2 - y1;

  context.strokeStyle = "#f6c95e";
  context.lineWidth = 4;
  context.strokeRect(x1, y1, width, height);

  const label = result.prediction
    ? `${result.prediction.label} | ${(confidence * 100).toFixed(0)}% face confidence`
    : `face ${(confidence * 100).toFixed(0)}%`;

  context.fillStyle = "rgba(44, 28, 16, 0.85)";
  context.fillRect(x1, Math.max(0, y1 - 34), Math.max(200, label.length * 7), 28);
  context.fillStyle = "#fffaf0";
  context.font = "16px Georgia";
  context.fillText(label, x1 + 10, Math.max(19, y1 - 14));
}

function clearOverlay() {
  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function renderNotes(notes) {
  runtimeNotesElement.innerHTML = "";
  notes.forEach((note) => {
    const listItem = document.createElement("li");
    listItem.textContent = note;
    runtimeNotesElement.appendChild(listItem);
  });
}

startButton.addEventListener("click", () => {
  void startWebcam();
});

stopButton.addEventListener("click", stopAnalysis);

void fetchHealth();
