# EduFER

EduFER is a first-pass project scaffold for the IDS 705 proposal **"Facial Expression Recognition for Student Engagement."**

This implementation focuses on the backend/system portion described in the proposal:

- a **one-page frontend** with webcam access
- **OpenCV + YOLO** face detection on frames captured from the browser
- a **replaceable facial-expression module** that is currently stubbed to always return `bored`
- a folder structure organized for future data, training, deployment, and experimentation work

The current demo is intentionally scoped to the team's Week 1 milestone: a simple prototype interface plus an initial face-detection demo.

## Proposal Alignment

The app content and structure are aligned to the proposal you shared:

- **Stakeholders:** EdTech product managers, instructors/TAs, school administrators, curriculum designers, and students
- **Emotion states in scope:** boredom, engagement, confusion, frustration
- **Goal:** turn facial-expression signals into interpretable classroom engagement feedback
- **Backend focus:** YOLO-based face detection with OpenCV for real-time webcam processing
- **Future handoff:** the DL classifier is isolated so your trained model can replace the placeholder without redesigning the rest of the system

## Why This YOLO Model

For the face-detection stage, this project uses **YOLOv8n-Face** in **ONNX** format.

Why this choice:

- it is trained specifically for **face detection**, not general object detection
- the `n` variant is the lightest model in the family, which makes it the best fit for a **real-time webcam demo**
- the ONNX weight can be loaded directly with **OpenCV DNN**, so the runtime stays simple and does not need PyTorch for inference

Default model location:

`data/models/face_detection/yolov8n-face-lindevs.onnx`

## Features

- Browser webcam access on a single landing page
- Live frame submission from frontend to backend
- YOLO-based face detection with bounding-box return
- Placeholder classifier module that always returns `bored`
- Proposal-aligned UI copy for stakeholder framing, engagement states, and ethics notes
- Clear modular separation for detector, classifier, pipeline, API, frontend, config, and data

## Project Structure

```text
EduFER/
├── .github/
│   └── workflows/
├── config/
│   └── app_config.json
├── data/
│   ├── models/
│   │   ├── classification/
│   │   └── face_detection/
│   ├── processed/
│   ├── raw/
│   └── README.md
├── models/
│   ├── resnet18_placeholder.pt
│   ├── vit_b16_placeholder.pt
│   ├── vgg16_placeholder.pt
│   └── README.md
├── notebooks/
│   ├── CompareModels.ipynb
│   └── ResnetFineTune.ipynb
├── requirements-notebooks.txt
├── scripts/
│   └── download_face_model.py
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── edufer/
│       ├── api/
│       ├── classification/
│       ├── core/
│       ├── detection/
│       ├── pipeline/
│       ├── research/
│       ├── utils/
│       └── web/
│           └── static/
├── tests/
├── LICENSE
├── README.md
└── requirements.txt
```

## Architecture Overview

### 1. Frontend

The frontend is a single static page that:

- opens the webcam with `navigator.mediaDevices.getUserMedia`
- captures frames on an interval
- posts frames to `/api/analyze`
- draws the returned face box on a canvas overlay
- shows the predicted state, educational signal, recommendation, and backend notes

### 2. Face Detection Module

`src/edufer/detection/yolov8_face_detector.py`

- loads the ONNX model with `cv2.dnn.readNetFromONNX`
- preprocesses the frame into a square YOLO input
- runs inference and NMS
- returns one or more face detections as bounding boxes

### 3. Classification Module

`src/edufer/classification/placeholder_classifier.py`

- implements the same interface your real DL model will use
- currently returns:
  - label: `bored`
  - confidence: `0.88`
  - placeholder flag: `True`

### 4. Pipeline Layer

`src/edufer/pipeline/engagement_pipeline.py`

- coordinates the detector and classifier
- selects the primary face
- crops the face region
- calls the classifier
- packages response metadata for the frontend

### 5. API Layer

`src/edufer/api/routes.py`

- `GET /api/health`
- `POST /api/analyze`

## Setup

### 1. Use Python 3.11

This project is now standardized on **Python 3.11** for local development, Docker, and GitHub Actions.

If you use `pyenv`, the included `.python-version` file will point your shell to Python 3.11 automatically.

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the YOLO face detector

If the model file is not already present, run:

```bash
python3.11 scripts/download_face_model.py
```

This downloads the model into:

```text
data/models/face_detection/yolov8n-face-lindevs.onnx
```

### 5. Start the app

Option A:

```bash
PYTHONPATH=src python3.11 -m edufer
```

Option B:

```bash
PYTHONPATH=src python3.11 -m uvicorn edufer.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Docker

### Build and run with Docker Compose

```bash
docker compose up --build
```

The app will be available at:

```text
http://127.0.0.1:8000
```

Notes:

- the Docker image uses **Python 3.11**
- the image downloads the YOLO face detector during build, so the container is self-contained
- the container runs as a **non-root user**

### Build the image manually

```bash
docker build -t edufer:latest .
docker run --rm -p 8000:8000 \
  -e EDUFER_ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000 \
  -e EDUFER_TRUSTED_HOSTS=localhost,127.0.0.1 \
  edufer:latest
```

## How the Demo Works

1. Open the website and click **Start Webcam**.
2. The browser captures frames and sends them to the backend.
3. YOLO detects the face.
4. The backend crops the primary face.
5. The placeholder classifier returns `bored`.
6. The page displays:
   - the face box
   - the placeholder state
   - an educational signal
   - a suggested classroom action

## Replacing the Placeholder Classifier

When your DL model is ready, replace the current placeholder module instead of changing the full system.

Recommended workflow:

1. Add your model weights under `data/models/classification/`
2. Create a new classifier class in `src/edufer/classification/`
3. Make it implement the same interface as `EmotionClassifier`
4. Update the classifier construction in `src/edufer/app.py`
5. Keep the detector, pipeline, frontend, and API unchanged

## Notebook Comparison Workflow

For offline experimentation, the repo now also includes:

- `notebooks/CompareModels.ipynb`
- placeholder model artifacts in `models/`
- reusable notebook helpers under `src/edufer/research/`

The notebook is organized so you can:

1. point to `data/processed/0` (`not_engaged`) and `data/processed/1` (`engaged`)
2. compare ResNet, ViT, and VGG checkpoints from one config block at the top
3. visualize the preprocessing flow used by the ResNet fine-tuning notebook
4. inspect per-model confusion matrices, PR curves, and accuracy summaries

Notebook extras can be installed with:

```bash
pip install -r requirements-notebooks.txt
```

The key interface is:

```python
class EmotionClassifier(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def is_placeholder(self) -> bool:
        return False

    @abstractmethod
    def classify(self, face_image: np.ndarray) -> EmotionPrediction:
        ...
```

## OOP / Module Boundaries

This repo is structured so each responsibility stays isolated:

- `core/`: shared schemas and config loading
- `detection/`: face detector implementations
- `classification/`: expression classifier implementations
- `pipeline/`: orchestration logic
- `api/`: FastAPI request/response handling
- `utils/`: image decoding helpers
- `web/static/`: single-page frontend
- `data/`: datasets and model weights
- `scripts/`: operational utilities such as model download

## Security Defaults

The app now includes baseline production-minded security controls:

- **CORS allowlist** instead of `*`
- **Trusted host validation** to reject unexpected `Host` headers
- **request size limit** for inbound webcam payloads
- **security headers** including CSP, `X-Frame-Options`, `X-Content-Type-Options`, and `Permissions-Policy`
- optional **HTTPS redirect** via config or environment variable

Default local settings live in `config/app_config.json`. For Docker or deployment environments, you can override them with environment variables:

- `EDUFER_ALLOWED_ORIGINS`
- `EDUFER_TRUSTED_HOSTS`
- `EDUFER_ALLOWED_METHODS`
- `EDUFER_ALLOWED_HEADERS`
- `EDUFER_ALLOW_CREDENTIALS`
- `EDUFER_MAX_REQUEST_BYTES`
- `EDUFER_ENABLE_HTTPS_REDIRECT`
- `EDUFER_HOST`
- `EDUFER_PORT`

Example:

```bash
export EDUFER_ALLOWED_ORIGINS="https://your-frontend.example.com"
export EDUFER_TRUSTED_HOSTS="your-api.example.com"
export EDUFER_ENABLE_HTTPS_REDIRECT="true"
```

If you later split frontend and backend across different domains, update `EDUFER_ALLOWED_ORIGINS` rather than widening CORS to `*`.

## GitHub CI/CD

Two GitHub Actions workflows are included:

- `ci.yml`
  - runs on push / pull request
  - uses **Python 3.11**
  - installs dependencies
  - runs compile checks and unit tests
  - validates that the Docker image builds
- `docker-publish.yml`
  - runs on pushes to `main`, tags like `v1.0.0`, or manually
  - builds and publishes a Docker image to **GitHub Container Registry**
  - image target: `ghcr.io/<owner>/<repo>`

For image publishing, make sure GitHub Actions has package write permission enabled for the repository.

## Suggested Next Steps

To keep following the proposal, the natural next implementation steps are:

1. Connect the real facial-expression classifier for the four target states
2. Add temporal smoothing or clip-level aggregation for session analysis
3. Introduce Grad-CAM / interpretability outputs for the trained DL model
4. Add DAiSEE preprocessing and training scripts under a future `training/` module
5. Expand the frontend into a stakeholder-facing dashboard with aggregate session trends

## Running Tests

```bash
PYTHONPATH=src python3.11 -m unittest discover -s tests
```

## Important Notes

- This repo currently demonstrates **system integration**, not final model quality.
- The output is **not** a medical, psychological, or definitive judgment.
- The current classifier is a placeholder by design.
- Any real deployment should include privacy, consent, and bias review.

## References

- Proposal PDF: `Copy of IDS705 Project Proposal Template 2026.pdf`
- DAiSEE paper: https://arxiv.org/abs/1609.01885
- DAiSEE dataset page: https://people.iith.ac.in/vineethnb/resources/daisee/index.html
- YOLOv8-Face weights/repo: https://github.com/lindevs/yolov8-face
