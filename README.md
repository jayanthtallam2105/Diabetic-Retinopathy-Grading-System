# Diabetic Retinopathy Grading System

A comprehensive deep learning-based web application for automated diabetic retinopathy (DR) severity grading from retinal fundus images. This research prototype compares multiple state-of-the-art deep learning architectures to provide accurate and reliable DR classification.

---

## 📋 Project Overview

### Description

This project implements an end-to-end system that automatically grades diabetic retinopathy severity from retinal fundus images using ensemble deep learning models. The system provides a user-friendly web interface where medical professionals or researchers can upload retinal images and receive detailed predictions from multiple models simultaneously.

### Problem Statement

Diabetic retinopathy is a leading cause of blindness worldwide, affecting millions of diabetic patients. Early detection and accurate grading are crucial for preventing vision loss. Manual screening by ophthalmologists is time-consuming, expensive, and requires specialized expertise. Automated screening systems can help scale early detection efforts, especially in resource-limited settings.

### Objective

The primary objective is to develop a robust, multi-model deep learning system that:
- Accurately classifies diabetic retinopathy into 5 severity levels (No DR, Mild, Moderate, Severe, Proliferative)
- Provides comparative analysis across multiple model architectures
- Validates image quality before processing
- Offers an intuitive interface for medical image analysis

### Real-World Relevance

- **Healthcare Accessibility**: Enables faster screening in underserved areas
- **Clinical Decision Support**: Assists ophthalmologists with preliminary assessments
- **Research Tool**: Facilitates comparative studies of different deep learning architectures
- **Educational Purpose**: Demonstrates practical application of medical AI

---

## ✨ Features

- **Multi-Model Ensemble**: Runs 5 different deep learning models in parallel for comprehensive analysis
  - EfficientNet-B0
  - ResNet-50
  - Vision Transformer (ViT)
  - Hybrid EfficientNet-ViT
  - Hybrid ResNet-ViT

- **Image Quality Validation**: Automatic pre-processing checks including:
  - Blur detection
  - Brightness and contrast analysis
  - Retinal structure detection
  - Resolution validation

- **Comprehensive Results Dashboard**: 
  - Side-by-side comparison of all model predictions
  - Confidence scores and probability distributions
  - Visual representation of prediction results
  - Image quality metrics

- **Real-Time Processing**: Asynchronous inference for fast response times

- **Modern Web Interface**: 
  - Responsive design with Tailwind CSS
  - Intuitive image upload and preview
  - Real-time loading states and error handling

- **RESTful API**: Well-documented FastAPI backend with automatic Swagger documentation

---

## 🛠️ Tech Stack

### Backend

- **Python 3.8+**: Core programming language
  - *Why*: Industry standard for machine learning and scientific computing

- **FastAPI**: Modern, high-performance web framework
  - *Why*: Built-in async support, automatic API documentation, type hints, and excellent performance

- **PyTorch**: Deep learning framework
  - *Why*: Flexible, research-friendly, excellent model deployment capabilities

- **timm (PyTorch Image Models)**: Pre-trained model library
  - *Why*: Provides access to state-of-the-art pre-trained architectures (EfficientNet, ResNet, ViT)

- **Uvicorn**: ASGI server
  - *Why*: High-performance server for FastAPI with async support

- **OpenCV**: Computer vision library
  - *Why*: Essential for image preprocessing, quality validation, and retinal detection

- **Pillow (PIL)**: Image processing
  - *Why*: Standard library for image manipulation and format conversion

- **NumPy**: Numerical computing
  - *Why*: Foundation for all numerical operations in image processing and model inference

### Frontend

- **React 19**: Modern JavaScript framework
  - *Why*: Component-based architecture, excellent ecosystem, and developer experience

- **Vite**: Next-generation build tool
  - *Why*: Lightning-fast development server, optimized production builds

- **Tailwind CSS**: Utility-first CSS framework
  - *Why*: Rapid UI development, consistent design system, responsive by default

- **Axios**: HTTP client
  - *Why*: Promise-based API, interceptors, automatic JSON transformation

### Tools & Platforms

- **Git**: Version control
- **npm**: Node.js package manager
- **pip**: Python package manager

---

## 🏗️ Project Architecture

### System Architecture

```
┌─────────────────┐
│   React Frontend │  (Port 5173)
│   (Vite + React) │
└────────┬─────────┘
         │ HTTP/REST
         │ (Axios)
         ▼
┌─────────────────┐
│  FastAPI Backend │  (Port 8000)
│   (Python)       │
└────────┬─────────┘
         │
         ├─── Image Validation Module
         │    (Quality checks)
         │
         ├─── Preprocessing Module
         │    (Image normalization)
         │
         └─── Inference Engine
              │
              ├─── EfficientNet Model
              ├─── ResNet-50 Model
              ├─── ViT Model
              ├─── Hybrid EffViT Model
              └─── Hybrid ResViT Model
```

### Data Flow

1. **Image Upload**: User uploads retinal fundus image through React frontend
2. **Quality Validation**: Backend validates image quality (blur, brightness, retina detection)
3. **Preprocessing**: Image is resized to 224x224 and normalized
4. **Parallel Inference**: All 5 models process the image simultaneously
5. **Result Aggregation**: Predictions, confidence scores, and probabilities are collected
6. **Response**: JSON response sent to frontend with all model results
7. **Visualization**: Frontend displays results in an interactive dashboard

### Model Pipeline

```
Input Image (RGB)
    ↓
Quality Validation
    ↓
Preprocessing (224x224, normalization)
    ↓
┌─────────────────────────────────────┐
│  Parallel Model Inference           │
│  ┌──────────┐  ┌──────────┐        │
│  │EfficientNet│  │ ResNet50 │  ...  │
│  └──────────┘  └──────────┘        │
└─────────────────────────────────────┘
    ↓
Post-processing (softmax, argmax)
    ↓
Result Aggregation
    ↓
JSON Response
```

### Folder Structure

```
.
├── backend/                    # FastAPI backend application
│   ├── main.py                # FastAPI app, routes, and middleware
│   ├── inference.py           # Model loading and inference logic
│   ├── image_validation.py    # Image quality validation
│   ├── utils.py               # Constants and helper functions
│   ├── models/                # Trained model weights (.pt files)
│   │   ├── efficientnet_final.pt
│   │   ├── resnet50.pt
│   │   ├── vit.pt
│   │   ├── hybrid_effvit.pt
│   │   └── hybrid_resvit.pt
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── App.jsx            # Main application component
│   │   ├── components/
│   │   │   ├── Upload.jsx     # Image upload component
│   │   │   ├── Dashboard.jsx  # Results visualization
│   │   │   └── ResultCard.jsx # Individual model result card
│   │   ├── index.css          # Global styles
│   │   └── main.jsx           # React entry point
│   ├── package.json           # Node.js dependencies
│   ├── vite.config.js         # Vite configuration
│   └── tailwind.config.js     # Tailwind CSS configuration
│
└── README.md                   # This file
```

---

## 🚀 Installation & Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Node.js 16+ and npm** ([Download](https://nodejs.org/))
- **Git** (optional, for cloning repositories)

Verify installations:

```powershell
python --version
node --version
npm --version
```

### Step-by-Step Setup

#### 1. Clone or Navigate to Project Directory

```powershell
cd "path\to\Major Project IV-II Sem\Major Project IV-II Sem"
```

#### 2. Backend Setup

Navigate to the backend directory and install dependencies:

```powershell
cd backend
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Note**: The project uses system-wide Python installation. If you prefer using a virtual environment, you can create one:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

#### 3. Frontend Setup

Navigate to the frontend directory and install dependencies:

```powershell
cd ..\frontend
npm install
```

#### 4. Verify Model Files

Ensure all model weight files are present in `backend/models/`:
- `efficientnet_final.pt` (or `efficientnet.pt`)
- `resnet50.pt`
- `vit.pt`
- `hybrid_effvit.pt`
- `hybrid_resvit.pt`

### Running the Application

#### Start Backend Server

```powershell
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at:
- **API**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

#### Start Frontend Server

Open a new terminal window:

```powershell
cd frontend
npm run dev
```

The frontend will be available at:
- **Web Application**: http://localhost:5173

**Important**: Start the backend server before the frontend to ensure proper connectivity.

---

## 📦 Dependencies

### Backend Dependencies (`backend/requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.115.0 | Web framework for building REST API |
| `uvicorn[standard]` | 0.30.6 | ASGI server with performance optimizations |
| `torch` | ≥2.3.1 | Deep learning framework |
| `timm` | 1.0.9 | Pre-trained model library |
| `pillow` | ≥10.4.0 | Image processing and manipulation |
| `numpy` | ≥1.26.4 | Numerical computing foundation |
| `python-multipart` | 0.0.9 | File upload handling for FastAPI |
| `opencv-python` | 4.10.0.84 | Computer vision operations |

**Key Libraries Explained**:
- **FastAPI**: Provides automatic OpenAPI documentation, data validation, and async request handling
- **PyTorch**: Enables loading and running pre-trained deep learning models
- **timm**: Simplifies access to pre-trained architectures without manual implementation
- **OpenCV**: Performs image quality checks, blur detection, and retinal structure analysis

### Frontend Dependencies (`frontend/package.json`)

**Production Dependencies**:
- `react` (^19.2.0): UI library
- `react-dom` (^19.2.0): React DOM renderer
- `axios` (^1.13.4): HTTP client for API communication

**Development Dependencies**:
- `vite` (^7.2.4): Build tool and dev server
- `@vitejs/plugin-react-swc` (^4.2.2): Fast React refresh
- `tailwindcss` (^3.4.17): Utility-first CSS framework
- `autoprefixer` (^10.4.24): CSS vendor prefixing
- `postcss` (^8.5.6): CSS transformation tool
- `eslint` (^9.39.1): Code linting

---

## 💻 Usage

### Using the Web Interface

1. **Start both servers** (backend and frontend) as described in the Installation section
2. **Open your browser** and navigate to http://localhost:5173
3. **Upload a retinal fundus image** by clicking the upload area or dragging and dropping
4. **Click "Predict with All Models"** to run inference
5. **View results** in the dashboard showing:
   - Image quality assessment
   - Predictions from all 5 models
   - Confidence scores
   - Probability distributions for each severity level

### Using the API Directly

#### Health Check

```powershell
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": ["efficientnet", "resnet50", "vit", "hybrid_effvit", "hybrid_resvit"]
}
```

#### Image Prediction

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "image=@path/to/retinal_image.jpg"
```

#### Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
with open("retinal_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    results = response.json()
    print(results)
```

#### Response Format

```json
{
  "quality": {
    "accepted": true,
    "blur_score": 45.2,
    "brightness_score": 65.8,
    "retina_detected": true,
    "image_quality_score": 75.5,
    "message": "Image quality acceptable",
    "resolution": {"width": 1024, "height": 1024}
  },
  "predictions": {
    "efficientnet": {
      "grade": 1,
      "confidence": 0.89,
      "probs": [0.05, 0.89, 0.04, 0.01, 0.01]
    },
    "resnet50": {
      "grade": 1,
      "confidence": 0.92,
      "probs": [0.03, 0.92, 0.03, 0.01, 0.01]
    },
    ...
  }
}
```

### DR Severity Grades

The system classifies images into 5 severity levels:

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy detected |
| 1 | Mild | Mild non-proliferative diabetic retinopathy |
| 2 | Moderate | Moderate non-proliferative diabetic retinopathy |
| 3 | Severe | Severe non-proliferative diabetic retinopathy |
| 4 | Proliferative | Proliferative diabetic retinopathy |

---

## 🔧 Configuration

### Backend Configuration

The backend runs on `127.0.0.1:8000` by default. To change the port:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Update the frontend API URL in `frontend/src/App.jsx`:

```javascript
const API_BASE_URL = 'http://127.0.0.1:8001';
```

### CORS Settings

CORS is configured in `backend/main.py`. For production, restrict allowed origins:

```python
allow_origins=["http://localhost:5173", "https://yourdomain.com"]
```

---

## ⚠️ Important Notes

- **Research Prototype**: This is a research prototype and **NOT intended for clinical use**
- **Model Weights**: Ensure all model weight files are present in `backend/models/` before running
- **Image Requirements**: Images should be retinal fundus photographs in common formats (JPG, PNG)
- **Performance**: First inference may be slower as models are loaded into memory
- **GPU Support**: The system automatically detects and uses CUDA/MPS if available, otherwise uses CPU

---

## 🐛 Troubleshooting

### Backend Issues

**Models not loading**:
- Verify all `.pt` files exist in `backend/models/`
- Check file permissions and file integrity
- Review console output for specific error messages

**Port already in use**:
- Change the port: `--port 8001`
- Or stop the process using port 8000

**Dependencies installation fails**:
- Update pip: `python -m pip install --upgrade pip`
- For Python 3.14+, some packages may need version flexibility (already handled in requirements.txt)

### Frontend Issues

**Cannot connect to backend**:
- Ensure backend is running on `http://127.0.0.1:8000`
- Check CORS settings in `backend/main.py`
- Verify `API_BASE_URL` in `frontend/src/App.jsx`

**npm install fails**:
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then reinstall

**Port 5173 already in use**:
- Vite will automatically try the next available port
- Or specify: `npm run dev -- --port 5174`

---

## 📄 License

This project is a research prototype. Please ensure appropriate licensing for model weights and datasets used.

---

## 👥 Contributing

This is an academic/research project. For contributions or questions, please contact the project maintainers.

---

## 📚 References

- Diabetic Retinopathy: [American Academy of Ophthalmology](https://www.aao.org/)
- EfficientNet: [Paper](https://arxiv.org/abs/1905.11946)
- Vision Transformer: [Paper](https://arxiv.org/abs/2010.11929)
- FastAPI: [Documentation](https://fastapi.tiangolo.com/)

---

