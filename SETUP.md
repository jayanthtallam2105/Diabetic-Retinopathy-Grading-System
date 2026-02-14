# Setup Guide (Quick Reference)

> **Note**: For comprehensive documentation, please see [README.md](./README.md)

This is a quick reference guide for setting up the **FastAPI backend** and **React + Vite frontend**.

## Prerequisites

### Required Software

1. **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
2. **Node.js 16+ and npm** - [Download Node.js](https://nodejs.org/)

### Verify Installations

```powershell
python --version
node --version
npm --version
```

## Quick Setup

### 1. Backend Setup

Navigate to the backend directory and install dependencies:

```powershell
cd backend
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Note**: This project uses system-wide Python installation. No virtual environment is required.

### 2. Frontend Setup

Navigate to the frontend directory and install dependencies:

```powershell
cd frontend
npm install
```

### 3. Verify Model Files

Ensure all model weight files are present in `backend/models/`:
- `efficientnet_final.pt` (or `efficientnet.pt`)
- `resnet50.pt`
- `vit.pt`
- `hybrid_effvit.pt`
- `hybrid_resvit.pt`

> **Important**: Model files are not tracked in git (see `.gitignore`). You need to add them manually to the `backend/models/` directory.

## Running the Application

### Start Backend Server

Open a terminal/PowerShell window:

```powershell
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at:
- **API**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

### Start Frontend Server

Open a **new** terminal/PowerShell window:

```powershell
cd frontend
npm run dev
```

The frontend will be available at:
- **Web Application**: http://localhost:5173

**Important**: Start the backend server **before** the frontend to ensure proper connectivity.

## Accessing the Application

- **Frontend**: Open [http://localhost:5173](http://localhost:5173) in your browser
- **Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)
- **API Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI application
│   ├── inference.py           # Model inference logic
│   ├── image_validation.py    # Image quality validation
│   ├── utils.py               # Constants and helpers
│   ├── models/                # Trained model files (.pt) - not in git
│   │   ├── efficientnet_final.pt
│   │   ├── resnet50.pt
│   │   ├── vit.pt
│   │   ├── hybrid_effvit.pt
│   │   └── hybrid_resvit.pt
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx            # Main application component
│   │   ├── components/        # React components
│   │   │   ├── Upload.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── ResultCard.jsx
│   │   └── main.jsx           # React entry point
│   ├── package.json           # Node.js dependencies
│   └── vite.config.js         # Vite configuration
│
├── README.md                   # Comprehensive documentation
└── SETUP.md                    # This file (quick reference)
```

## Troubleshooting

### Backend Issues

**Models not loading**:
- Ensure all `.pt` model files are in `backend/models/` directory
- Check that model files are not corrupted
- Verify file names match expected names (see "Verify Model Files" section)

**Port 8000 already in use**:
- Change the port: `python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001`
- Update `API_BASE_URL` in `frontend/src/App.jsx` to match the new port

**Dependencies installation fails**:
- Update pip: `python -m pip install --upgrade pip`
- For Python 3.14+, some packages may need version flexibility (already handled in requirements.txt)

**Module not found errors**:
- Ensure you're in the `backend` directory when running uvicorn
- Verify all dependencies are installed: `python -m pip list`

### Frontend Issues

**Node.js not found**:
- Install Node.js from [nodejs.org](https://nodejs.org/)
- Restart terminal after installation

**npm install fails**:
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then run `npm install` again

**Port 5173 already in use**:
- Vite will automatically try the next available port
- Or specify a port: `npm run dev -- --port 5174`

**Cannot connect to backend**:
- Ensure backend is running on `http://127.0.0.1:8000`
- Check CORS settings in `backend/main.py`
- Verify `API_BASE_URL` in `frontend/src/App.jsx` matches backend URL

### Connection Issues

- **Backend not responding**: Check if backend server is running and accessible at http://127.0.0.1:8000/health
- **CORS errors**: Verify backend CORS settings allow requests from frontend URL (http://localhost:5173)
- **Network errors**: Ensure both servers are running and ports are not blocked by firewall

## Development Notes

- **Backend**: FastAPI with PyTorch for deep learning inference
- **Frontend**: React 19 with Vite for fast development
- **Models**: 5 models run in parallel (EfficientNet, ResNet50, ViT, Hybrid EffViT, Hybrid ResViT)
- **Image Validation**: Automatic quality checks before inference
- **No Virtual Environment**: Project uses system-wide Python installation

## Stopping the Servers

- Press `Ctrl+C` in each terminal window to stop the respective server
- Or simply close the terminal windows

## Next Steps

After setup, you can:
1. Open the frontend at http://localhost:5173
2. Upload a retinal fundus image
3. View predictions from all 5 models
4. Check API documentation at http://127.0.0.1:8000/docs

For detailed information, see the [README.md](./README.md) file.
