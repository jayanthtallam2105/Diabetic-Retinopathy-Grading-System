# Setup Guide (Quick Reference)

> **Note**: For comprehensive documentation, please see [README.md](./README.md)

This is a quick reference guide for setting up the **FastAPI backend** and **React + Vite frontend**.

## Prerequisites

### Required Software

1. **Python 3.8+** ✅ (You have Python 3.14.2 installed)
2. **Node.js 16+ and npm** ⚠️ (Not detected - see installation below)

### Installing Node.js

If Node.js is not installed:

1. Download Node.js from [https://nodejs.org/](https://nodejs.org/)
2. Install the LTS (Long Term Support) version
3. Restart your terminal/PowerShell after installation
4. Verify installation by running:
   ```powershell
   node --version
   npm --version
   ```

## Quick Start

### Option 1: Start Both Servers Automatically (Recommended)

Run the following command in PowerShell from the project root:

```powershell
.\start_both.ps1
```

This will open two separate PowerShell windows:
- One for the backend server (port 8000)
- One for the frontend server (port 5173)

### Option 2: Start Servers Manually

#### Start Backend Server

```powershell
.\start_backend.ps1
```

Or manually:
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### Start Frontend Server

```powershell
.\start_frontend.ps1
```

Or manually:
```powershell
cd frontend
npm install
npm run dev
```

## Accessing the Application

- **Frontend**: Open [http://localhost:5173](http://localhost:5173) in your browser
- **Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)
- **API Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── main.py          # FastAPI application
│   ├── inference.py     # Model inference logic
│   ├── image_validation.py  # Image quality validation
│   ├── models/          # Trained model files (.pt)
│   └── requirements.txt # Python dependencies
│
├── frontend/            # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx      # Main application component
│   │   └── components/  # React components
│   ├── package.json     # Node.js dependencies
│   └── vite.config.js   # Vite configuration
│
├── start_backend.ps1    # Backend startup script
├── start_frontend.ps1   # Frontend startup script
└── start_both.ps1       # Start both servers
```

## Troubleshooting

### Backend Issues

1. **Virtual environment not activating**:
   - Run PowerShell as Administrator
   - Set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

2. **Models not loading**:
   - Ensure all `.pt` model files are in `backend/models/` directory
   - Check that model files are not corrupted

3. **Port 8000 already in use**:
   - Change the port in `start_backend.ps1` or use: `uvicorn main:app --reload --port 8001`
   - Update `API_BASE_URL` in `frontend/src/App.jsx` to match

### Frontend Issues

1. **Node.js not found**:
   - Install Node.js from [nodejs.org](https://nodejs.org/)
   - Restart terminal after installation

2. **npm install fails**:
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules` and `package-lock.json`, then run `npm install` again

3. **Port 5173 already in use**:
   - Vite will automatically try the next available port
   - Or specify a port: `npm run dev -- --port 5174`

### Connection Issues

- Ensure backend is running before starting frontend
- Check that backend CORS settings allow requests from frontend URL
- Verify `API_BASE_URL` in `frontend/src/App.jsx` matches backend URL

## Development Notes

- Backend uses **FastAPI** with **PyTorch** for deep learning inference
- Frontend uses **React 19** with **Vite** for fast development
- Backend runs 5 models in parallel: EfficientNet, ResNet50, ViT, and hybrid models
- Image quality validation is performed before inference

## Stopping the Servers

- Press `Ctrl+C` in each server window to stop
- Or simply close the PowerShell windows
