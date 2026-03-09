# Dermalyze Frontend

Web application for AI-assisted skin lesion classification intended as an educational tool.

> ⚠️ **DISCLAIMER**: Educational/research purposes only. Not for medical diagnosis. Consult healthcare professionals for medical advice.

## Overview

Dermalyze allows users to upload dermoscopic images and receive AI-powered classification results across 7 skin lesion categories:

| Abbreviation | Condition |
|---|---|
| **akiec** | Actinic keratoses and intraepithelial carcinoma |
| **bcc** | Basal cell carcinoma |
| **bkl** | Benign keratosis-like lesions |
| **df** | Dermatofibroma |
| **mel** | Melanoma |
| **nv** | Melanocytic nevi |
| **vasc** | Vascular lesions |

## Features

- **User Authentication** — Login, signup, and password recovery flows (Supabase Auth)
- **Image Upload** — Upload dermoscopic skin images for analysis
- **AI Classification** — Get probability scores across 7 lesion classes
- **Analysis History** — Review past classification results
- **Responsive UI** — Clean, mobile-friendly interface built with React + Tailwind CSS

## Architecture

This frontend communicates with the **Dermalyze Inference Service** (`../inference_service/`) for skin lesion classification:

- Frontend: React + Vite + TypeScript
- Backend API: FastAPI inference service (separate deployment)
- Auth & Storage: Supabase
- ML Models: EfficientNet-B0 / ConvNeXt-Tiny trained on HAM10000 dataset

See `../inference_service/README.md` for API documentation.

## Getting Started


### Prerequisites

Ensure the **inference service** is running (see `../inference_service/README.md`):

```bash
cd ../inference_service
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create a `.env.local` file in the project root:

```bash
cp .env.local.example .env.local
code .env.local
```

Set values in `.env.local` (API points to `../inference_service/` in local dev):

```env
# Local development (default)
VITE_API_URL=http://localhost:8000

# Production deployment
VITE_API_URL=https://your-backend-domain.com
```

**Local dev fallback:**
- If `VITE_API_URL` is not set, frontend calls `/api`.
- Vite proxies `/api/*` to `http://localhost:8000` by default.
- Override local backend target with:

```bash
BACKEND_URL=http://localhost:9000 npm run dev
```




### Run

```bash
npm run dev
```
Frontend runs at `http://localhost:5173` by default.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## Related Components

- **Inference API**: [`../inference_service/`](../inference_service/README.md) - FastAPI server for model predictions
- **Training Pipeline**: [`../skin_lesion_classifier/`](../skin_lesion_classifier/README.md) - ML model training and evaluation

