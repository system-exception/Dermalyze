# Dermalyze

AI-assisted skin lesion classification system for educational purposes.

> ⚠️ **DISCLAIMER**: Educational/research purposes only. Not for medical diagnosis. Consult healthcare professionals for medical advice.

## Overview

Dermalyze is a full-stack machine learning application that classifies dermoscopic images across 7 skin lesion types using deep learning models trained on the HAM10000 dataset.

**Classes**: akiec (Actinic keratoses), bcc (Basal cell carcinoma), bkl (Benign keratosis), df (Dermatofibroma), mel (Melanoma), nv (Melanocytic nevi), vasc (Vascular lesions)

## Architecture

```
Dermalyze/
├── frontend/              # React + Vite web application
├── inference_service/     # FastAPI inference API
└── skin_lesion_classifier/ # ML training pipeline
```

### Components

| Component | Purpose | Tech Stack |
|-----------|---------|------------|
| [**Frontend**](frontend/README.md) | Web UI for image upload and results | React, Vite, TypeScript, Tailwind CSS |
| [**Inference Service**](inference_service/README.md) | Production-ready classification API | FastAPI, PyTorch, uvicorn |
| [**Training Pipeline**](skin_lesion_classifier/README.md) | Model training and evaluation | PyTorch, EfficientNet-B0, ConvNeXt-Tiny |

## Quick Start

### 1. Train a Model

```bash
cd skin_lesion_classifier
bash scripts/install_pytorch.sh
pip install -r requirements.txt
python src/train.py --config config.yaml
```

### 2. Export to Inference Service

```bash
cp skin_lesion_classifier/outputs/run_xxx/checkpoint_best.pt inference_service/model/
```

### 3. Run Inference API

```bash
cd inference_service
pip install -r requirements.txt
uvicorn inference_service.app:app --host 0.0.0.0 --port 8000
```

### 4. Launch Frontend

```bash
cd frontend
npm install
npm run dev
```

Configure `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key
```

## Workflow

1. **Train**: Use `skin_lesion_classifier/` to train models on HAM10000 dataset
2. **Export**: Copy best checkpoint to `inference_service/model/`
3. **Deploy**: Run inference API independently from training code
4. **Integrate**: Frontend connects to inference API for predictions

## Features

- **EfficientNet-B0 & ConvNeXt-Tiny** architectures
- **Test-Time Augmentation (TTA)** for improved accuracy
- **K-Fold cross-validation** and ensemble training
- **Hyperparameter tuning** with Optuna
- **User authentication** via Supabase
- **Analysis history** tracking
- **Responsive web interface**

## Documentation

- [Frontend Setup & Usage](frontend/README.md)
- [Inference API Documentation](inference_service/README.md)
- [Training Pipeline Guide](skin_lesion_classifier/README.md)

## License

Educational use only. See individual component READMEs for details.
