# 🏗️ ARCHITECTURE.md - System Architecture Documentation

## Project: AI-Based Intelligent System for Skin Disease Detection

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Database Schema](#database-schema)
7. [API Endpoints](#api-endpoints)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)

---

## System Overview

The Skin Disease Detection and Healthcare Recommendation System is a **web-based AI application** that:
- Classifies skin diseases from uploaded images using deep learning
- Provides health recommendations based on detected conditions
- Recommends appropriate hospitals and doctors
- Collects user feedback for continuous improvement
- Manages user authentication and data privacy

**Key Metrics:**
- **Model Classes**: 22 different skin disease categories
- **Model Architecture**: MobileNetV2 (transfer learning)
- **Framework**: Streamlit for web UI
- **Model Accuracy**: Baseline ~85-90% (see training reports)
- **User Base**: Multi-user system with Google OAuth

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │   Web Browser    │  │   Google OAuth   │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│             STREAMLIT APPLICATION LAYER                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  app.py - Main Streamlit Application                │ │
│  │  • User authentication                              │ │
│  │  • Image upload interface                           │ │
│  │  • Prediction display                               │ │
│  │  • Hospital recommendations                         │ │
│  │  • Feedback form                                    │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│          BUSINESS LOGIC LAYER                             │
│  ┌──────────────────┐  ┌─────────────────────────────┐   │
│  │   predict.py     │  │  data_validation_           │   │
│  │ • Image prep     │  │  cleaning.py                │   │
│  │ • Model inference│  │ • Data cleaning             │   │
│  │ • Predictions    │  │ • Validation checks         │   │
│  └──────────────────┘  └─────────────────────────────┘   │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│        MACHINE LEARNING LAYER                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Models (Fine-tuned MobileNetV2)                     │ │
│  │  • skin_disease_model_baseline_best.keras            │ │
│  │  • skin_disease_model_baseline_final.keras           │ │
│  │  • Metadata files (.meta.json)                        │ │
│  │  • Class mappings (22 diseases)                       │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│          DATA & STORAGE LAYER                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  SQLite Database │  │  JSON Data Files              │  │
│  │  • users.db      │  │  • hospital_data.json         │  │
│  │  • User auth     │  │  • doctor_consultation_       │  │
│  │  • Feedback      │  │    data.json                  │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. **Frontend (Presentation Layer)**

**File**: `SkinDisease/app.py`

**Responsibilities:**
- User authentication (Google OAuth + local credentials)
- Image upload interface
- Disease classification results display
- Hospital/doctor recommendations
- Feedback collection form
- User profile management

**Key Features:**
- Multi-page navigation (Home, Prediction, Profile, Feedback)
- Real-time prediction results
- Confidence score visualization
- Hospital search and filtering
- User feedback history

### 2. **ML Inference Engine**

**File**: `SkinDisease/predict.py`

**Responsibilities:**
- Load pre-trained models
- Preprocess input images (224x224 normalization)
- Run inference on images
- Generate prediction confidence scores
- Map predictions to disease names
- Provide health recommendations

**Key Functions:**
- `load_model()` - Load Keras model
- `predict_skin_disease()` - Get predictions
- `get_health_recommendation()` - Health advice
- `load_class_names()` - Load disease categories

### 3. **Data Processing**

**File**: `SkinDisease/data_validation_cleaning.py`

**Responsibilities:**
- Image file validation
- Corrupt file detection
- Dataset statistics
- Data quality reporting

**Validation Checks:**
- File format validation (JPG, PNG, BMP)
- Image dimension checks (50x50 to 5000x5000)
- Duplicate detection
- Missing data handling

### 4. **Model Training Pipeline**

**Files**: 
- `SkinDisease/train_baseline.py` - Baseline training
- `SkinDisease/fine_tune_model.py` - Fine-tuning with GPU
- `SkinDisease/validate_model.py` - Model evaluation

**Training Pipeline:**
```
Raw Data (train/val/test split)
    ↓
Data Augmentation
    ↓
MobileNetV2 Base Model Loading
    ↓
Head Training (20 epochs)
    ↓
Fine-tuning (50 epochs)
    ↓
Evaluation & Metrics
    ↓
Model Saving & Metadata
```

### 5. **Data Storage**

**Database Schema:**
```sql
-- User Authentication
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    phone_number TEXT,
    age INTEGER,
    gender TEXT,
    section TEXT,
    oauth_provider TEXT,
    oauth_sub TEXT,
    created_at TIMESTAMP
);

-- User Feedback
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    prediction TEXT,
    actual_condition TEXT,
    confidence FLOAT,
    feedback TEXT,
    timestamp TIMESTAMP
);

-- Reset Tokens
CREATE TABLE reset_tokens (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    token_hash TEXT,
    expires_at TIMESTAMP
);
```

---

## Data Flow

### Flow 1: User Authentication
```
User
  ↓
Login Page (Google OAuth or Email/Password)
  ↓
Verify Credentials (users.db)
  ↓
Create Session
  ↓
Redirect to Dashboard
```

### Flow 2: Image Prediction
```
User uploads image
  ↓
Image validation (format, size)
  ↓
Image preprocessing (224x224 + normalization)
  ↓
Load pre-trained model
  ↓
Run inference
  ↓
Get top-3 predictions with confidence
  ↓
Look up health recommendations
  ↓
Find matching hospitals/doctors
  ↓
Display results to user
  ↓
Store in feedback log
```

### Flow 3: Feedback Collection
```
User submits feedback
  ↓
Validate feedback data
  ↓
Store in users.db
  ↓
Save to feedback.csv
  ↓
Acknowledge to user
```

---

## Technology Stack

### Backend & ML
| Component | Technology | Version |
|-----------|-----------|---------|
| Deep Learning | TensorFlow/Keras | ≥2.10.0 |
| Transfer Learning | MobileNetV2 | Pre-trained |
| Data Processing | NumPy, Pandas | ≥1.21.0, ≥1.5.0 |
| Image Processing | Pillow | ≥9.0.0 |
| ML Utilities | Scikit-learn | ≥1.0.0 |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | Streamlit | ≥1.28.0 |
| Visualization | Plotly | ≥5.14.0 |
| Charts | Seaborn | ≥0.11.0 |
| Plotting | Matplotlib | ≥3.5.0 |

### Authentication & APIs
| Component | Technology | Version |
|-----------|-----------|---------|
| Authentication | Google OAuth 2.0 | - |
| HTTP Requests | Requests | ≥2.31.0 |
| Google Auth | google-auth | ≥2.28.0 |

### Database
| Component | Technology |
|-----------|-----------|
| Primary | SQLite 3 |
| Format | users.db |
| Backup Format | CSV files |

### Environment
| Component | Specification |
|-----------|-----------|
| Python | 3.8+ |
| OS | Windows, Linux, macOS |
| Runtime | Streamlit Server |

---

## API Endpoints

### Authentication APIs
```
POST /auth/login
  Input: {username, password}
  Output: {token, user_id, status}

POST /auth/google-callback
  Input: {code}
  Output: {token, user_id}

POST /auth/logout
  Input: {token}
  Output: {status}
```

### Prediction APIs
```
POST /predict
  Input: {image_file, user_id}
  Output: {
    predictions: [
      {disease, confidence},
      ...
    ],
    recommendations: [...],
    hospitals: [...]
  }
```

### Recommendation APIs
```
GET /hospitals/{disease}
  Input: disease name
  Output: [hospital objects]

GET /doctors/{language}/{disease}
  Input: language, disease
  Output: [doctor objects]

GET /recommendations/{disease}
  Input: disease name
  Output: {health_tips, prevention, treatment}
```

### Feedback APIs
```
POST /feedback
  Input: {user_id, prediction, actual, feedback}
  Output: {status, feedback_id}

GET /feedback/{user_id}
  Input: user_id
  Output: [feedback records]
```

---

## Security Architecture

### Authentication Security
- **Google OAuth 2.0**: Industry-standard authentication
- **Password Hashing**: PBKDF2 with 200,000 iterations
- **Session Management**: Secure token-based sessions
- **Reset Tokens**: Time-limited (30 minutes) password reset

### Data Security
- **User Data**: Encrypted in transit (HTTPS recommended)
- **Credentials**: Stored hashed in SQLite
- **Database**: SQLite with file permissions
- **API Keys**: Environment variables + Streamlit secrets

### Input Validation
- Image format validation
- File size limits
- Filename sanitization
- SQL injection prevention (parameterized queries)

### Privacy Considerations
- User data stored locally
- Feedback data separate from user credentials
- Optional data collection
- GDPR-compliant design

---

## Deployment Architecture

### Local Deployment
```
Developer Machine
  ├── Python 3.8+
  ├── Virtual Environment (.venv)
  ├── Installed Dependencies
  ├── Trained Models (models/)
  ├── Data (SkinDisease/)
  └── Streamlit Server (localhost:8501)
```

### Production Deployment (Recommended)
```
Cloud Server (Azure/AWS/GCP)
  ├── Container: Docker
  ├── Orchestration: Kubernetes (optional)
  ├── Web Server: Nginx/Apache
  ├── App Server: Streamlit Cloud / Gunicorn
  ├── Database: Cloud SQLite / PostgreSQL
  ├── Storage: Cloud Storage (models, data)
  └── Monitoring: Logging & Analytics
```

### Scaling Considerations
- Model caching for faster inference
- Load balancing for multiple users
- Database replication
- CDN for image storage
- Async processing for feedback

---

## File Organization

```
project-root/
├── SkinDisease/
│   ├── app.py                          # Main Streamlit app
│   ├── predict.py                      # ML inference
│   ├── train_baseline.py               # Model training
│   ├── fine_tune_model.py             # Fine-tuning
│   ├── validate_model.py              # Evaluation
│   ├── data_validation_cleaning.py    # Data processing
│   ├── requirements.txt                # Dependencies
│   ├── users.db                        # SQLite database
│   ├── .streamlit/                     # Streamlit config
│   ├── SkinDisease/                    # Dataset
│   │   ├── train/                      # Training images
│   │   ├── val/                        # Validation images
│   │   └── test/                       # Test images
│   ├── hospital_data.json              # Hospital info
│   └── doctor_consultation_data.json   # Doctor info
├── models/                             # Pre-trained models
│   ├── skin_disease_model_baseline_best.keras
│   ├── skin_disease_model_baseline_final.keras
│   ├── *.meta.json
│   └── class_names.txt
├── logs/                               # Training logs
├── results/                            # Test reports
├── tests/                              # Unit tests
│   └── test_suite.py
├── docs/                               # Documentation
├── README.md                           # Setup guide
└── MILESTONE_MAPPING.md               # Project milestones
```

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Model Inference Time | ~1-2 seconds | On GPU: <1s, CPU: 1-2s |
| Image Upload Size | <10 MB | Type: JPG, PNG, BMP |
| Supported Image Sizes | 50x50 to 5000x5000 px | Auto-resized to 224x224 |
| Database Query Time | <100 ms | For typical queries |
| API Response Time | <3 seconds | Including model inference |
| Concurrent Users | 10-50 | Depends on server hardware |

---

## Future Enhancement Opportunities

1. **Real-time Model Updates**: A/B testing new models
2. **Federated Learning**: Distributed model training
3. **Explainability**: GradCAM visualization of predictions
4. **Multi-language Support**: UI localization
5. **Mobile App**: Native iOS/Android applications
6. **Advanced Recommendations**: ML-based doctor matching
7. **Telemedicine Integration**: Direct consultation booking
8. **Blockchain**: Secure medical records

---

## Maintenance & Monitoring

### Regular Tasks
- Monitor model performance metrics
- Update hospital/doctor database
- Review user feedback
- Security patching
- Dependency updates

### Alerting
- High error rate alerts
- Model accuracy degradation
- Database performance
- API latency warnings
- Unusual user activity

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Maintained By**: Development Team
