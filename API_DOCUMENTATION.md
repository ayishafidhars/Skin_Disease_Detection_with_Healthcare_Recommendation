# 📚 API_DOCUMENTATION.md - Complete API Reference

## Skin Disease Detection System - API Documentation

**Version**: 1.0  
**Last Updated**: March 2026  
**Base URL (Local)**: `http://localhost:8501`  
**Base URL (Production)**: `https://api.skindisease.com`

---

## Table of Contents
1. [Authentication](#authentication)
2. [Disease Prediction API](#disease-prediction-api)
3. [Hospital Recommendation API](#hospital-recommendation-api)
4. [Doctor Recommendation API](#doctor-recommendation-api)
5. [Health Information API](#health-information-api)
6. [Feedback API](#feedback-api)
7. [User Management API](#user-management-api)
8. [Error Codes](#error-codes)

---

## Authentication

### Overview
The API uses two authentication methods:
1. **Streamlit Session**: For web interface
2. **Google OAuth 2.0**: For third-party apps

### Headers

All API requests must include:
```http
Content-Type: application/json
Authorization: Bearer {token}
```

### Login Endpoint

**Endpoint**: `POST /api/auth/login`

**Request**:
```json
{
  "username": "user123",
  "password": "securepassword123"
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user_id": 42,
  "username": "user123",
  "email": "user@example.com"
}
```

**Errors**:
- `401 Unauthorized`: Invalid credentials
- `400 Bad Request`: Missing username or password

---

## Disease Prediction API

### POST /api/predict

Predict skin disease from an uploaded image.

**Request** (Multipart Form Data):
```
POST /api/predict HTTP/1.1
Content-Type: multipart/form-data

image: <binary image file>
user_id: 42
```

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image` | File | Yes | JPG, PNG, or BMP image (< 10 MB) |
| `user_id` | Integer | Yes | Authenticated user ID |

**Response** (200 OK):
```json
{
  "status": "success",
  "prediction_id": "pred_12345",
  "timestamp": "2026-03-27T10:30:00Z",
  "predictions": [
    {
      "rank": 1,
      "disease": "Acne",
      "confidence": 0.95,
      "probability": 95.0
    },
    {
      "rank": 2,
      "disease": "Folliculitis",
      "confidence": 0.03,
      "probability": 3.0
    },
    {
      "rank": 3,
      "disease": "Eczema",
      "confidence": 0.02,
      "probability": 2.0
    }
  ],
  "processing_time_ms": 1250
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `status` | String | "success" or "error" |
| `prediction_id` | String | Unique prediction identifier |
| `timestamp` | DateTime | UTC timestamp of prediction |
| `predictions` | Array | Array of predictions (top 3) |
| `confidence` | Float | Confidence score (0-1) |
| `disease` | String | Predicted disease name |
| `processing_time_ms` | Integer | Time taken (milliseconds) |

**Example with cURL**:
```bash
curl -X POST http://localhost:8501/api/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@skin_image.jpg" \
  -F "user_id=42"
```

**Example with Python**:
```python
import requests

files = {'image': open('skin_image.jpg', 'rb')}
data = {'user_id': 42}
headers = {'Authorization': 'Bearer YOUR_TOKEN'}

response = requests.post(
    'http://localhost:8501/api/predict',
    files=files,
    data=data,
    headers=headers
)

predictions = response.json()['predictions']
print(f"Top prediction: {predictions[0]['disease']} ({predictions[0]['confidence']*100}%)")
```

**Error Responses**:
- `400 Bad Request`: Invalid image format
- `413 Payload Too Large`: Image file too large
- `401 Unauthorized`: Invalid or missing token
- `404 Not Found`: User ID not found
- `500 Internal Server Error`: Model prediction failed

---

## Hospital Recommendation API

### GET /api/hospitals/{disease}

Get list of recommended hospitals for a disease.

**Parameters**:
| Name | Type | Location | Description |
|------|------|----------|-------------|
| `disease` | String | URL | Disease name (URL-encoded) |
| `latitude` | Float | Query | User latitude (optional) |
| `longitude` | Float | Query | User longitude (optional) |
| `radius_km` | Integer | Query | Search radius in kilometers (default: 50) |

**Example Request**:
```http
GET /api/hospitals/Acne?latitude=40.7128&longitude=-74.0060&radius_km=100
Authorization: Bearer YOUR_TOKEN
```

**Response** (200 OK):
```json
{
  "status": "success",
  "disease": "Acne",
  "hospitals_found": 5,
  "hospitals": [
    {
      "hospital_id": "hosp_001",
      "name": "City Dermatology Center",
      "address": "123 Main Street, New York, NY 10001",
      "phone": "+1-212-555-0123",
      "email": "info@citydermatology.com",
      "specialization": ["Dermatology", "Acne Treatment"],
      "distance_km": 2.3,
      "rating": 4.8,
      "reviews": 156,
      "availability": {
        "today": true,
        "this_week": true,
        "appointment_types": ["In-person", "Telemedicine"]
      },
      "website": "https://citydermatology.com",
      "hours": "9:00-18:00 (Mon-Fri)"
    },
    // ... more hospitals
  ]
}
```

**Example with Python**:
```python
import requests

headers = {'Authorization': 'Bearer YOUR_TOKEN'}
params = {
    'latitude': 40.7128,
    'longitude': -74.0060,
    'radius_km': 50
}

response = requests.get(
    'http://localhost:8501/api/hospitals/Acne',
    headers=headers,
    params=params
)

hospitals = response.json()['hospitals']
for hospital in hospitals[:3]:  # Top 3
    print(f"{hospital['name']} - {hospital['distance_km']}km away")
```

**Error Codes**:
- `400 Bad Request`: Invalid disease name or coordinates
- `401 Unauthorized`: Missing authentication
- `404 Not Found`: No hospitals found

---

## Doctor Recommendation API

### GET /api/doctors/{disease}

Get list of recommended doctors for a disease.

**Parameters**:
| Name | Type | Location | Description |
|------|------|----------|-------------|
| `disease` | String | URL | Disease name (URL-encoded) |
| `language` | String | Query | Preferred language (e.g., "en", "es", "fr") |
| `latitude` | Float | Query | User latitude (optional) |
| `longitude` | Float | Query | User longitude (optional) |

**Example Request**:
```http
GET /api/doctors/Acne?language=en&latitude=40.7128&longitude=-74.0060
Authorization: Bearer YOUR_TOKEN
```

**Response** (200 OK):
```json
{
  "status": "success",
  "disease": "Acne",
  "language": "en",
  "doctors_found": 3,
  "doctors": [
    {
      "doctor_id": "doc_042",
      "name": "Dr. Sarah Johnson, MD",
      "specialization": "Dermatology",
      "credentials": ["MD", "Board Certified", "15+ years experience"],
      "languages": ["English", "Spanish"],
      "clinic": {
        "name": "Dermatology Associates",
        "address": "456 Columbus Ave, New York, NY 10025",
        "phone": "+1-212-555-0456",
        "hours": "9:00-17:00 (Mon-Fri)"
      },
      "consultation_fee": {
        "currency": "USD",
        "in_person": 150,
        "telemedicine": 100
      },
      "availability": {
        "next_available": "2026-03-29T14:00:00Z",
        "appointment_types": ["In-person", "Video Call"]
      },
      "rating": 4.9,
      "reviews": 87,
      "website": "https://dermatologyassociates.com",
      "distance_km": 3.1
    },
    // ... more doctors
  ]
}
```

**Example with Python**:
```python
import requests
from datetime import datetime

headers = {'Authorization': 'Bearer YOUR_TOKEN'}
params = {'language': 'en'}

response = requests.get(
    'http://localhost:8501/api/doctors/Melanoma',  # Urgent case
    headers=headers,
    params=params
)

doctors = response.json()['doctors']
if doctors:
    best_doctor = doctors[0]
    print(f"Recommended: {best_doctor['name']}")
    print(f"Rating: {best_doctor['rating']}/5.0")
    print(f"Available: {best_doctor['availability']['next_available']}")
```

**Error Codes**:
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing authentication
- `404 Not Found`: No doctors available

---

## Health Information API

### GET /api/health-info/{disease}

Get detailed health information for a disease.

**Parameters**:
| Name | Type | Location | Description |
|------|------|----------|-------------|
| `disease` | String | URL | Disease name |

**Example Request**:
```http
GET /api/health-info/Acne
Authorization: Bearer YOUR_TOKEN
```

**Response** (200 OK):
```json
{
  "status": "success",
  "disease": "Acne",
  "information": {
    "description": "Acne is a skin condition that occurs when hair follicles become clogged with dead skin cells and sebum...",
    "common_symptoms": [
      "Pimples (blackheads, whiteheads, pustules)",
      "Oily skin",
      "Inflamed skin",
      "Scarring (in severe cases)"
    ],
    "causes": [
      "Excess sebum production",
      "Bacterial growth (Propionibacterium acnes)",
      "Hormonal changes",
      "Clogged pores"
    ],
    "risk_factors": [
      "Teenage years (hormones)",
      "Family history",
      "Certain medications",
      "High humidity"
    ],
    "prevention": [
      "Keep skin clean with gentle cleanser",
      "Use non-comedogenic moisturizers",
      "Avoid touching face",
      "Change pillow cases frequently"
    ],
    "treatment_options": [
      "Topical treatments: benzoyl peroxide, salicylic acid",
      "Oral medications: antibiotics, birth control (for women)",
      "Procedures: laser therapy, chemical peels"
    ],
    "when_to_see_doctor": "If acne persists despite home treatment for 8-12 weeks, or if cystic acne develops",
    "severity": "Low to Medium",
    "contagious": false,
    "in_urg_emergency": false
  },
  "disclaimer": "This is for informational purposes. Consult a healthcare professional for diagnosis."
}
```

---

## Feedback API

### POST /api/feedback

Submit feedback on a prediction.

**Request**:
```json
{
  "prediction_id": "pred_12345",
  "user_id": 42,
  "predicted_disease": "Acne",
  "actual_disease": "Acne",
  "accuracy": 10,
  "comments": "Perfect diagnosis!",
  "doctor_confirmed": true
}
```

**Response** (201 Created):
```json
{
  "status": "success",
  "feedback_id": "fb_98765",
  "message": "Thank you for your feedback!"
}
```

### GET /api/feedback/{user_id}

Retrieve user's feedback history.

**Response** (200 OK):
```json
{
  "status": "success",
  "user_id": 42,
  "feedback_count": 5,
  "feedback": [
    {
      "feedback_id": "fb_98765",
      "timestamp": "2026-03-27T10:30:00Z",
      "prediction_id": "pred_12345",
      "predicted": "Acne",
      "actual": "Acne",
      "accuracy_rating": 10,
      "comments": "Perfect diagnosis!"
    }
    // ... more feedback
  ]
}
```

---

## User Management API

### GET /api/user/profile

Get user profile information.

**Response** (200 OK):
```json
{
  "user_id": 42,
  "username": "user123",
  "email": "user@example.com",
  "phone": "+1-555-0123",
  "age": 28,
  "gender": "Female",
  "created_at": "2026-01-15T08:30:00Z",
  "total_predictions": 5,
  "total_feedback": 3,
  "profile_complete": true
}
```

### PUT /api/user/profile

Update user profile.

**Request**:
```json
{
  "phone": "+1-555-9876",
  "age": 29,
  "gender": "Female"
}
```

### POST /api/user/change-password

Change user password.

**Request**:
```json
{
  "old_password": "currentpassword",
  "new_password": "newpassword123"
}
```

### DELETE /api/user/delete-account

Delete user account (permanent).

**Request**:
```json
{
  "password": "userpassword",
  "confirm": true
}
```

---

## Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid parameters or format |
| 401 | Unauthorized | Missing/invalid authentication token |
| 403 | Forbidden | Permission denied |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File exceeds max size |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Server Error | Internal server error |
| 503 | Service Unavailable | Server temporarily unavailable |

**Error Response Format**:
```json
{
  "status": "error",
  "error_code": 400,
  "message": "Invalid image format. Only JPG, PNG, BMP supported.",
  "details": "Image format: WEBP"
}
```

---

## Rate Limiting

Current rate limits (for production):
- **Unauthenticated requests**: 10 requests/minute
- **Authenticated requests**: 100 requests/minute
- **Predictions**: 10 per hour per user

---

## Versioning

Current API Version: **1.0**

All endpoints include version in URL structure (future):
```
/api/v1/predict
/api/v2/hospitals
```

---

## Code Examples

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function predictDisease(imageFile, token) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('user_id', 42);

  const response = await axios.post('http://localhost:8501/api/predict', formData, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'multipart/form-data'
    }
  });

  return response.data.predictions;
}
```

### Java
```java
import okhttp3.*;
import java.io.File;

public class SkinDiseaseAPI {
    public static void predict(String token, File imageFile) throws Exception {
        OkHttpClient client = new OkHttpClient();
        RequestBody body = new MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "image.jpg", RequestBody.create(imageFile, MediaType.parse("image/jpeg")))
            .addFormDataPart("user_id", "42")
            .build();

        Request request = new Request.Builder()
            .url("http://localhost:8501/api/predict")
            .header("Authorization", "Bearer " + token)
            .post(body)
            .build();

        Response response = client.newCall(request).execute();
        System.out.println(response.body().string());
    }
}
```

---

## Support & Feedback

- **Issues**: GitHub Issues
- **Questions**: support@skindisease.com
- **Documentation**: https://docs.skindisease.com

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Next Update**: June 2026
