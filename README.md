# 🤖 Escrow Dispute Resolution AI Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered system that automatically predicts dispute resolution outcomes between buyers and sellers in escrow transactions based on complaint text analysis.

**📊 Model Performance:** 81.92% Accuracy

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage - Python](#usage---python)
- [Usage - Node.js](#usage---nodejs)
- [API Deployment](#api-deployment)
- [Model Files](#model-files)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This machine learning model analyzes consumer complaint narratives and predicts dispute resolution outcomes with **81.92% accuracy**.

### Prediction Classes

- **`favour_customer`** - Customer wins (refund/relief granted)
- **`favor_seller`** - Seller/Company wins (position upheld)  
- **`split_payment`** - Compromise (partial refund/non-monetary relief)

### Key Features

✅ **High Accuracy** - 81.92% on test data  
✅ **Real-time Predictions** - Process complaints in milliseconds  
✅ **Batch Processing** - Handle multiple complaints simultaneously  
✅ **Confidence Scores** - Get probability scores for each outcome  
✅ **Production Ready** - Pre-trained models ready to use  
✅ **Multi-language Support** - Python & Node.js integration  

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.92% |
| **Precision** | 78.62% |
| **Recall** | 81.92% |
| **F1-Score** | 76.63% |

**Algorithm:** Linear Support Vector Classifier (Linear SVC)  
**Training Data:** 125,250 complaint records (SMOTE-balanced)  
**Test Data:** 31,313 records  
**Features:** 3,007 (3,000 TF-IDF + 7 numerical features)

---

## 🚀 Quick Start

### Clone the Repository
```bash
git clone https://github.com/Maxetoo/escrow-dispute-model.git
cd escrow-dispute-model
```

### Python Quick Test
```python
import pickle
import numpy as np
from scipy.sparse import csr_matrix, hstack

# Load model
with open('models/dispute_resolution_model_latest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer_latest.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/target_encoder_latest.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Predict
complaint = "Company charged unauthorized fees. Want refund now!"
text_features = vectorizer.transform([complaint])
prediction = model.predict(text_features)
outcome = encoder.inverse_transform(prediction)[0]

print(f"Predicted Outcome: {outcome}")
# Output: Predicted Outcome: favour_customer
```

---

## 📦 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **Node.js:** 14 or higher (for Node.js integration)
- **Git:** For cloning the repository

### Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
imbalanced-learn>=0.8.0
```

### Node.js Environment Setup
```bash
# Install Node.js dependencies
npm install
```

**package.json:**
```json
{
  "name": "escrow-dispute-resolver",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.2",
    "python-shell": "^5.0.0",
    "body-parser": "^1.20.2"
  }
}
```

---

## 🐍 Usage - Python

### Method 1: Direct Model Loading
```python
import pickle
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack

# ===========================
# STEP 1: LOAD MODEL COMPONENTS
# ===========================

def load_model_components():
    """Load all saved model components"""
    
    # Load main model
    with open('models/dispute_resolution_model_latest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open('models/tfidf_vectorizer_latest.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load target encoder
    with open('models/target_encoder_latest.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Load feature info
    with open('models/feature_info_latest.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    
    return model, vectorizer, encoder, feature_info

# ===========================
# STEP 2: TEXT PREPROCESSING
# ===========================

def clean_text(text):
    """Clean and preprocess complaint text"""
    if not text:
        return ''
    
    text = str(text).lower()
    
    # Remove URLs, emails, phone numbers
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_stopwords(text):
    """Remove common English stopwords"""
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 
        'he', 'him', 'his', 'she', 'her', 'it', 'they', 'them',
        'what', 'which', 'who', 'this', 'that', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to',
        'from', 'in', 'out', 'on', 'off', 'over', 'under'
    }
    
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    return ' '.join(filtered)

def extract_features(text):
    """Extract numerical features from text"""
    
    negative_words = ['terrible', 'horrible', 'worst', 'awful', 'bad', 
                     'poor', 'unacceptable', 'frustrated', 'angry']
    positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy']
    financial_terms = ['payment', 'fee', 'charge', 'refund', 'money', 
                      'balance', 'account', 'credit']
    urgency_words = ['urgent', 'immediately', 'asap', 'emergency']
    
    words = text.split()
    
    return {
        'word_count': len(words),
        'negative_count': sum(1 for w in words if w in negative_words),
        'positive_count': sum(1 for w in words if w in positive_words),
        'urgency': 1 if any(w in words for w in urgency_words) else 0,
        'financial_count': sum(1 for w in words if w in financial_terms),
        'question_count': text.count('?'),
        'exclamation_count': text.count('!')
    }

# ===========================
# STEP 3: PREDICTION FUNCTION
# ===========================

def predict_dispute(complaint_text, model, vectorizer, encoder):
    """
    Predict dispute resolution outcome
    
    Args:
        complaint_text (str): Raw complaint text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        encoder: Target label encoder
        
    Returns:
        dict: Prediction result with confidence scores
    """
    
    # Preprocess text
    cleaned = clean_text(complaint_text)
    processed = remove_stopwords(cleaned)
    
    if len(processed) < 5:
        return {'error': 'Text too short after preprocessing'}
    
    # Extract features
    features = extract_features(processed)
    
    # Convert text to TF-IDF
    text_tfidf = vectorizer.transform([processed])
    
    # Create numerical features
    numerical = np.array([[
        features['word_count'],
        features['negative_count'],
        features['positive_count'],
        features['urgency'],
        features['financial_count'],
        features['question_count'],
        features['exclamation_count']
    ]])
    
    # Combine features (keep sparse)
    numerical_sparse = csr_matrix(numerical)
    X = hstack([text_tfidf, numerical_sparse])
    
    # Make prediction
    prediction = model.predict(X)[0]
    outcome = encoder.inverse_transform([prediction])[0]
    
    # Get confidence scores
    try:
        scores = model.decision_function(X)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        confidence = {
            label: float(prob) 
            for label, prob in zip(encoder.classes_, probs)
        }
    except:
        confidence = {outcome: 1.0}
    
    return {
        'prediction': outcome,
        'confidence': confidence,
        'text_features': features
    }

# ===========================
# STEP 4: USAGE EXAMPLE
# ===========================

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model, vectorizer, encoder, feature_info = load_model_components()
    print("✅ Model loaded successfully!\n")
    
    # Example complaint
    complaint = """
    I was charged a $50 late fee even though I paid 2 days before the due date.
    I have proof from my bank statement. This is completely unacceptable and unfair.
    I want my money refunded immediately!
    """
    
    # Predict
    print("Making prediction...")
    result = predict_dispute(complaint, model, vectorizer, encoder)
    
    # Display result
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\n🎯 Predicted Outcome: {result['prediction'].upper()}")
    print(f"\n📊 Confidence Scores:")
    for label, score in sorted(result['confidence'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"   {label:20s} {score*100:>6.2f}%")
    print("\n" + "="*60)
```

### Method 2: Using the Prediction Module

Create a file `predict.py`:
```python
# predict.py - Simple prediction module

from model_loader import load_model_components, predict_dispute

def main():
    # Load model once
    model, vectorizer, encoder, _ = load_model_components()
    
    # Make predictions
    complaints = [
        "Unauthorized charges on my account. Need refund!",
        "Applied for loan but denied due to credit score.",
        "Product damaged, they want me to pay return shipping."
    ]
    
    for i, complaint in enumerate(complaints, 1):
        result = predict_dispute(complaint, model, vectorizer, encoder)
        print(f"\nComplaint {i}: {result['prediction']}")
        print(f"Confidence: {max(result['confidence'].values())*100:.1f}%")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python predict.py
```

---

## 🟢 Usage - Node.js

### Method 1: Using Python Shell (Recommended)

**File: `predictor.js`**
```javascript
const { PythonShell } = require('python-shell');

/**
 * DisputePredictor class for Node.js
 */
class DisputePredictor {
    constructor(modelPath = './models') {
        this.modelPath = modelPath;
    }

    /**
     * Predict dispute resolution outcome
     * @param {string} complaintText - The complaint text
     * @returns {Promise<object>} - Prediction result
     */
    async predict(complaintText) {
        return new Promise((resolve, reject) => {
            const options = {
                mode: 'json',
                pythonPath: 'python3', // or 'python' on Windows
                pythonOptions: ['-u'],
                scriptPath: './python_scripts',
                args: [complaintText, this.modelPath]
            };

            PythonShell.run('predict.py', options, (err, results) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(results[0]);
                }
            });
        });
    }

    /**
     * Predict multiple complaints
     * @param {Array<string>} complaints - Array of complaint texts
     * @returns {Promise<Array<object>>} - Array of predictions
     */
    async predictBatch(complaints) {
        const predictions = [];
        
        for (const complaint of complaints) {
            try {
                const result = await this.predict(complaint);
                predictions.push(result);
            } catch (error) {
                predictions.push({ error: error.message });
            }
        }
        
        return predictions;
    }
}

module.exports = DisputePredictor;

// ============================================
// USAGE EXAMPLE
// ============================================

async function main() {
    const predictor = new DisputePredictor();
    
    const complaint = `
        I was charged unauthorized fees. Customer service was rude.
        I want a full refund immediately. This is unacceptable!
    `;
    
    try {
        console.log('Making prediction...\n');
        
        const result = await predictor.predict(complaint);
        
        console.log('================================');
        console.log('PREDICTION RESULT');
        console.log('================================');
        console.log(`\n🎯 Outcome: ${result.prediction}`);
        console.log(`\n📊 Confidence Scores:`);
        
        for (const [label, score] of Object.entries(result.confidence)) {
            console.log(`   ${label.padEnd(20)} ${(score * 100).toFixed(2)}%`);
        }
        
        console.log('\n================================\n');
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}
```

**Python script: `python_scripts/predict.py`**
```python
# python_scripts/predict.py
import sys
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, hstack
import re

def load_models(model_path):
    """Load model components"""
    with open(f'{model_path}/dispute_resolution_model_latest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_path}/tfidf_vectorizer_latest.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'{model_path}/target_encoder_latest.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, vectorizer, encoder

def clean_text(text):
    """Clean text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_features(text):
    """Extract numerical features"""
    negative_words = ['terrible', 'horrible', 'worst', 'awful', 'bad', 'poor']
    positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy']
    financial_terms = ['payment', 'fee', 'charge', 'refund', 'money']
    urgency_words = ['urgent', 'immediately', 'asap', 'emergency']
    
    words = text.split()
    
    return [
        len(words),
        sum(1 for w in words if w in negative_words),
        sum(1 for w in words if w in positive_words),
        1 if any(w in words for w in urgency_words) else 0,
        sum(1 for w in words if w in financial_terms),
        text.count('?'),
        text.count('!')
    ]

def predict(complaint_text, model_path='./models'):
    """Make prediction"""
    # Load models
    model, vectorizer, encoder = load_models(model_path)
    
    # Preprocess
    cleaned = clean_text(complaint_text)
    
    # TF-IDF
    text_tfidf = vectorizer.transform([cleaned])
    
    # Numerical features
    features = extract_features(cleaned)
    numerical = csr_matrix([features])
    
    # Combine
    X = hstack([text_tfidf, numerical])
    
    # Predict
    prediction = model.predict(X)[0]
    outcome = encoder.inverse_transform([prediction])[0]
    
    # Confidence
    try:
        scores = model.decision_function(X)[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        confidence = {label: float(prob) for label, prob in zip(encoder.classes_, probs)}
    except:
        confidence = {outcome: 1.0}
    
    return {
        'prediction': outcome,
        'confidence': confidence
    }

if __name__ == '__main__':
    complaint_text = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else './models'
    
    result = predict(complaint_text, model_path)
    print(json.dumps(result))
```

**Usage in Node.js:**
```javascript
const DisputePredictor = require('./predictor');

const predictor = new DisputePredictor();

// Single prediction
predictor.predict('Unauthorized charges. Want refund!')
    .then(result => {
        console.log('Prediction:', result.prediction);
        console.log('Confidence:', result.confidence);
    })
    .catch(error => console.error(error));

// Batch prediction
const complaints = [
    'Charged fees without authorization',
    'Loan denied due to credit score',
    'Product damaged, unfair return policy'
];

predictor.predictBatch(complaints)
    .then(results => {
        results.forEach((result, i) => {
            console.log(`\nComplaint ${i+1}:`);
            console.log('Prediction:', result.prediction);
        });
    });
```

### Method 2: REST API with Express

**File: `server.js`**
```javascript
const express = require('express');
const bodyParser = require('body-parser');
const DisputePredictor = require('./predictor');

const app = express();
const predictor = new DisputePredictor();

app.use(bodyParser.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'OK', message: 'Dispute Resolution API is running' });
});

// Single prediction endpoint
app.post('/api/predict', async (req, res) => {
    try {
        const { complaint_text } = req.body;
        
        if (!complaint_text) {
            return res.status(400).json({ 
                error: 'complaint_text is required' 
            });
        }
        
        const result = await predictor.predict(complaint_text);
        
        res.json({
            success: true,
            data: result
        });
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Batch prediction endpoint
app.post('/api/predict/batch', async (req, res) => {
    try {
        const { complaints } = req.body;
        
        if (!Array.isArray(complaints)) {
            return res.status(400).json({ 
                error: 'complaints must be an array' 
            });
        }
        
        const results = await predictor.predictBatch(complaints);
        
        res.json({
            success: true,
            count: results.length,
            data: results
        });
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`✅ Dispute Resolution API running on port ${PORT}`);
    console.log(`\n📍 Endpoints:`);
    console.log(`   GET  http://localhost:${PORT}/health`);
    console.log(`   POST http://localhost:${PORT}/api/predict`);
    console.log(`   POST http://localhost:${PORT}/api/predict/batch`);
});
```

**Start the server:**
```bash
node server.js
```

**Test with cURL:**
```bash
# Single prediction
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "Company charged unauthorized fees. Want refund now!"
  }'

# Batch prediction
curl -X POST http://localhost:3000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      "Unauthorized fees charged",
      "Loan denied due to credit",
      "Product damaged, unfair policy"
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": "favour_customer",
    "confidence": {
      "favour_customer": 0.7845,
      "favor_seller": 0.1523,
      "split_payment": 0.0632
    }
  }
}
```

---

## 🌐 API Deployment

### Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create app
heroku create escrow-dispute-api

# Add Python buildpack
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 heroku/nodejs

# Deploy
git push heroku main

# Open
heroku open
```

### Deploy to AWS Lambda

Use AWS SAM or Serverless Framework with Lambda Layers for Python dependencies.

### Deploy to Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/dispute-resolver

# Deploy
gcloud run deploy dispute-resolver \
  --image gcr.io/PROJECT_ID/dispute-resolver \
  --platform managed
```

---

## 📂 Model Files

All trained model components are in the `models/` directory:

| File | Description | Size |
|------|-------------|------|
| `dispute_resolution_model_latest.pkl` | Trained Linear SVC model | ~50 MB |
| `tfidf_vectorizer_latest.pkl` | TF-IDF vectorizer (3000 features) | ~15 MB |
| `target_encoder_latest.pkl` | Label encoder for outcomes | <1 MB |
| `feature_info_latest.pkl` | Feature metadata | <1 MB |
| `model_metadata_latest.pkl` | Training metadata | <1 MB |

**Total:** ~66 MB

---

## 📊 Dataset

**Source:** [CFPB Consumer Complaints Database (Kaggle)](https://www.kaggle.com/datasets/heemalichaudhari/consumer-complaints)

- **Original Records:** 777,959
- **After Preprocessing:** 156,563 (with complaint narratives)
- **Training Set:** 125,250 (after SMOTE balancing)
- **Test Set:** 31,313

### Class Distribution (Before SMOTE)

- `favor_seller`: 80.15%
- `split_payment`: 12.46%
- `favour_customer`: 7.39%

---

## 🔧 Training Pipeline

The complete training pipeline is in the `notebooks/` directory:

1. **`01_explore_data.ipynb`** - Data exploration and analysis
2. **`02_create_target_labels.ipynb`** - Label mapping and creation
3. **`03_data_cleaning_text_preprocessing.ipynb`** - Text cleaning
4. **`04_feature_engineering_model_building.ipynb`** - Model training
5. **`05_prediction_system.ipynb`** - Prediction interface

### To Retrain the Model:
```bash
# Run notebooks in order
jupyter notebook notebooks/01_explore_data.ipynb

# Or use the training script
python train_model.py --data data/raw/complaints.csv --output models/
```

---

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

### Issue: Model file not found
```bash
# Ensure you're in the correct directory
ls models/  # Should show all .pkl files

# Or download models from releases
wget https://github.com/yourusername/escrow-dispute-model/releases/download/v1.0/models.zip
unzip models.zip
```

### Issue: Python version mismatch
```bash
# Check Python version
python --version  # Should be 3.8+

# Use specific Python version
python3.9 predict.py
```

### Issue: Node.js can't find Python
```javascript
// In predictor.js, specify Python path explicitly
const options = {
    pythonPath: '/usr/bin/python3',  // Your Python path
    // ... rest of options
};
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Development Setup
```bash
# Clone repo
git clone https://github.com/yourusername/escrow-dispute-model.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---


## 📧 Contact

**Etombi Maxwell** 

**Project Link:** [https://github.com/yourusername/escrow-dispute-model](https://github.com/Maxetoo/escrow-dispute-model)

---



---

**Made with ❤️ for fair dispute resolution**
