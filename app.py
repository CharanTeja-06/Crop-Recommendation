"""
Crop Recommendation System
===========================
A machine learning-powered web application that recommends the best crop
to grow based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall.

Uses Random Forest Classifier trained on the Crop_recommendation.csv dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template_string, request, jsonify
import warnings

warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

print("=" * 60)
print("  [*] Crop Recommendation System - Loading Data...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"[DATA] Columns: {list(df.columns)}")
print(f"[DATA] Unique Crops: {df['label'].nunique()} -> {sorted(df['label'].unique())}")
print(f"\n[DATA] Dataset Summary:")
print(df.describe().round(2))

# Features & target
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. MODEL TRAINING & EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("  [*] Training ML Models...")
print("=" * 60)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
}

best_model = None
best_accuracy = 0
best_model_name = ""
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  [OK] {name}: {acc:.4f} ({acc*100:.2f}%)")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n[BEST] Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# Detailed report for the best model
y_pred_best = best_model.predict(X_test_scaled)
print(f"\n[REPORT] Classification Report for {best_model_name}:")
print(
    classification_report(
        y_test, y_pred_best, target_names=label_encoder.classes_
    )
)

# ============================================================
# 3. CROP INFO DATABASE
# ============================================================

CROP_INFO = {
    "rice": {
        "emoji": "🌾",
        "color": "#4CAF50",
        "season": "Kharif (June–November)",
        "water": "High",
        "description": "Rice is a staple food crop that thrives in warm, humid climates with abundant water supply. It requires clay or loam soil with good water retention.",
    },
    "maize": {
        "emoji": "🌽",
        "color": "#FFC107",
        "season": "Kharif / Rabi",
        "water": "Moderate",
        "description": "Maize (corn) is a versatile cereal crop grown for food, feed, and industrial purposes. It prefers well-drained fertile soils and warm temperatures.",
    },
    "chickpea": {
        "emoji": "🫘",
        "color": "#8D6E63",
        "season": "Rabi (October–March)",
        "water": "Low",
        "description": "Chickpea is a protein-rich pulse crop well-suited for dry regions. It fixes atmospheric nitrogen, improving soil fertility for subsequent crops.",
    },
    "kidneybeans": {
        "emoji": "🫘",
        "color": "#D32F2F",
        "season": "Kharif / Rabi",
        "water": "Moderate",
        "description": "Kidney beans are high-protein legumes that grow well in well-drained loamy soil. They prefer cool to moderate temperatures.",
    },
    "pigeonpeas": {
        "emoji": "🌱",
        "color": "#689F38",
        "season": "Kharif (June–November)",
        "water": "Low to Moderate",
        "description": "Pigeon peas (tur/arhar dal) are drought-resistant legumes rich in protein. They are a key pulse crop in tropical and subtropical regions.",
    },
    "mothbeans": {
        "emoji": "🌿",
        "color": "#AFB42B",
        "season": "Kharif (July–October)",
        "water": "Very Low",
        "description": "Moth beans are extremely drought-tolerant legumes ideal for arid and semi-arid regions. They enrich soil with nitrogen fixation.",
    },
    "mungbean": {
        "emoji": "🫛",
        "color": "#66BB6A",
        "season": "Kharif / Spring",
        "water": "Low to Moderate",
        "description": "Mung beans are short-duration pulse crops rich in protein and easy to digest. They are widely used in Asian cuisines.",
    },
    "blackgram": {
        "emoji": "⚫",
        "color": "#37474F",
        "season": "Kharif / Rabi",
        "water": "Low to Moderate",
        "description": "Black gram (urad dal) is a nutrient-rich pulse crop that thrives in warm, humid conditions. It is used extensively in Indian cuisine.",
    },
    "lentil": {
        "emoji": "🟤",
        "color": "#795548",
        "season": "Rabi (October–March)",
        "water": "Low",
        "description": "Lentils are high-protein, drought-tolerant pulses ideal for cool, dry climates. They are a staple in many cuisines worldwide.",
    },
    "pomegranate": {
        "emoji": "🍎",
        "color": "#C62828",
        "season": "Mrig Bahar (June–February)",
        "water": "Low to Moderate",
        "description": "Pomegranate is a hardy fruit crop that tolerates drought and saline conditions. It is rich in antioxidants and has high market value.",
    },
    "banana": {
        "emoji": "🍌",
        "color": "#FFD54F",
        "season": "Year-round (tropical)",
        "water": "High",
        "description": "Banana is a tropical fruit crop that requires warm temperatures, high humidity, and adequate water. It is one of the most consumed fruits worldwide.",
    },
    "mango": {
        "emoji": "🥭",
        "color": "#FF8F00",
        "season": "Summer (March–June)",
        "water": "Moderate",
        "description": "Mango, the 'King of Fruits', is a premium tropical fruit with high demand. It requires well-drained soil and dry periods for flowering.",
    },
    "grapes": {
        "emoji": "🍇",
        "color": "#7B1FA2",
        "season": "Winter–Summer",
        "water": "Low to Moderate",
        "description": "Grapes are high-value fruit crops suitable for wine, raisins, and fresh consumption. They prefer warm, dry climates with well-drained soil.",
    },
    "watermelon": {
        "emoji": "🍉",
        "color": "#E53935",
        "season": "Summer (February–June)",
        "water": "Moderate to High",
        "description": "Watermelon is a refreshing summer fruit that loves warm temperatures and sandy loam soil. It has short growing cycles and good market returns.",
    },
    "muskmelon": {
        "emoji": "🍈",
        "color": "#FFA726",
        "season": "Summer (February–May)",
        "water": "Moderate",
        "description": "Muskmelon is a sweet, aromatic fruit that grows well in hot, dry climates. It prefers well-drained sandy soil and ample sunshine.",
    },
    "apple": {
        "emoji": "🍎",
        "color": "#EF5350",
        "season": "Autumn (August–November)",
        "water": "Moderate",
        "description": "Apples are temperate fruit crops that require cold winters for proper fruiting. They thrive at higher altitudes with well-drained loamy soil.",
    },
    "orange": {
        "emoji": "🍊",
        "color": "#EF6C00",
        "season": "Winter (November–March)",
        "water": "Moderate",
        "description": "Oranges are citrus fruits rich in vitamin C. They prefer subtropical climates with moderate rainfall and well-drained soil.",
    },
    "papaya": {
        "emoji": "🍈",
        "color": "#FF7043",
        "season": "Year-round (tropical)",
        "water": "Moderate",
        "description": "Papaya is a fast-growing tropical fruit rich in vitamins and enzymes. It prefers warm climates and well-drained, fertile soil.",
    },
    "coconut": {
        "emoji": "🥥",
        "color": "#8D6E63",
        "season": "Year-round (tropical)",
        "water": "Moderate to High",
        "description": "Coconut is a versatile tropical palm crop providing oil, fiber, food, and water. It thrives in coastal areas with warm, humid climates.",
    },
    "cotton": {
        "emoji": "☁️",
        "color": "#ECEFF1",
        "season": "Kharif (April–November)",
        "water": "Moderate",
        "description": "Cotton is a major cash crop and key raw material for the textile industry. It requires warm temperatures and moderately moist conditions.",
    },
    "jute": {
        "emoji": "🌿",
        "color": "#827717",
        "season": "Kharif (March–July)",
        "water": "High",
        "description": "Jute, known as the 'Golden Fiber', is a natural fiber crop used for packaging and textiles. It grows best in warm, humid alluvial plains.",
    },
    "coffee": {
        "emoji": "☕",
        "color": "#4E342E",
        "season": "Year-round (perennial)",
        "water": "Moderate",
        "description": "Coffee is a high-value plantation crop grown in tropical highlands. It prefers shade, moderate rainfall, and well-drained, slightly acidic soil.",
    },
}


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Predict the best crop based on input parameters."""
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)

    # Get prediction & probabilities
    prediction = best_model.predict(features_scaled)
    probabilities = best_model.predict_proba(features_scaled)[0]

    predicted_crop = label_encoder.inverse_transform(prediction)[0]

    # Get top 3 recommendations
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = []
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx] * 100
        info = CROP_INFO.get(crop_name, {"emoji": "🌱", "color": "#4CAF50", "season": "N/A", "water": "N/A", "description": "No additional info available."})
        top_crops.append({
            "name": crop_name,
            "confidence": round(confidence, 2),
            "emoji": info["emoji"],
            "color": info["color"],
            "season": info["season"],
            "water": info["water"],
            "description": info["description"],
        })

    return predicted_crop, top_crops


# ============================================================
# 5. FLASK WEB APPLICATION
# ============================================================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI-powered Crop Recommendation System that suggests the best crop based on soil nutrients and weather conditions.">
    <title>🌾 Crop Recommendation System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1c;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(99, 102, 241, 0.2);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --accent-light: #818cf8;
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.3);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
            --gradient-2: linear-gradient(135deg, #10b981, #34d399);
            --gradient-3: linear-gradient(135deg, #f59e0b, #f97316);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 1;
        }

        /* Header */
        header {
            text-align: center;
            padding: 3rem 0 2rem;
            animation: fadeInDown 0.8s ease;
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .logo {
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3));
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 300;
        }

        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
            padding: 0.5rem 1.2rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--border-color);
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--accent-light);
        }

        .model-badge .dot {
            width: 8px; height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--success-glow); }
            50% { opacity: 0.7; box-shadow: 0 0 0 8px transparent; }
        }

        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            h1 { font-size: 1.8rem; }
            .logo { font-size: 2.5rem; }
        }

        /* Glass Card */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            animation: fadeInUp 0.8s ease;
            transition: border-color 0.3s ease;
        }

        .card:hover {
            border-color: rgba(99, 102, 241, 0.4);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .card-title .icon {
            font-size: 1.4rem;
        }

        /* Form Styling */
        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
            letter-spacing: 0.02em;
        }

        .input-wrapper {
            position: relative;
        }

        .input-wrapper .unit {
            position: absolute;
            right: 14px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.8rem;
            color: var(--text-muted);
            pointer-events: none;
        }

        input[type="number"] {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            outline: none;
        }

        input[type="number"]:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }

        input[type="number"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
            background: rgba(15, 23, 42, 1);
        }

        input[type="number"]::placeholder {
            color: var(--text-muted);
        }

        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        /* Buttons */
        .btn-predict {
            width: 100%;
            padding: 1rem;
            margin-top: 0.5rem;
            background: var(--gradient-1);
            border: none;
            border-radius: 14px;
            color: white;
            font-size: 1.05rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-predict::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
            transition: left 0.5s ease;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px var(--accent-glow);
        }

        .btn-predict:hover::before {
            left: 100%;
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        .btn-predict:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Section */
        .results-container {
            display: none;
        }

        .results-container.show {
            display: block;
            animation: fadeInUp 0.6s ease;
        }

        .primary-result {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(99, 102, 241, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 16px;
            margin-bottom: 1.5rem;
        }

        .primary-result .crop-emoji {
            font-size: 4rem;
            display: block;
            margin-bottom: 0.5rem;
            animation: bounceIn 0.6s ease;
        }

        @keyframes bounceIn {
            0% { transform: scale(0); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }

        .primary-result .crop-name {
            font-size: 2rem;
            font-weight: 800;
            text-transform: capitalize;
            margin-bottom: 0.3rem;
        }

        .primary-result .confidence {
            font-size: 1.1rem;
            color: var(--success);
            font-weight: 600;
        }

        .primary-result .description {
            margin-top: 1rem;
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .crop-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .detail-chip {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 10px;
            font-size: 0.85rem;
        }

        .detail-chip .label {
            color: var(--text-muted);
        }

        .detail-chip .value {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Alternative Crops */
        .alt-section-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .alt-crop {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 12px;
            margin-bottom: 0.6rem;
            transition: all 0.3s ease;
        }

        .alt-crop:hover {
            background: rgba(15, 23, 42, 0.8);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateX(5px);
        }

        .alt-crop .emoji {
            font-size: 2rem;
            min-width: 45px;
            text-align: center;
        }

        .alt-crop .info { flex: 1; }

        .alt-crop .name {
            font-weight: 600;
            text-transform: capitalize;
            font-size: 1rem;
        }

        .alt-crop .alt-confidence {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .confidence-bar {
            width: 100%;
            height: 4px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 4px;
            margin-top: 0.4rem;
            overflow: hidden;
        }

        .confidence-bar .fill {
            height: 100%;
            border-radius: 4px;
            background: var(--gradient-1);
            transition: width 1s ease;
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 20px; height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem 0;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--accent-light);
            text-decoration: none;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

        /* Range Hints */
        .range-hint {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">🌾</div>
            <h1>Crop Recommendation System</h1>
            <p class="subtitle">AI-powered suggestions for optimal crop selection based on soil & weather data</p>
            <div class="model-badge">
                <span class="dot"></span>
                {{ model_name }} — {{ accuracy }}% accuracy
            </div>
        </header>

        <div class="main-grid">
            <!-- Input Form -->
            <div class="card" style="animation-delay: 0.1s;">
                <div class="card-title">
                    <span class="icon">🧪</span> Soil & Weather Parameters
                </div>
                <form id="predict-form" onsubmit="handleSubmit(event)">

                    <div class="card-title" style="font-size: 0.95rem; color: var(--accent-light); margin-bottom: 1rem;">
                        <span class="icon">🧬</span> Soil Nutrients
                    </div>

                    <div class="input-grid">
                        <div class="form-group">
                            <label for="nitrogen">Nitrogen (N)</label>
                            <div class="input-wrapper">
                                <input type="number" id="nitrogen" name="nitrogen" placeholder="e.g. 90" step="any" required>
                                <span class="unit">kg/ha</span>
                            </div>
                            <div class="range-hint">Range: 0–140</div>
                        </div>
                        <div class="form-group">
                            <label for="phosphorus">Phosphorus (P)</label>
                            <div class="input-wrapper">
                                <input type="number" id="phosphorus" name="phosphorus" placeholder="e.g. 42" step="any" required>
                                <span class="unit">kg/ha</span>
                            </div>
                            <div class="range-hint">Range: 5–145</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="potassium">Potassium (K)</label>
                        <div class="input-wrapper">
                            <input type="number" id="potassium" name="potassium" placeholder="e.g. 43" step="any" required>
                            <span class="unit">kg/ha</span>
                        </div>
                        <div class="range-hint">Range: 5–205</div>
                    </div>

                    <div class="card-title" style="font-size: 0.95rem; color: var(--accent-light); margin-top: 1rem; margin-bottom: 1rem;">
                        <span class="icon">🌤️</span> Weather Conditions
                    </div>

                    <div class="input-grid">
                        <div class="form-group">
                            <label for="temperature">Temperature</label>
                            <div class="input-wrapper">
                                <input type="number" id="temperature" name="temperature" placeholder="e.g. 25" step="any" required>
                                <span class="unit">°C</span>
                            </div>
                            <div class="range-hint">Range: 8–44°C</div>
                        </div>
                        <div class="form-group">
                            <label for="humidity">Humidity</label>
                            <div class="input-wrapper">
                                <input type="number" id="humidity" name="humidity" placeholder="e.g. 80" step="any" required>
                                <span class="unit">%</span>
                            </div>
                            <div class="range-hint">Range: 14–100%</div>
                        </div>
                    </div>

                    <div class="input-grid">
                        <div class="form-group">
                            <label for="ph">Soil pH</label>
                            <div class="input-wrapper">
                                <input type="number" id="ph" name="ph" placeholder="e.g. 6.5" step="any" required>
                            </div>
                            <div class="range-hint">Range: 3.5–10</div>
                        </div>
                        <div class="form-group">
                            <label for="rainfall">Rainfall</label>
                            <div class="input-wrapper">
                                <input type="number" id="rainfall" name="rainfall" placeholder="e.g. 200" step="any" required>
                                <span class="unit">mm</span>
                            </div>
                            <div class="range-hint">Range: 20–300 mm</div>
                        </div>
                    </div>

                    <button type="submit" class="btn-predict" id="predict-btn">
                        🔍 Recommend Crop
                    </button>
                </form>
            </div>

            <!-- Results Panel -->
            <div>
                <div class="card results-container" id="results-panel">
                    <div class="card-title">
                        <span class="icon">✨</span> Recommendation
                    </div>

                    <div class="primary-result" id="primary-result">
                        <!-- Filled by JS -->
                    </div>

                    <div class="alt-section-title">Other suitable crops</div>
                    <div id="alt-crops">
                        <!-- Filled by JS -->
                    </div>
                </div>

                <!-- Placeholder when no results -->
                <div class="card" id="placeholder-panel">
                    <div style="text-align: center; padding: 3rem 1rem;">
                        <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.3;">🌱</div>
                        <p style="color: var(--text-muted); font-size: 1rem;">
                            Enter your soil and weather parameters<br>to get a crop recommendation
                        </p>
                        <div style="margin-top: 1.5rem; display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem;">
                            {% for crop in crops %}
                            <span style="padding: 0.3rem 0.7rem; background: rgba(99,102,241,0.08); border-radius: 20px; font-size: 0.8rem; color: var(--text-secondary);">
                                {{ crop }}
                            </span>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>Built with 🤍 using Flask &amp; Scikit-learn — Trained on {{ n_samples }} samples across {{ n_crops }} crop types</p>
        </footer>
    </div>

    <script>
        async function handleSubmit(e) {
            e.preventDefault();

            const btn = document.getElementById('predict-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Analyzing...';

            const formData = {
                nitrogen: parseFloat(document.getElementById('nitrogen').value),
                phosphorus: parseFloat(document.getElementById('phosphorus').value),
                potassium: parseFloat(document.getElementById('potassium').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                humidity: parseFloat(document.getElementById('humidity').value),
                ph: parseFloat(document.getElementById('ph').value),
                rainfall: parseFloat(document.getElementById('rainfall').value),
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData),
                });

                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data);
            } catch (err) {
                alert('Failed to get prediction. Please try again.');
                console.error(err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🔍 Recommend Crop';
            }
        }

        function displayResults(data) {
            const panel = document.getElementById('results-panel');
            const placeholder = document.getElementById('placeholder-panel');

            placeholder.style.display = 'none';
            panel.classList.add('show');

            const top = data.top_crops[0];
            document.getElementById('primary-result').innerHTML = `
                <span class="crop-emoji">${top.emoji}</span>
                <div class="crop-name" style="color: ${top.color};">${top.name}</div>
                <div class="confidence">${top.confidence}% confidence</div>
                <p class="description">${top.description}</p>
                <div class="crop-details">
                    <div class="detail-chip">
                        <span class="label">📅 Season:</span>
                        <span class="value">${top.season}</span>
                    </div>
                    <div class="detail-chip">
                        <span class="label">💧 Water:</span>
                        <span class="value">${top.water}</span>
                    </div>
                </div>
            `;

            let altHTML = '';
            data.top_crops.slice(1).forEach(crop => {
                altHTML += `
                    <div class="alt-crop">
                        <div class="emoji">${crop.emoji}</div>
                        <div class="info">
                            <div class="name">${crop.name}</div>
                            <div class="alt-confidence">${crop.confidence}% confidence • ${crop.season}</div>
                            <div class="confidence-bar">
                                <div class="fill" style="width: ${crop.confidence}%;"></div>
                            </div>
                        </div>
                    </div>
                `;
            });
            document.getElementById('alt-crops').innerHTML = altHTML;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    """Serve the main page."""
    return render_template_string(
        HTML_TEMPLATE,
        model_name=best_model_name,
        accuracy=round(best_accuracy * 100, 2),
        crops=sorted(df["label"].unique()),
        n_samples=len(df),
        n_crops=df["label"].nunique(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        nitrogen = float(data["nitrogen"])
        phosphorus = float(data["phosphorus"])
        potassium = float(data["potassium"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        predicted_crop, top_crops = predict_crop(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
        )

        return jsonify({
            "predicted_crop": predicted_crop,
            "top_crops": top_crops,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================
# 6. RUN THE APP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  [*] Starting Crop Recommendation System Web App...")
    print(f"  [*] Open http://127.0.0.1:5000 in your browser")
    print("=" * 60 + "\n")

    app.run(debug=False, host="127.0.0.1", port=5000)
