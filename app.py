"""
AI Car Price Advisor
=====================
Multimodal AI application combining:
  1. Computer Vision  — CLIP (zero-shot) + EfficientNet-B0 (comparison)
  2. Machine Learning — Gradient Boosting price prediction
  3. NLP              — GPT-4o-mini explanation (Prompt V1 vs V2)

Modes:
  - Image only   → body type classification shown immediately
  - Details only → price estimated using manually selected body type
  - Both         → body type detected from image + price predicted

Environment variables:
  OPENAI_API_KEY  — for NLP explanation generation
"""

import json
import os
import pickle
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import CLIPModel, CLIPProcessor
import requests

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
ML_PATH      = BASE_DIR / "car_price_model.pkl"
CV_PATH      = BASE_DIR / "vehicle_classifier.pt"
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
CURRENT_YEAR = 2024

MAKES = [
    "Alfa Romeo", "Aston Martin", "Audi", "BMW", "BYD", "Dongfeng",
    "Ford", "Honda", "Hyundai", "Kia", "Maserati", "Maybach",
    "Mercedes-Benz", "Mitsubishi", "Opel", "Porsche", "Renault",
    "Rolls-Royce", "SsangYong", "Suzuki", "Tesla", "Volkswagen",
    "Volvo", "smart",
]
BODY_TYPES        = ["SUV", "Sedan", "Hatchback", "Coupe"]
FUEL_OPTIONS      = ["Petrol", "Diesel", "Electric", "Other"]
TRANSMISSION_OPTIONS = ["Automatic", "Manual", "Semi-automatic"]

# ─── CarFeatureEngineer (must be defined here so pickle can find it) ───────────
TARGET_ENCODING_SMOOTHING = 20.0

def _extract_first_number(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"([-+]?\d+(?:[\.,]\d+)?)", expand=False)
    return pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")

_TRAIN_FUEL_MAPPING = {
    "Super 95": "Petrol", "Super Plus E10 98": "Petrol",
    "Super E10 95": "Petrol", "Regular/Benzine 91": "Petrol",
    "Regular/Benzine E10 91": "Petrol", "Super Plus 98": "Petrol",
    "Diesel": "Diesel", "Biodiesel": "Diesel",
    "Electricity": "Electric",
    "Liquid petroleum gas (LPG)": "Other",
    "Domestic gas H": "Other", "Vegetable oil": "Other",
}

class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Mirrors the class in train_ml.py exactly so pickle can deserialize the saved model."""

    def __init__(self, current_year=CURRENT_YEAR, smoothing=TARGET_ENCODING_SMOOTHING):
        self.current_year = current_year
        self.smoothing    = smoothing

    def fit(self, X, y=None):
        if y is None:
            self.global_mean_      = 50_000.0
            self.model_target_map_ = {}
        else:
            y_s = pd.Series(y, index=X.index, dtype=float)
            self.global_mean_ = float(y_s.mean())
            stats = (
                pd.DataFrame({"model": X["model"].fillna("Unknown"), "price": y_s})
                .groupby("model")["price"].agg(["mean", "count"])
            )
            smoothed = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) \
                       / (stats["count"] + self.smoothing)
            self.model_target_map_ = smoothed.to_dict()
        years = pd.to_datetime(X["registration_date"], errors="coerce").dt.year
        fallback = years.dropna().median()
        self.fallback_year_ = int(fallback) if not pd.isna(fallback) else self.current_year - 8
        return self

    def transform(self, X):
        X = X.copy()
        for col in ["make", "model", "body_type", "transmission", "primary_fuel"]:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
        reg_year = pd.to_datetime(X["registration_date"], errors="coerce").dt.year \
                     .fillna(getattr(self, "fallback_year_", self.current_year - 8)).astype(int)
        X["car_age"]          = (self.current_year - reg_year).clip(lower=1)
        X["mileage_km"]       = _extract_first_number(X["mileage_km"]).fillna(0)
        X["mileage_per_year"] = X["mileage_km"] / X["car_age"].replace(0, 1)
        X["log_mileage_km"]   = np.log1p(X["mileage_km"].clip(lower=0))
        X["age_x_mileage"]    = X["car_age"] * X["mileage_km"]
        X["fuel_category"]    = X["primary_fuel"].map(_TRAIN_FUEL_MAPPING).fillna("Other")
        X["model_price_encoded"] = (
            X["model"].map(getattr(self, "model_target_map_", {}))
            .fillna(getattr(self, "global_mean_", 50_000.0))
        )
        return X


CLIP_PROMPTS = {
    "SUV":       "a photo of an SUV or crossover with high ground clearance and large body",
    "Sedan":     "a photo of a sedan with four doors and a separate trunk compartment",
    "Hatchback": "a photo of a compact hatchback with a rear liftgate door",
    "Coupe":     "a photo of a coupe sports car with two doors and a sloped roofline",
}

# ─── LOAD ML MODEL ─────────────────────────────────────────────────────────────
ml_artifact = None
ml_metadata = {}

def load_ml_model():
    global ml_artifact, ml_metadata
    if ML_PATH.exists():
        with ML_PATH.open("rb") as f:
            ml_artifact = pickle.load(f)
        ml_metadata = ml_artifact.get("metadata", {})
        print(f"ML model loaded: {ml_metadata.get('best_model')} | R²={ml_metadata.get('r2', 0):.3f}")
    else:
        print("WARNING: car_price_model.pkl not found. Run train_ml.py first.")

load_ml_model()

# ─── LOAD CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP loaded.")

# ─── LOAD EFFICIENTNET ─────────────────────────────────────────────────────────
eff_model   = None
eff_classes = BODY_TYPES
cv_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_efficientnet():
    global eff_model, eff_classes
    if CV_PATH.exists():
        ckpt = torch.load(CV_PATH, map_location="cpu")
        eff_classes = ckpt.get("class_names", BODY_TYPES)
        m = efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, len(eff_classes)),
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        eff_model = m
        print(f"EfficientNet-B0 loaded (val_acc={ckpt.get('val_acc', 0):.3f})")
    else:
        print("WARNING: vehicle_classifier.pt not found.")

load_efficientnet()

# ─── CV: CLASSIFY IMAGE ────────────────────────────────────────────────────────
def classify_image(pil: Image.Image) -> tuple[str, str, str]:
    """Returns (clip_pred, eff_pred, comparison_text)."""
    pil = pil.convert("RGB")

    # CLIP
    names  = list(CLIP_PROMPTS.keys())
    texts  = list(CLIP_PROMPTS.values())
    inputs = clip_processor(text=texts, images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_clip = clip_model(**inputs).logits_per_image.softmax(dim=1).squeeze().tolist()
    clip_probs = {names[i]: round(probs_clip[i], 4) for i in range(len(names))}
    clip_pred  = max(clip_probs, key=clip_probs.get)

    # EfficientNet
    eff_lines = "Model not loaded."
    eff_pred  = clip_pred  # fallback to CLIP if EfficientNet not available
    if eff_model is not None:
        tensor = cv_transform(pil).unsqueeze(0)
        with torch.no_grad():
            probs_eff = torch.softmax(eff_model(tensor), dim=1).squeeze().tolist()
        eff_probs = {eff_classes[i]: round(probs_eff[i], 4) for i in range(len(eff_classes))}
        eff_pred  = max(eff_probs, key=eff_probs.get)
        eff_lines = "\n".join(
            f"  {c}: {p:.1%}" for c, p in sorted(eff_probs.items(), key=lambda x: -x[1])
        )

    clip_lines = "\n".join(
        f"  {c}: {p:.1%}" for c, p in sorted(clip_probs.items(), key=lambda x: -x[1])
    )
    comparison = (
        f"CLIP (zero-shot)  →  {clip_pred}\n{clip_lines}\n\n"
        f"EfficientNet-B0 (trained)  →  {eff_pred}\n{eff_lines}"
    )
    return clip_pred, eff_pred, comparison


# ─── CV: AUTO-FILL ON IMAGE UPLOAD ────────────────────────────────────────────
def on_image_upload(image):
    """Runs automatically when user uploads an image."""
    if image is None:
        return gr.update(value="SUV"), ""
    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image
    clip_pred, eff_pred, comparison = classify_image(pil)
    return gr.update(value=clip_pred), comparison


# ─── LIVE FEATURE UPDATE ───────────────────────────────────────────────────────
def update_features_live(year, mileage_km):
    """Updates engineered features in real-time as sliders change."""
    car_age = max(CURRENT_YEAR - int(year), 1)
    mileage_per_year = int(mileage_km) / car_age
    return (
        f"car_age          = {CURRENT_YEAR} − {int(year)} = {car_age} Jahr(e)\n"
        f"mileage_per_year = {int(mileage_km):,} ÷ {car_age} = {mileage_per_year:,.0f} km/Jahr"
    )


# ─── ML: PRICE PREDICTION ──────────────────────────────────────────────────────
def predict_price(make, model_name, year, mileage_km, fuel, transmission, body_type):
    car_age          = max(CURRENT_YEAR - int(year), 1)
    mileage_per_year = float(mileage_km) / car_age

    if ml_artifact is None:
        return 25000.0, car_age, mileage_per_year

    # Try new pipeline format (optimized script — CarFeatureEngineer inside pipeline)
    fuel_raw_map = {"Petrol": "Super 95", "Diesel": "Diesel",
                    "Electric": "Electricity", "Other": "Liquid petroleum gas (LPG)"}
    row_new = pd.DataFrame([{
        "make": make,
        "model": model_name,
        "mileage_km": float(mileage_km),
        "registration_date": f"{int(year)}-06-01",
        "body_type": body_type,
        "transmission": transmission,
        "primary_fuel": fuel_raw_map.get(fuel, "Super 95"),
    }])

    # Fallback: old pipeline format (pre-engineered features passed manually)
    model_target_map = ml_metadata.get("model_target_map", {})
    global_mean      = ml_metadata.get("global_mean_price", 56905.0)
    row_old = pd.DataFrame([{
        "make": make, "mileage_km": float(mileage_km),
        "car_age": car_age, "mileage_per_year": mileage_per_year,
        "body_type": body_type, "transmission": transmission,
        "fuel_category": fuel,
        "model_price_encoded": model_target_map.get(model_name, global_mean),
    }])

    for row in [row_new, row_old]:
        try:
            price = max(float(ml_artifact["pipeline"].predict(row)[0]), 500.0)
            return price, car_age, mileage_per_year
        except Exception:
            continue

    return 25000.0, car_age, mileage_per_year


# ─── NLP: EXPLANATION ──────────────────────────────────────────────────────────
PROMPT_V1 = """You are a car valuation expert. Write 3 sentences explaining this price estimate.

Vehicle: {year} {make} {model} ({body_type})
Mileage: {mileage_km:,} km | Car age: {car_age} years | km/year: {mileage_per_year:,.0f}
Fuel: {fuel} | Transmission: {transmission}
Estimated market value: EUR {price:,.0f}

Mention the body type and the key pricing factors."""

PROMPT_V2 = """You are an expert automotive market analyst. Provide a 3-4 sentence valuation commentary.

Vehicle: {year} {make} {model} ({body_type}, detected from image by CLIP)
Mileage: {mileage_km:,} km over {car_age} years = {mileage_per_year:,.0f} km/year
Fuel: {fuel} | Gearbox: {transmission}
Estimated market value: EUR {price:,.0f}

Cover: (1) how body type influences price, (2) impact of age and mileage, (3) market segment."""

def generate_explanation(make, model_name, year, mileage_km, fuel, transmission,
                         body_type, car_age, mileage_per_year, price, use_v2):
    template = PROMPT_V2 if use_v2 else PROMPT_V1
    prompt = template.format(
        make=make, model=model_name, year=year, body_type=body_type,
        mileage_km=int(mileage_km), car_age=car_age,
        mileage_per_year=mileage_per_year, fuel=fuel,
        transmission=transmission, price=price,
    )
    if not OPENAI_KEY:
        return (
            f"The vehicle was identified as a **{body_type}**. "
            f"A {year} {make} {model_name} with {int(mileage_km):,} km "
            f"({mileage_per_year:,.0f} km/year over {car_age} years) "
            f"has an estimated market value of **EUR {price:,.0f}**."
        )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "max_tokens": 200, "temperature": 0.2,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            f"The vehicle was identified as a **{body_type}**. "
            f"Estimated market value: **EUR {price:,.0f}** "
            f"({year} {make} {model_name}, {int(mileage_km):,} km)."
        )


# ─── MAIN ANALYZE FUNCTION ─────────────────────────────────────────────────────
def analyze(image, body_type_manual, make, model_name, year, mileage_km,
            fuel, transmission, use_v2):
    """
    Three modes:
      - Image only   → body type classification shown, no price
      - Details only → price with manually selected body type
      - Both         → body type from CLIP + price
    """
    has_image   = image is not None
    has_details = bool(make and model_name)

    features_text = ""
    price_str     = ""
    explanation   = ""

    body_type = body_type_manual

    if has_image:
        pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image
        clip_body, eff_body, _ = classify_image(pil)
        body_type = clip_body  # CLIP is primary

    if has_details:
        try:
            car_age = max(CURRENT_YEAR - int(year), 1)
            mileage_per_year = float(mileage_km) / car_age
            features_text = (
                f"car_age          = {CURRENT_YEAR} − {int(year)} = {car_age} Jahr(e)\n"
                f"mileage_per_year = {int(mileage_km):,} ÷ {car_age} = {mileage_per_year:,.0f} km/Jahr"
            )
            price, car_age, mileage_per_year = predict_price(
                make, model_name, year, mileage_km, fuel, transmission, body_type
            )
            price_str = f"EUR {price:,.0f}"
            explanation = generate_explanation(
                make, model_name, year, mileage_km, fuel, transmission,
                body_type, car_age, mileage_per_year, price, use_v2,
            )
        except Exception as e:
            features_text = f"Error: {e}"
            price_str     = f"Error: {e}"
            explanation   = f"Error: {e}"

    return features_text, price_str, explanation


# ─── THEME ─────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#171717",
    body_background_fill_dark="#171717",
    block_background_fill="#262626",
    block_background_fill_dark="#262626",
    block_border_color="#404040",
    block_border_color_dark="#404040",
    block_label_background_fill="#262626",
    block_label_background_fill_dark="#262626",
    block_label_text_color="#a3a3a3",
    block_label_text_color_dark="#a3a3a3",
    input_background_fill="#1a1a1a",
    input_background_fill_dark="#1a1a1a",
    input_border_color="#404040",
    input_border_color_dark="#404040",
    body_text_color="#e5e5e5",
    body_text_color_dark="#e5e5e5",
    button_primary_background_fill="#f97316",
    button_primary_background_fill_dark="#f97316",
    button_primary_background_fill_hover="#ea6f10",
    button_primary_background_fill_hover_dark="#ea6f10",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#404040",
    button_secondary_background_fill_dark="#404040",
    button_secondary_text_color="#e5e5e5",
    button_secondary_text_color_dark="#e5e5e5",
    slider_color="#f97316",
    slider_color_dark="#f97316",
    checkbox_background_color="#262626",
    checkbox_background_color_dark="#262626",
    checkbox_background_color_selected="#f97316",
    checkbox_background_color_selected_dark="#f97316",
    checkbox_border_color="#606060",
    checkbox_border_color_dark="#606060",
    shadow_drop="none",
    shadow_drop_lg="none",
)

# ─── GRADIO UI ─────────────────────────────────────────────────────────────────
EXAMPLES_DIR = BASE_DIR / "examples"
examples = [[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.jpg"))[:4]] if EXAMPLES_DIR.exists() else []

with gr.Blocks(title="AI Car Price Advisor", theme=theme) as demo:

    gr.Markdown("""
    # AI Car Price Advisor
    Multimodal AI — Computer Vision + Machine Learning + NLP

    Upload a photo to detect the body type. Enter vehicle details to estimate the market price.
    """)

    with gr.Row():

        # ── LEFT: Inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(label="Vehicle Photo (optional)", type="numpy")
            if examples:
                gr.Examples(examples=examples, inputs=image_input, label="Example Images")

            body_type_input = gr.Dropdown(
                choices=BODY_TYPES, value="SUV",
                label="Body Type (auto-filled from image)"
            )
            make_input = gr.Dropdown(
                choices=MAKES, value="BMW",
                label="Make (Marke)"
            )
            model_input = gr.Textbox(
                value="X5",
                label="Model (Modell)",
                placeholder="e.g. X5, A4, 911, Golf"
            )
            year_input = gr.Slider(
                minimum=2000, maximum=2024, value=2019, step=1,
                label="Registration Year (Erstzulassung)"
            )
            mileage_input = gr.Slider(
                minimum=0, maximum=300000, value=85000, step=1000,
                label="Mileage in km (Kilometerstand)"
            )
            fuel_input = gr.Dropdown(
                choices=FUEL_OPTIONS, value="Petrol",
                label="Fuel Type (Treibstoff)"
            )
            transmission_input = gr.Dropdown(
                choices=TRANSMISSION_OPTIONS, value="Automatic",
                label="Transmission (Getriebe)"
            )
            prompt_toggle = gr.Checkbox(
                label="Use extended explanation prompt (Prompt V2)",
                value=False
            )
            analyze_btn = gr.Button("Estimate Price", variant="primary", size="lg")

        # ── RIGHT: Results ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            cv_out = gr.Textbox(
                label="Body Type Classification — CLIP vs EfficientNet-B0",
                lines=10, interactive=False,
                placeholder="Upload an image to see body type classification..."
            )
            gr.Markdown("#### Feature Engineering (live)")
            features_out = gr.Textbox(
                label="Computed from your slider inputs",
                lines=2, interactive=False,
                value=f"car_age          = {CURRENT_YEAR} − 2019 = 5 Jahr(e)\nmileage_per_year = 85,000 ÷ 5 = 17,000 km/Jahr",
            )
            price_clip_out = gr.Textbox(
                label="Estimated Market Price",
                lines=1, interactive=False,
                placeholder="Click 'Estimate Price'..."
            )
            explanation_out = gr.Textbox(
                label="Explanation (NLP — GPT-4o-mini)",
                lines=6, interactive=False,
                placeholder="Click 'Estimate Price'..."
            )

    # ── Engineered features update live as sliders move ───────────────────────
    year_input.change(
        fn=update_features_live,
        inputs=[year_input, mileage_input],
        outputs=[features_out],
    )
    mileage_input.change(
        fn=update_features_live,
        inputs=[year_input, mileage_input],
        outputs=[features_out],
    )

    # ── Auto-fill body type when image is uploaded ────────────────────────────
    image_input.change(
        fn=on_image_upload,
        inputs=[image_input],
        outputs=[body_type_input, cv_out],
    )

    # ── Estimate price on button click ────────────────────────────────────────
    analyze_btn.click(
        fn=analyze,
        inputs=[
            image_input, body_type_input, make_input, model_input,
            year_input, mileage_input, fuel_input, transmission_input,
            prompt_toggle,
        ],
        outputs=[features_out, price_clip_out, explanation_out],
    )

if __name__ == "__main__":
    demo.launch()
