import onnxruntime as ort
import numpy as np
import re

# Load ONNX fake-news model
session = ort.InferenceSession("backend/models/fake_news_onnx.onnx")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text

def predict_fake_news(text: str):
    cleaned = preprocess(text)
    input_data = np.array([[cleaned]])

    # Run ONNX model
    output = session.run(None, {"input": input_data})[0][0]
    fake_conf = float(output[0]) * 100
    genuine_conf = 100 - fake_conf

    is_fake = fake_conf >= 50

    return {
        "confidence": round(fake_conf, 2),
        "isFake": is_fake,
        "explanation": f"Predicted as {'FAKE' if is_fake else 'GENUINE'} with {fake_conf:.2f}% confidence.",
        "correctInfo": (
            "This content might be misleading—verify from official sources."
            if is_fake else
            "This content looks genuine but cross-checking is always recommended."
        ),
    }
