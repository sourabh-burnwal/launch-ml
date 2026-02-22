# predict.py

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------
# Validate CLI argument
# ---------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py \"your text here\"")
    sys.exit(1)

input_text = " ".join(sys.argv[1:])


# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Load Model + Tokenizer
# ---------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


# ---------------------------
# Prediction
# ---------------------------
def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    label_map = model.config.id2label

    return {
        "text": text,
        "label": label_map[predicted_class.item()],
        "confidence": round(confidence.item(), 4)
    }


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("\nRunning inference...\n")
    result = predict(input_text)

    print("Prediction:")
    print(f"Text       : {result['text']}")
    print(f"Sentiment  : {result['label']}")
    print(f"Confidence : {result['confidence']}")