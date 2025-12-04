import os
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import emoji
import language_tool_python
import nltk
from fastapi.middleware.cors import CORSMiddleware

nltk.download("punkt")

# -----------------------------
# ORIGINAL CODE (UNCHANGED)
# -----------------------------
MODEL_DIR = "onnx-ai-detector"
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}. Place model.onnx in {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str


def onnx_predict(text: str) -> float:
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    probs = softmax(logits)[0][1]
    return float(probs)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


# -----------------------------
# NEW ADVANCED AI DETECTOR ADD-ONS
# -----------------------------
tool = language_tool_python.LanguageTool('en-US')


def detect_structure(text):
    score = 0
    if "##" in text:
        score += 0.15
    if "-" in text or "*" in text:
        score += 0.15
    if re.search(r"\d+\.", text):
        score += 0.15
    if len(text.split("\n")) >= 4:
        score += 0.15
    return min(score, 1.0)


def detect_tone(text):
    ai_phrases = ["in conclusion", "overall", "furthermore", "moreover", "additionally", "here are some", "importantly"]
    count = sum(text.lower().count(p) for p in ai_phrases)
    return min(count * 0.1, 1.0)


def detect_grammar(text):
    errors = tool.check(text)
    error_rate = len(errors) / max(1, len(text.split()))
    if error_rate == 0:
        return 1.0
    if error_rate < 0.01:
        return 0.7
    return 0.2


def detect_emoji_pattern(text):
    ems = emoji.emoji_list(text)
    count = len(ems)
    if count == 0:
        return 0.3
    if all(e['emoji'] in "🎯🚀🔍🔥✨" for e in ems):
        return 0.7
    return 0.2


def detect_burstiness(text):
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.5
    avg = np.mean(lengths)
    std = np.std(lengths)
    burstiness = std / (avg + 1e-9)
    if burstiness < 0.3:
        return 0.7
    if burstiness < 0.15:
        return 0.9
    return 0.3


def detect_creativity(text):
    edgy_words = ["lol", "damn", "wtf", "lmao", "bro", "crazy"]
    if any(w in text.lower() for w in edgy_words):
        return 0.1
    return 0.6


def combined_ai_score(text):
    onnx_prob = onnx_predict(text)
    structure = detect_structure(text)
    tone = detect_tone(text)
    grammar = detect_grammar(text)
    burstiness = detect_burstiness(text)
    emoji_score = detect_emoji_pattern(text)
    creativity = detect_creativity(text)

    final = (
        onnx_prob * 0.55 +
        structure * 0.10 +
        tone * 0.08 +
        grammar * 0.10 +
        burstiness * 0.07 +
        emoji_score * 0.05 +
        creativity * 0.05
    )

    return final, onnx_prob, {
        "structure": structure,
        "tone": tone,
        "grammar_cleanliness": grammar,
        "burstiness": burstiness,
        "emoji_patterns": emoji_score,
        "creativity": creativity
    }


# -----------------------------
# NEW HELPER: Sentence-level highlighting (with markup/code detection)
# -----------------------------
def combined_ai_score_with_line_analysis(text):
    final, onnx_prob, heuristics = combined_ai_score(text)

    # Split into sentences/lines
    potential_lines = re.split(r'(?<=[.!?])\s+|\n', text)
    lines = [line.strip() for line in potential_lines if line.strip()]

    line_level_analysis = []

    # Only analyze lines if more than one line
    if len(lines) > 1:
        for line in lines:
            # Markup/code detection
            code_or_markup_pattern = r'^\s*(```|<[^>]+>|#include|def |class |for |while |if |else |return )'
            if re.search(code_or_markup_pattern, line):
                line_score = 1.0
                label = "highly AI"
            else:
                line_score, _, _ = combined_ai_score(line)
                if line_score > 0.7:
                    label = "highly AI"
                elif line_score > 0.5:
                    label = "moderately AI"
                elif line_score > 0.3:
                    label = "likely AI"
                else:
                    label = "low AI / Human-like"

            line_level_analysis.append({
                "line": line,
                "ai_score": line_score,
                "ai_label": label
            })

    return final, onnx_prob, heuristics, {
        "overall_ai_score": final,
        "overall_ai_label": "AI" if final >= 0.5 else "Human",
        "line_level_analysis": line_level_analysis
    }


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/predict")
async def predict(payload: TextPayload):
    text = payload.text

    final_score, onnx_prob, heuristics, detailed_lines = combined_ai_score_with_line_analysis(text)
    
    return {
        "label": "AI" if final_score >= 0.5 else "Human",
        "final_confidence": final_score,
        "onnx_model_confidence": onnx_prob,
        "heuristics": heuristics,
        "line_level_analysis": detailed_lines["line_level_analysis"]
    }
