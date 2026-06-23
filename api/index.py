import os
import re
import logging
from pathlib import Path

from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_MODEL_ID = "1ASIM1/sentimenmalaysia-distilbert"
HF_TOKEN = os.environ.get("HF_TOKEN")
DASHBOARD_HTML = Path(__file__).parent.parent / "dashboard.html"


def analyze_with_hf(text: str) -> dict:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable not set")
    client = InferenceClient(token=HF_TOKEN)
    results = client.text_classification(text, model=HF_MODEL_ID)
    if not results:
        raise RuntimeError("HF API returned empty result")
    results = results if isinstance(results, list) else [results]
    best = max(results, key=lambda x: x.score if hasattr(x, "score") else x["score"])
    label = best.label if hasattr(best, "label") else best["label"]
    score = best.score if hasattr(best, "score") else best["score"]
    return {"label": label, "score": score}

app = FastAPI(title="SentimenMalaysia API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ASPECT_KEYWORDS = {
    "Usability / Interface": [
        "interface", "ui", "ux", "intuitive", "layout", "design",
        "user-friendly", "usability", "navigation", "dashboard",
    ],
    "Performance / Speed": [
        "loading", "slow", "fast", "speed", "performance", "lag",
        "crash", "optimize", "fastest", "slowly", "delay",
    ],
    "Customer Support": [
        "support", "help", "service", "team", "response",
        "assistance", "helpline", "customer",
    ],
    "Economy / Finance": [
        "gdp", "economy", "economi", "market", "trade", "ringgit",
        "investment", "growth", "inflation", "budget", "fdi",
        "bursa", "stock", "rally", "surge", "profit", "revenue",
        "financial", "monetary",
    ],
    "Politics / Governance": [
        "government", "policy", "reform", "corruption", "political",
        "parliament", "minister", "pm", "anwar", "muhyiddin",
        "najib", "election", "democracy", "law",
    ],
    "Society / Health": [
        "flood", "health", "education", "crime", "accident",
        "patient", "hospital", "school", "covid", "disease",
        "safety", "disaster", "relief",
    ],
    "Technology / Innovation": [
        "tech", "digital", "innovation", "ai", "artificial intelligence",
        "platform", "software", "app", "system", "data", "online",
    ],
}

POSITIVE_WORDS = {
    "surge", "surges", "surged", "rally", "rallies", "strong", "stronger",
    "growth", "grow", "gain", "gains", "record", "boost", "boosted",
    "recover", "recovery", "rebound", "improve", "improved", "improvement",
    "success", "successful", "win", "wins", "winner", "breakthrough",
    "optimism", "optimistic", "confident", "confidence", "resilient",
    "thrive", "thriving", "prosper", "prosperous", "innovative",
    "efficient", "effective", "excellent", "outstanding", "positive",
    "good", "great", "best", "leading", "advanced",
}

NEGATIVE_WORDS = {
    "plunge", "plunged", "crash", "crashed", "fall", "falls", "fell",
    "decline", "declined", "drop", "dropped", "weak", "weaker", "weakness",
    "crisis", "scandal", "corruption", "graft", "layoff", "layoffs",
    "retrenchment", "death", "die", "died", "fatal", "fatality",
    "accident", "flood", "floods", "disaster", "destruction",
    "worse", "worst", "turmoil", "protest", "arrest", "illegal",
    "crime", "criminal", "threat", "fear", "anxiety", "anxious",
    "struggle", "struggling", "downturn", "slowdown", "unemployment",
    "pain", "painful", "bad", "poor", "terrible", "negative",
    "frustrating", "frustrated", "slow", "delay", "lag",
}

KNOWN_ENTITIES = {
    "BNM": "ORG", "Bank Negara": "ORG", "Bursa Malaysia": "ORG",
    "MACC": "ORG", "SPRM": "ORG", "DOSM": "ORG", "EPF": "ORG",
    "KWSP": "ORG", "MITI": "ORG", "MOF": "ORG", "Khazanah": "ORG",
    "Petronas": "ORG", "Maybank": "ORG", "CIMB": "ORG", "Celcom": "ORG",
    "Digi": "ORG", "Maxis": "ORG", "TM": "ORG", "Tenaga": "ORG",
    "Sime Darby": "ORG", "Genting": "ORG", "IOI": "ORG",
    "Anwar": "PER", "Anwar Ibrahim": "PER", "Najib": "PER",
    "Najib Razak": "PER", "Muhyiddin": "PER", "Muhyiddin Yassin": "PER",
    "Mahathir": "PER", "Mahathir Mohamad": "PER", "Zahid": "PER",
    "Hamzah": "PER", "Rafizi": "PER",
}


def extract_aspects(text: str) -> list:
    text_lower = text.lower()
    words = set(text_lower.split())
    aspects = []

    for aspect_name, keywords in ASPECT_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text_lower]
        if not matched:
            continue

        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            score = 50
        else:
            ratio = pos_count / total
            score = int(ratio * 100)

        if score >= 65:
            aspect_sentiment = "positive"
        elif score <= 35:
            aspect_sentiment = "negative"
        else:
            aspect_sentiment = "neutral"

        aspects.append({
            "aspect": aspect_name,
            "sentiment": aspect_sentiment,
            "score": score,
            "keywords": matched[:3],
        })

    return aspects


def extract_entities(text: str) -> list:
    entities = []

    for name, etype in KNOWN_ENTITIES.items():
        if name.lower() in text.lower():
            entities.append({"entity": name, "label": etype})

    metric_patterns = [
        (r"RM\d+(?:\.\d+)?\s*(?:bil|mil|billion|million|trillion)?", "METRIC"),
        (r"\d+(?:\.\d+)?\s*%", "METRIC"),
        (r"\d+(?:\.\d+)?\s*(?:pct|percent|percentage)", "METRIC"),
    ]
    for pattern, etype in metric_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({"entity": match.group().strip(), "label": etype})

    seen_texts = set()
    unique = []
    for e in entities:
        key = e["entity"].lower()
        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(e)

    return unique[:10]


def extract_linguistic_signals(text: str) -> dict:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+"
    )
    emojis = emoji_pattern.findall(text)

    pivot_words = ["but", "however", "although", "though", "despite",
                   "nevertheless", "nonetheless", "yet", "while"]
    found_pivots = [w for w in pivot_words if w in text.lower().split()]

    exclamation_count = text.count("!")
    all_caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    intensifiers = ["very", "extremely", "highly", "incredibly", "absolutely",
                    "totally", "completely", "remarkably", "significantly"]
    found_intensifiers = [w for w in intensifiers if w in text.lower().split()]

    intensity = "low"
    if exclamation_count > 0 or all_caps > 0 or len(found_intensifiers) > 1:
        intensity = "high"
    elif len(found_intensifiers) > 0:
        intensity = "medium"

    return {
        "emojis": emojis[:5],
        "pivots": found_pivots,
        "intensity": intensity,
        "exclamation_count": exclamation_count,
        "all_caps_count": all_caps,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    if not DASHBOARD_HTML.exists():
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok", "model": HF_MODEL_ID, "hf_token_set": HF_TOKEN is not None}


@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        result = analyze_with_hf(text)
    except RuntimeError as e:
        logger.error("HF API error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        logger.exception("HF API call failed unexpectedly")
        return JSONResponse(
            {"error": f"Inference failed: {str(e)}"}, status_code=502
        )

    label_str = result.get("label", "NEUTRAL")
    score = result.get("score", 0.0)

    if label_str.upper() == "LABEL_0" or label_str.lower() == "negative":
        sentiment = "negative"
    elif label_str.upper() == "LABEL_1" or label_str.lower() == "neutral":
        sentiment = "neutral"
    elif label_str.upper() == "LABEL_2" or label_str.lower() == "positive":
        sentiment = "positive"
    else:
        sentiment = "neutral"

    aspects = extract_aspects(text)
    entities = extract_entities(text)
    signals = extract_linguistic_signals(text)

    signals_array = []
    for emoji in signals["emojis"]:
        signals_array.append({"type": "emoji", "value": emoji, "label": "EMOJI"})
    for pivot in signals["pivots"]:
        signals_array.append({"type": "pivot", "value": pivot, "label": "PIVOT"})
    signals_array.append({
        "type": "intensity",
        "value": signals["intensity"].upper(),
        "label": "INTENSITY",
    })

    return {
        "sentiment": sentiment,
        "confidence": round(score, 4),
        "aspects": aspects,
        "entities": entities,
        "linguistic_signals": signals,
        "signals": signals_array,
    }
