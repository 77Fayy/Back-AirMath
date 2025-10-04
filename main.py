# gamemath/backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os, re

load_dotenv()

# ===== OpenAI config =====
# You can hard-code here; .env fallback still works.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== FastAPI app =====
app = FastAPI(title="AirMath API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "see": "/docs"}

@app.get("/health")
def health():
    return {"ok": True}

# ===== Schemas =====
class CheckBody(BaseModel):
    image_b64: str  # raw base64 or full data URL
    expected: int

# ===== Helpers =====
ALLOWED = set("0123456789+-*/xX×÷()= ")

def strip_data_url_prefix(b64: str) -> str:
    """If b64 is a data URL, strip 'data:image/png;base64,' and return base64 only."""
    if b64 and b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64

def clean_ascii(s: str) -> str:
    """Keep only allowed characters; trim whitespace/backticks."""
    s = (s or "").strip().strip("`").replace(" ", "")
    return "".join(ch for ch in s if ch in ALLOWED)

# ===== Routes =====
@app.post("/check")
def check(body: CheckBody):
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        return {"ok": False, "error": "no_api_key"}

    # allow either full data URL or raw base64
    body.image_b64 = strip_data_url_prefix(body.image_b64)

    # Vision prompt: ask for just the final number
    prompt = (
        "You are an OCR for a single hand-drawn math answer image.\n"
        "Return ONLY the final numeric answer (digits only). Examples: 5 or 12.\n"
        "No spaces. No words. No LaTeX. No symbols. If the image shows `=12`, return `12`."
    )

    data_url = "data:image/png;base64," + body.image_b64

    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0,
            max_tokens=8,
        )
    except Exception as e:
        return {"ok": False, "error": f"openai_error: {e.__class__.__name__}"}

    raw = (r.choices[0].message.content or "").strip()
    ascii_only = clean_ascii(raw)

    # Accept '=12' OR '12'
    m = re.search(r"=([0-9]+)\s*$", ascii_only)
    if m:
        rhs = int(m.group(1))
    else:
        m2 = re.search(r"([0-9]+)\s*$", ascii_only)
        if not m2:
            return {"ok": False, "raw": ascii_only, "error": "no_number_found"}
        rhs = int(m2.group(1))

    correct = (rhs == int(body.expected))
    return {"ok": True, "raw": ascii_only, "number": rhs, "correct": correct}
