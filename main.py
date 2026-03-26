import os
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

HIVE_API_KEY = os.environ.get("HIVE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HIVE_URL = "https://api.thehive.ai/api/v2/task/sync"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"


@app.get("/")
def root():
    return {"status": "POH backend running", "hive": bool(HIVE_API_KEY), "claude": bool(ANTHROPIC_API_KEY)}


@app.post("/detect")
async def detect(
    file: UploadFile = File(default=None),
    url: str = Form(default=None),
):
    if not HIVE_API_KEY:
        raise HTTPException(status_code=500, detail="HIVE_API_KEY not set on server.")
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL.")

    headers = {"Authorization": f"Token {HIVE_API_KEY}"}

    async with httpx.AsyncClient(timeout=60) as client:
        if file:
            contents = await file.read()
            files = {"media": (file.filename, contents, file.content_type)}
            hive_response = await client.post(HIVE_URL, headers=headers, files=files)
        else:
            hive_response = await client.post(HIVE_URL, headers=headers, data={"url": url})

    if hive_response.status_code != 200:
        raise HTTPException(
            status_code=hive_response.status_code,
            detail=f"Hive AI error: {hive_response.text}"
        )

    hive_data = hive_response.json()

    # Optionally enrich with Claude explanation
    if ANTHROPIC_API_KEY:
        try:
            explanation = await get_claude_explanation(hive_data)
            hive_data["explanation"] = explanation
        except Exception:
            pass  # Claude explanation is optional — don't fail if it errors

    return JSONResponse(content=hive_data)


async def get_claude_explanation(hive_data: dict) -> str:
    output = hive_data.get("status", [{}])[0].get("response", {}).get("output", [{}])[0]
    classes = output.get("classes", [])

    def get_score(name):
        return next((c["score"] for c in classes if c["class"] == name), 0)

    ai_score = get_score("ai_generated")
    deepfake_score = get_score("deepfake")
    real_score = get_score("real")
    not_ai_score = get_score("not_ai_generated")

    combined = max(ai_score, deepfake_score)
    verdict = "AI-generated" if combined > 0.5 else "authentic human"
    confidence = round(combined * 100 if combined > 0.5 else max(not_ai_score, real_score, 1 - combined) * 100)

    signals = []
    if ai_score > 0.05:
        signals.append(f"AI-generated probability: {round(ai_score * 100)}%")
    if deepfake_score > 0.05:
        signals.append(f"Deepfake probability: {round(deepfake_score * 100)}%")
    if real_score > 0.5:
        signals.append(f"Real human face detected: {round(real_score * 100)}%")
    for c in classes:
        if c["class"] not in ["ai_generated", "not_ai_generated", "deepfake", "real"] and c["score"] > 0.2:
            signals.append(f"{c['class'].replace('_', ' ')}: {round(c['score'] * 100)}%")

    signal_text = "\n".join(f"- {s}" for s in signals) if signals else "- No strong signals detected"

    prompt = f"""You are POH (Proof of Humanity), an AI content detection assistant.

Verdict: {verdict}
Confidence: {confidence}%
Detection signals:
{signal_text}

Write a clear 3-paragraph plain-English explanation for the user:
1. What the verdict means and why
2. The most important signals that led to this conclusion
3. What this means practically

Be specific but accessible. No bullet points. Under 200 words."""

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            CLAUDE_URL,
            headers={"Content-Type": "application/json", "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}
        )

    data = response.json()
    return data["content"][0]["text"]
