import os
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow your Netlify frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your Netlify URL for extra security e.g. https://provehumanity.app
    allow_methods=["POST"],
    allow_headers=["*"],
)

HIVE_API_KEY = os.environ.get("HIVE_API_KEY", "")
HIVE_URL = "https://api.thehive.ai/api/v2/task/sync"


@app.get("/")
def root():
    return {"status": "POH backend is running"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(default=None),
    url: str = Form(default=None),
):
    if not HIVE_API_KEY:
        raise HTTPException(status_code=500, detail="HIVE_API_KEY not configured on server.")

    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL.")

    headers = {"Authorization": f"Token {HIVE_API_KEY}"}

    async with httpx.AsyncClient(timeout=30) as client:
        if file:
            contents = await file.read()
            files = {"media": (file.filename, contents, file.content_type)}
            response = await client.post(HIVE_URL, headers=headers, files=files)
        else:
            response = await client.post(HIVE_URL, headers=headers, data={"url": url})

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Hive AI error: {response.text}"
        )

    return JSONResponse(content=response.json())
