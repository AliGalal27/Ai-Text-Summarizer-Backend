import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv  # <-- import dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Allow frontend to access the API from localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for frontend input
class PromptRequest(BaseModel):
    text: str
    length: str | None = None  # "short" | "medium" | "long"

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/generate")
async def generate_text(request: PromptRequest):
    # Map length to target size instructions
    length_map = {
        "short": "2-3 sentences (≈60–90 words)",
        "medium": "4-6 sentences (≈120–180 words)",
        "long": "8-12 sentences or concise bullets (≈250–350 words)",
    }
    target = length_map.get((request.length or "medium").lower(), length_map["medium"])

    prompt = f"""
You are an expert writing assistant. Summarize the user's text.

Constraints:
- Length: {target}
- Preserve key facts, numbers, names, and causal links.
- Prefer clear, neutral tone.
- Output MUST be valid Markdown. Use short paragraphs and bullet lists when helpful.
- Do not include extraneous lead-in like "Here is the summary"; return only the summary.

Text to summarize:
```
{request.text}
```
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    config = types.GenerateContentConfig()

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model="gemma-3n-e2b-it",  # change to another Gemini model if you want
        contents=contents,
        config=config,
    ):
        response_text += chunk.text or ""

    return {"result": response_text}
