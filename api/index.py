from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from google import genai
import json
import io
import re

app = FastAPI(title="gTTS + Gemini API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_with_gemini(raw_text: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are a text preprocessing assistant for a Text-to-Speech (TTS) engine.
    Clean up the following raw text for spoken delivery:
    - Remove unwanted control codes, escape sequences (like \\bu, \\bur), and OCR artifacts.
    - Normalize line breaks, broken words, and extra spaces into smooth sentences.
    - Output ONLY the cleaned text without Markdown, explanations, or introductions.

    Raw Text:
    {raw_text}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()

@app.post("/api/tts")
async def generate_tts(request: Request):
    try:
        # Read raw request body bytes
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8", errors="ignore")

        # Replace physical line breaks inside text string before JSON parsing
        sanitized_body = body_str.replace('\r\n', '\\n').replace('\n', '\\n').replace('\t', ' ')
        data = json.loads(sanitized_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON Payload: {str(e)}")

    text = data.get("text", "")
    gemini_key = data.get("gemini_api_key", "")
    lang = data.get("lang", "en")
    slow = data.get("slow", False)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Process via Gemini if API Key is provided
        if gemini_key and gemini_key.strip():
            cleaned_text = clean_with_gemini(text, gemini_key.strip())
        else:
            cleaned_text = re.sub(r'\\[a-zA-Z]+', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Generate Audio
        tts = gTTS(text=cleaned_text, lang=lang, slow=slow)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Generation Failed: {str(e)}")
