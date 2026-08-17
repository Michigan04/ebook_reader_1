from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gtts import gTTS
from google import genai
from typing import Optional
import io
import re

app = FastAPI(
    title="gTTS + Gemini Preprocessor API",
    description="A gTTS API with optional Gemini text cleaning.",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def fallback_regex_clean(raw_text: str) -> str:
    """Basic regex fallback if no Gemini key is provided."""
    cleaned = re.sub(r'\\[a-zA-Z]+', '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_with_gemini(raw_text: str, api_key: str) -> str:
    """Uses Gemini 2.5 Flash to clean and prepare raw text for TTS."""
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are a text preprocessing assistant for a Text-to-Speech (TTS) engine.
        Clean up the following raw text for spoken delivery:
        - Remove unwanted control codes, escape sequences, line breaks within sentences, and unprintable characters.
        - Fix broken words, missing spaces, and formatting artifacts (e.g., PDF headers/footers).
        - Expand special symbols or abbreviations into natural words if needed.
        - Output ONLY the clean text without any introduction, explanations, or Markdown tags.

        Raw Text:
        {raw_text}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini API Error: {str(e)}")

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        description="The raw text you want to convert into speech.",
        examples=["To observe your mind in automatic mode..."]
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Optional: Your Gemini API key for LLM-powered text cleaning.",
        examples=["AIzaSy..."]
    )
    lang: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'hi').",
        examples=["en"]
    )
    slow: bool = Field(
        default=False,
        description="Set to True for slower speed.",
        examples=[False]
    )

@app.post(
    "/api/tts",
    summary="Convert Text to Audio",
    response_description="Returns MP3 binary audio stream"
)
def generate_tts(data: TTSRequest):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Step 1: Preprocess Text
        if data.gemini_api_key and data.gemini_api_key.strip():
            cleaned_text = clean_with_gemini(data.text, data.gemini_api_key.strip())
        else:
            cleaned_text = fallback_regex_clean(data.text)

        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text became empty after processing.")

        # Step 2: Generate gTTS Stream
        tts = gTTS(text=cleaned_text, lang=data.lang, slow=data.slow)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Language error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
