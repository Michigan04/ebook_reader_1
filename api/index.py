from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gtts import gTTS
import io
import re

app = FastAPI(
    title="gTTS Serverless API",
    description="A simple Google Text-to-Speech API with preprocessing, ready for deployment.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Preprocessor function ---
def preprocess_text(raw_text: str) -> str:
    """
    Cleans raw text for safe use in TTS.
    - Removes invalid control sequences (\bu, \bur, stray backslashes).
    - Normalizes whitespace and line breaks.
    - Adds paragraph breaks for natural TTS pauses.
    """
    # Remove artifacts like \bu, \bur, etc.
    cleaned = re.sub(r'\\[a-zA-Z]+', '', raw_text)

    # Normalize multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Add pauses after sentences for smoother speech
    cleaned = cleaned.replace('. ', '.\n\n')

    return cleaned

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        description="The raw text you want to convert into speech. Preprocessing is handled internally.",
        examples=["To observe your mind in automatic mode, glance at the image below..."]
    )
    lang: str = Field(
        default="en",
        description="Two-letter IETF language code (e.g., 'en' for English, 'hi' for Hindi).",
        examples=["en"]
    )
    slow: bool = Field(
        default=False,
        description="Set to True to speak at a slower pace.",
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
        # Preprocess text internally
        cleaned_text = preprocess_text(data.text)

        # Generate speech in-memory
        tts = gTTS(text=cleaned_text, lang=data.lang, slow=data.slow)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Language error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
