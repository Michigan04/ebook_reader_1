from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gtts import gTTS
import io

app = FastAPI(
    title="gTTS Serverless API",
    description="A simple Google Text-to-Speech API ready for Vercel deployment.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text content you want to convert into speech.",
        examples=["नमस्ते! यह एक साधारण पाठ से वाक् परिवर्तन का उदाहरण है।"]
    )
    lang: str = Field(
        default="hi",
        description="Two-letter IETF language code (e.g., 'hi' for Hindi, 'en' for English).",
        examples=["hi"]
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
        # Generate speech in-memory
        tts = gTTS(text=data.text, lang=data.lang, slow=data.slow)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except ValueError as e:
        # Typically raised if an unsupported language code is provided
        raise HTTPException(status_code=400, detail=f"Language error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")