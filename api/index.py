import json
import io
import re
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from google import genai

app = FastAPI(
    title="gTTS + Gemini Robust Preprocessor API",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_raw_fields(raw_body: str) -> Tuple[str, Optional[str], str, bool]:
    """
    Extracts fields directly using regex so messy, unescaped strings 
    with raw newlines/control chars won't fail standard JSON parsers.
    """
    # Extract "text": "..." handling multiline contents safely
    text_match = re.search(r'"text"\s*:\s*"(.*?)"\s*,\s*"gemini_api_key"', raw_body, re.DOTALL)
    if not text_match:
        # Fallback regex if gemini_api_key isn't the immediate next key
        text_match = re.search(r'"text"\s*:\s*"(.*?)"\s*,\s*"[a-zA-Z_]+"', raw_body, re.DOTALL)
    
    raw_text = text_match.group(1) if text_match else ""

    # Extract optional Gemini API Key
    key_match = re.search(r'"gemini_api_key"\s*:\s*"(.*?)"', raw_body)
    gemini_key = key_match.group(1) if key_match and key_match.group(1).strip() else None

    # Extract language parameter
    lang_match = re.search(r'"lang"\s*:\s*"(.*?)"', raw_body)
    lang = lang_match.group(1) if lang_match else "en"

    # Extract slow parameter
    slow_match = re.search(r'"slow"\s*:\s*(true|false)', raw_body, re.IGNORECASE)
    slow = slow_match.group(1).lower() == "true" if slow_match else False

    # Fallback to standard json.loads if regex extraction fails completely
    if not raw_text:
        try:
            # Replace physical control characters with escaped equivalents
            clean_json = raw_body.replace('\r\n', '\\n').replace('\n', '\\n').replace('\t', '\\t')
            parsed = json.loads(clean_json)
            raw_text = parsed.get("text", "")
            gemini_key = parsed.get("gemini_api_key", gemini_key)
            lang = parsed.get("lang", lang)
            slow = parsed.get("slow", slow)
        except Exception:
            pass

    return raw_text, gemini_key, lang, slow


def fallback_clean(raw_text: str) -> str:
    """Basic fallback cleaner if Gemini API Key is missing."""
    # Strip escape sequences like \bu, \bur
    cleaned = re.sub(r'\\[a-zA-Z]+', '', raw_text)
    # Collapse whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def clean_with_gemini(raw_text: str, api_key: str) -> str:
    """Uses Gemini 2.5 Flash to transform noisy text into spoken-script format."""
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert audio script producer and text-to-speech preprocessor.
        Convert the following raw text into smooth, natural, spoken-word text.

        Instructions:
        1. Clean up broken words, OCR mistakes, unwanted escape sequences (e.g., \\bu, \\bur), and orphan symbols.
        2. Convert mathematical equations (e.g., '17 * 24') or special figures into conversational English (e.g., 'seventeen times twenty-four').
        3. Remove non-spoken layout markers like 'Figure 1'.
        4. Adjust punctuation so the TTS model reads with natural pauses and cadence.
        5. Output ONLY the finalized clean text. Do NOT wrap in markdown, quotes, or codeblocks. Do NOT add meta commentary.

        Raw Input Text:
        {raw_text}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini Processing Error: {str(e)}")


@app.post("/api/tts")
async def generate_tts(request: Request):
    # Step 1: Read raw request bytes and decode safely
    raw_bytes = await request.body()
    body_str = raw_bytes.decode("utf-8", errors="ignore")

    if not body_str.strip():
        raise HTTPException(status_code=400, detail="Empty payload received.")

    # Step 2: Extract text and parameters robustly
    text, gemini_key, lang, slow = extract_raw_fields(body_str)

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' is missing or empty.")

    try:
        # Step 3: Run LLM Preprocessing or Fallback
        if gemini_key:
            spoken_text = clean_with_gemini(text, gemini_key)
        else:
            spoken_text = fallback_clean(text)

        if not spoken_text.strip():
            raise HTTPException(status_code=400, detail="Text resulted in an empty string after preprocessing.")

        # Step 4: Feed the Gemini-processed output into gTTS
        tts = gTTS(text=spoken_text, lang=lang, slow=slow)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "X-Processed-Text-Length": str(len(spoken_text))
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")
