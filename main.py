import io
import os
import re
import tempfile
import Levenshtein
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import pipeline
import librosa
from g2p_en import G2p
import nltk

app = FastAPI(title="Whisper + G2P Pronunciation Scorer")

print("Loading ultra-lightweight Whisper model...")
# Using the automatic speech recognition pipeline optimized for CPU execution
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    device="cpu"
)

# --- ADD THESE TWO LINES ---
# This ensures the required dictionaries are downloaded silently on startup
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True) 
# ---------------------------

print("Loading G2P converter...")
g2p = G2p()
print("All systems ready!")

def clean_phonemes(text: str) -> list:
    """Converts a raw string into a list of clean, normalized phoneme tokens."""
    raw_phones = g2p(text)
    # Filter out spaces and punctuation markings
    return [p.lower() for p in raw_phones if p.strip() and p.isalnum()]

@app.post("/api/score")
async def score_pronunciation(
    file: UploadFile = File(...), 
    reference_text: str = Form(...)
):
    # Memory Guard: Read into local RAM buffer
    file_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    # 1. Standardize audio format using librosa
    try:
        audio_data, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding failed: {str(e)}")
        
    duration = len(audio_data) / 16000
    if duration < 30.0 or duration > 45.0:
        raise HTTPException(status_code=400, detail=f"Audio must be 30-45s. Got {duration:.1f}s.")

    # 2. Local Whisper Transcription
    try:
        # The pipeline accepts raw numpy arrays directly if sample_rate matches
        inference_result = asr_pipeline({"raw": audio_data, "sampling_rate": 16000})
        user_transcript = inference_result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper inference failed: {str(e)}")

    if not user_transcript:
        raise HTTPException(status_code=422, detail="Whisper could not detect any speech in the audio.")

    # 3. Grapheme-to-Phoneme Translation
    target_phonemes = clean_phonemes(reference_text)
    spoken_phonemes = clean_phonemes(user_transcript)

    if not target_phonemes:
        raise HTTPException(status_code=400, detail="Provided reference text contains no valid words.")

    # 4. Levenshtein Sequence Alignment & Scoring
    # Levenshtein distance works on character strings. We map each unique phoneme 
    # token to a unique temporary character so the edit distance matches phoneme-by-phoneme.
    unique_phonemes = list(set(target_phonemes + spoken_phonemes))
    phone_to_char = {phone: chr(i + 1000) for i, phone in enumerate(unique_phonemes)}
    
    target_str = "".join([phone_to_char[p] for p in target_phonemes])
    spoken_str = "".join([phone_to_char[p] for p in spoken_phonemes])
    
    edit_dist = Levenshtein.distance(target_str, spoken_str)
    
    # Calculate accuracy percentage against the expected reference sequence length
    max_len = len(target_phonemes)
    accuracy = max(0.0, (1.0 - (edit_dist / max_len)) * 100)

    # Memory cleanup
    del file_bytes
    del audio_data

    return {
        "status": "success",
        "pronunciation_score": round(accuracy, 2),
        "metrics": {
            "edit_distance_errors": edit_dist,
            "expected_phoneme_count": max_len,
            "detected_phoneme_count": len(spoken_phonemes)
        },
        "alignment": {
            "reference_text": reference_text,
            "whisper_transcription": user_transcript,
            "expected_phonemes": target_phonemes,
            "spoken_phonemes": spoken_phonemes
        },
        "duration_seconds": round(duration, 2)
    }