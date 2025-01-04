from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from src.core.jarvis import JARVIS
import soundfile as sf
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
jarvis = JARVIS()

app.mount("/ui", StaticFiles(directory="src/ui"), name="ui")


@app.post("/document")
async def load_document(
    type: str, url: str = None, text: str = None, file: UploadFile = None
):
    """
    Load a document to the Jarvis context."""

    if type == "url":
        jarvis.load_context_from_wiki(url)
        return {"message": "Document loaded. You can now ask questions."}

    if type == "file":
        file_content = await file.read()
        file_str = file_content.decode("utf-8")
        jarvis.load_context_from_raw_text(file_str)
        return {"message": "Document loaded. You can now ask questions."}

    if type == "raw_text":
        raw_text = text
        jarvis.load_context_from_raw_text(raw_text)
        return {"message": "Document loaded. You can now ask questions."}

    return {"message": "No context was loaded"}


@app.get("/ask")
def ask(question: str):
    """
    Ask a question to Jarvis."""
    answer = jarvis.answer(question)
    return {"answer": answer}


@app.post("/transcript")
async def transcript(audio_file: UploadFile):
    """
    Transcribe an audio file."""
    audio_bytes = await audio_file.read()
    audio_data, _ = sf.read(BytesIO(audio_bytes))
    transcript = jarvis.transcript([audio_data])
    return {"text": transcript}
