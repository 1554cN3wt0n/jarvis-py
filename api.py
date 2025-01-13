from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from src.core.jarvis import JARVIS
import soundfile as sf
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
jarvis = JARVIS()

app.mount("/ui", StaticFiles(directory="src/ui"), name="ui")


@app.get("/documents")
def get_all_documents():
    return jarvis.get_all_documents_list()


@app.post("/documents/snapshot/save")
async def save_documents_snapshot():
    """
    Save current context history in an snapshot."""

    jarvis.save_snapshot("snapshots/memory.pkl")
    return {"message": "Snapshot saved successfully"}


@app.post("/documents/snapshot/load")
async def load_documents_snapshot():
    """
    Load snapshot into memory."""

    jarvis.load_snapshot("snapshots/memory.pkl")
    return {"message": "Snapshot loaded successfully"}


@app.post("/document")
async def load_document(file: UploadFile = None):
    """
    Load a document to the Jarvis context."""

    file_content = await file.read()
    file_str = file_content.decode("utf-8")
    jarvis.load_context(file_str, file.filename)
    return {"message": "Document loaded. You can now ask questions."}


@app.delete("/cluster/{cluster_id}/document/{document_id}")
def delete_document(cluster_id: int, document_id: int):
    ok = jarvis.delete_document(cluster_id, document_id)
    if ok:
        return {"status": "OK"}
    return {"status": "Failed"}


@app.post("/image")
async def classify_image(image_file: UploadFile):
    """
    Transcribe an audio file."""
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    classification = jarvis.classify_image(image)
    return {"label": classification}


@app.post("/detect")
async def detect_objects(image_file: UploadFile):
    """
    Transcribe an audio file."""
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    result = jarvis.detect_objects(image)
    return {"objects": result}


@app.get("/ask")
def ask(question: str):
    """
    Ask a question to Jarvis based on the uploaded contexts (documents)."""
    return jarvis.answer(question)


@app.post("/transcript")
async def transcript(audio_file: UploadFile):
    """
    Transcribe an audio file."""
    audio_bytes = await audio_file.read()
    audio_data, _ = sf.read(BytesIO(audio_bytes))
    transcript = jarvis.transcript([audio_data])
    return {"text": transcript}
