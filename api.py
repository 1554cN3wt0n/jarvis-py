from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.core.jarvis import JARVIS
import soundfile as sf
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="JARVIS",
    description="APIs for JARVIS",
    summary="",
    version="0.0.1",
)
jarvis = JARVIS()

app.mount("/ui", StaticFiles(directory="src/ui"), name="ui")


@app.get("/documents", tags=["Documents"])
def get_all_documents():
    return jarvis.get_all_documents_list()


@app.post("/documents/snapshot/save", tags=["Documents"])
async def save_documents_snapshot():
    """
    Save current context history in an snapshot."""

    jarvis.save_snapshot("tmp/memory.pkl")
    return {"message": "Snapshot saved successfully"}


@app.post("/documents/snapshot/load", tags=["Documents"])
async def load_documents_snapshot():
    """
    Load snapshot into memory."""

    jarvis.load_snapshot("tmp/memory.pkl")
    return {"message": "Snapshot loaded successfully"}


@app.post("/document", tags=["Documents"])
async def load_document(file: UploadFile = None):
    """
    Load a document to the Jarvis context."""

    file_content = await file.read()
    file_str = file_content.decode("utf-8")
    jarvis.load_context(file_str, file.filename)
    return {"message": "Document loaded. You can now ask questions."}


@app.delete("/cluster/{cluster_id}/document/{document_id}", tags=["Documents"])
def delete_document(cluster_id: int, document_id: int):
    ok = jarvis.delete_document(cluster_id, document_id)
    if ok:
        return {"status": "OK"}
    return {"status": "Failed"}


@app.post("/image/classify", tags=["Images"])
async def classify_image(image_file: UploadFile):
    """
    Transcribe an audio file."""
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    classification = jarvis.classify_image(image)
    return {"label": classification}


@app.post("/image/detect", tags=["Images"])
async def detect_objects(image_file: UploadFile):
    """
    Transcribe an audio file."""
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    result = jarvis.detect_objects(image)
    return {"objects": result}


@app.get("/jarvis/ask", tags=["Jarvis"])
def ask(question: str, context: str = None):
    """
    Ask a question to Jarvis based on the uploaded contexts (documents) or an specific context.
    """
    if context is None:
        return jarvis.answer(question)
    ans = jarvis.answer_from_context(question, context)
    return {"answer": ans}


@app.post("/audio/transcript", tags=["Audio"])
async def transcript(audio_file: UploadFile):
    """
    Transcribe an audio file."""
    audio_bytes = await audio_file.read()
    audio_data, _ = sf.read(BytesIO(audio_bytes))
    transcript = jarvis.transcript([audio_data])
    return {"text": transcript}


@app.get("/audio/speak", tags=["Audio"])
async def text_to_speech(text: str):
    jarvis.speak(text)
    return FileResponse("tmp/spoken.wav", media_type="audio/wav")
