from src.ai.models.bert_qa import BertQA
from src.ai.models.whisper import Whisper
from src.ai.models.vit import ViT
from src.ai.models.yolos import Yolos
from src.core.knowledge import Knowledge
import numpy as np
from PIL import Image


def topk(arr, k, axis=0):
    idx = np.argpartition(arr, -k, axis=axis)[-k:]
    return np.take_along_axis(arr, idx, axis=axis)


class JARVIS(Knowledge):
    def __init__(self):
        super().__init__()

        # Loading Models
        self.bert_qa = BertQA()
        self.whisper = Whisper()
        self.vit = ViT()
        self.yolos = Yolos()

    def transcript(self, audio_data: np.ndarray) -> str:
        return self.whisper.transcript(audio_data)

    def classify_image(self, image: Image) -> str:
        return self.vit.classify(image)

    def detect_objects(self, image: np.ndarray) -> str:
        return self.yolos.detect_objects(image)

    def answer(self, question: str) -> str:
        # Embed the question
        emb_question = self.embed(question)

        # Get context and context embeddings based on the question
        document = self.get_document(emb_question)

        # Find all possible paragraphs that are related to the question
        chunk = document.get_chunk(emb_question)

        # Find the answer to the question in the context
        return self.bert_qa.answer(question, chunk.text)
