from typing import List, Union
from src.ai.models.bert_qa import BertQA
from src.ai.models.whisper import Whisper
from src.ai.models.vit import ViT
from src.ai.models.yolos import Yolos
from src.ai.models.bert_emb import BertEmbedding
import numpy as np
from PIL import Image


class AIEngine:
    def __init__(self):
        # Loading Models
        self.bert_emb = BertEmbedding()
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

    def embed(self, text: Union[str, List[str]]) -> np.array:
        if isinstance(text, str):
            text = [text]
        sentence_embeddings = []
        for text_ in text:
            assert isinstance(
                text_, str
            ), "Input must be a string or a list of strings."
            sentence_embeddings.append(self.bert_emb.embed(text_))
        return np.array(sentence_embeddings)

    def answer_from_context(self, question: str, context: str) -> str:
        return self.bert_qa.answer(question, context)
