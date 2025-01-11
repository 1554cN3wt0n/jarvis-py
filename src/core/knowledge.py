from typing import List, Union
from src.core.documents.document import Document, DocumentManager
from src.ai.models.bert_emb import BertEmbedding
import numpy as np
from src.core.documents.utils import (
    split_paragraphs,
)


class Knowledge(DocumentManager):
    def __init__(self):
        super().__init__()
        # Load the LLM model to embed text
        # Embed Dimension: 384
        self.bert_emb = BertEmbedding()

    # Embed the text or the list of texts
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

    # Load the text from a file as a context
    def load_context(self, raw_text: str) -> None:
        # Load all paragraphs from a file
        paragraphs = split_paragraphs(raw_text)
        # Calculate the embeddings from all paragraphs
        embeddings = self.embed(paragraphs)
        # Add the document to the document manager
        self.add_document(Document(paragraphs, embeddings, "raw_text"))
