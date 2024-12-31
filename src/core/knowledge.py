from typing import List, Union
from src.core.document import Document, DocumentManager
from src.llm.bert_emb import BertEmbedding
import numpy as np
from src.core.utils import (
    get_wikipedia_text,
    read_and_split_paragraphs,
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

    # Load the text from a wikipedia page as a context
    def load_context_from_wiki(self, url: str) -> None:
        # Load all paragraphs from the wikipedia URL
        paragraphs = get_wikipedia_text(url)
        # Calculate the embeddings from all paragraphs
        embeddings = self.embed(paragraphs)
        # Add the document to the document manager
        self.add_document(Document(paragraphs, embeddings, url))

    # Load the text from a file as a context
    def load_context_from_txt(self, path: str) -> None:
        # Load all paragraphs from a file
        paragraphs = read_and_split_paragraphs(path)
        # Calculate the embeddings from all paragraphs
        embeddings = self.embed(paragraphs)
        # Add the document to the document manager
        self.add_document(Document(paragraphs, embeddings, path))

    # Load the text from a file as a context
    def load_context_from_raw_text(self, raw_text: str) -> None:
        # Load all paragraphs from a file
        paragraphs = split_paragraphs(raw_text)
        # Calculate the embeddings from all paragraphs
        embeddings = self.embed(paragraphs)
        # Add the document to the document manager
        self.add_document(Document(paragraphs, embeddings, "raw_text"))
