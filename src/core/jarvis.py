from src.ai.engine import AIEngine
from src.core.documents import DocumentManager, Document
from src.core.utils import split_text
from typing import Dict
from src.core.utils import generate_speech


class JARVIS(AIEngine, DocumentManager):
    def __init__(self):
        AIEngine.__init__(self)
        DocumentManager.__init__(self)

    # Load the text from a file as a context
    def load_context(self, raw_text: str, filename: str = "raw_text") -> None:
        # Load all paragraphs from a file
        paragraphs = split_text(raw_text)
        # Calculate the embeddings from all paragraphs
        embeddings = self.embed(paragraphs)
        # Add the document to the document manager
        self.add_document(Document(paragraphs, embeddings, filename))

    def get_context(self, question: str) -> str:
        # Embed the question
        emb_question = self.embed(question)

        # Get context and context embeddings based on the question
        document = self.get_document(emb_question)
        if document is None:
            return ""

        # Find all possible paragraphs that are related to the question
        chunk = document.get_chunk(emb_question)

        return chunk.text

    def answer(self, question: str) -> Dict[str, any]:
        # Embed the question
        emb_question = self.embed(question)

        # Get context and context embeddings based on the question
        document = self.get_document(emb_question)

        if document is None:
            return {"answer": "Could not find any document to answer your question"}

        # Find all possible paragraphs that are related to the question
        chunk = document.get_chunk(emb_question)

        # Find the answer to the question in the context
        ans = self.answer_from_context(question, chunk.text)

        return {"answer": ans, "chunk": chunk}

    def speak(self, text: str):
        generate_speech(text)
