from src.llm.bert_qa import BertQA
from src.core.knowledge import Knowledge
import numpy as np


def topk(arr, k, axis=0):
    idx = np.argpartition(arr, -k, axis=axis)[-k:]
    return np.take_along_axis(arr, idx, axis=axis)


class Chatbot(Knowledge):
    def __init__(self):
        super().__init__()

        # Loading Models
        self.bert_qa = BertQA()

    # Find the answer to the given question in the given context
    def answer_from_context(self, question: str, context: str) -> str:
        return self.bert_qa.answer(question, context)

    def answer(self, question: str) -> str:
        # Embed the question
        emb_question = self.embed(question)

        # Get context and context embeddings based on the question
        document = self.get_document(emb_question)

        # Find all possible paragraphs that are related to the question
        chunk = document.get_chunk(emb_question)

        # Find the answer to the question in the context
        return self.answer_from_context(question, chunk.text)


class AskMeAnything(Chatbot):
    def __init__(self):
        super(AskMeAnything, self).__init__()
