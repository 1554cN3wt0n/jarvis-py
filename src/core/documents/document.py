from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    reference: str


class Document:
    chunks: List[Chunk]
    chunk_embeddings: np.array
    embedding: np.array
    document_ref: str

    def __init__(self, chunk_texts, chunk_embeddings, document_ref: str):
        self.chunks = [Chunk(text, document_ref) for text in chunk_texts]
        self.chunk_embeddings = chunk_embeddings
        self.embedding = np.mean(self.chunk_embeddings, axis=0)
        self.document_ref = document_ref

    def get_chunk(self, emb_question: np.array) -> Chunk:
        idx = np.argmax(self.chunk_embeddings @ emb_question.T).item()
        return self.chunks[idx]


@dataclass
class DocumentCluster:
    documents: List[Document]
    document_embeddings: np.array
    embedding: np.array

    def add_document(self, document: Document):
        self.documents.append(document)
        document_embeddings_avg = np.mean(document.chunk_embeddings, axis=0)
        if self.document_embeddings is None:
            self.document_embeddings = np.vstack([document_embeddings_avg])
        else:
            self.document_embeddings = np.vstack(
                [self.document_embeddings, document_embeddings_avg]
            )
        self.embedding = np.mean(self.document_embeddings, axis=0)

    def get_document(self, emb_question: np.array) -> Document:
        idx = np.argmax(self.document_embeddings @ emb_question.T).item()
        return self.documents[idx]


class DocumentManager:
    clusters: List[DocumentCluster]
    cluster_embeddings: np.array

    def __init__(self):
        self.clusters = [DocumentCluster([], None, None)]

    def add_document(self, document: Document):
        cluster_idx = 0
        self.clusters[cluster_idx].add_document(document)

    def get_document(self, emb_question: np.array) -> Document:
        cluster_idx = 0
        return self.clusters[cluster_idx].get_document(emb_question)
