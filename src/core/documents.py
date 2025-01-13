from typing import List
import numpy as np
from dataclasses import dataclass
import pickle


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
        if len(self.document_embeddings) > 0:
            idx = np.argmax(self.document_embeddings @ emb_question.T).item()
            return self.documents[idx]
        return None

    def get_documents_list(self):
        return [
            {"id": idx, "name": doc.document_ref}
            for idx, doc in enumerate(self.documents)
        ]

    def delete_document(self, idx):
        if idx >= len(self.documents):
            return False
        self.document_embeddings = np.delete(self.document_embeddings, idx, axis=0)
        self.documents.pop(idx)
        self.embedding = np.mean(self.document_embeddings, axis=0)
        return True


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

    def get_all_documents_list(self):
        return {
            idx: cluster.get_documents_list()
            for idx, cluster in enumerate(self.clusters)
        }

    def delete_document(self, cluster_id, document_id):
        return self.clusters[cluster_id].delete_document(document_id)

    def save_snapshot(self, path):
        snapshot = {
            "clusters": self.clusters,
            # "cluster_embeddings": self.cluster_embeddings,
        }
        with open(path, "wb") as f:
            pickle.dump(snapshot, f)

    def load_snapshot(self, path):
        with open(path, "rb") as f:
            snapshot = pickle.load(f)
        self.clusters = snapshot["clusters"]
        # self.cluster_embeddings = snapshot["cluster_embeddings"]
