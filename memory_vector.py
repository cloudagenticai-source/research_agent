import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os

EMBEDDING_MODEL = "text-embedding-3-small"

class VectorMemory:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize Chroma Persistent Client
        self.chroma_client = chromadb.PersistentClient(
            path="data/chroma",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create/Get Collections
        self.episodic = self.chroma_client.get_or_create_collection(name="episodic_memory")
        self.semantic = self.chroma_client.get_or_create_collection(name="semantic_memory")
        self.procedural = self.chroma_client.get_or_create_collection(name="procedural_memory")

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def upsert_episode(self, episode_id: int, canonical_text: str, meta: dict):
        """Upsert an episode into the episodic memory collection."""
        embedding = self.embed(canonical_text)
        self.episodic.upsert(
            ids=[f"episode:{episode_id}"],
            embeddings=[embedding],
            documents=[canonical_text],
            metadatas=[meta]
        )

    def upsert_fact(self, fact_id: int, canonical_text: str, meta: dict):
        """Upsert a fact into the semantic memory collection."""
        embedding = self.embed(canonical_text)
        self.semantic.upsert(
            ids=[f"fact:{fact_id}"],
            embeddings=[embedding],
            documents=[canonical_text],
            metadatas=[meta]
        )

    def upsert_skill(self, skill_id: str, canonical_text: str, meta: dict):
        """Upsert a skill into the procedural memory collection."""
        embedding = self.embed(canonical_text)
        self.procedural.upsert(
            ids=[f"skill:{skill_id}"],
            embeddings=[embedding],
            documents=[canonical_text],
            metadatas=[meta]
        )

    def query_episodic(self, query: str, k=10):
        """Query episodic memory."""
        embedding = self.embed(query)
        return self.episodic.query(
            query_embeddings=[embedding],
            n_results=k
        )

    def query_semantic(self, query: str, k=10):
        """Query semantic memory."""
        embedding = self.embed(query)
        return self.semantic.query(
            query_embeddings=[embedding],
            n_results=k
        )

    def query_procedural(self, query: str, k=3):
        """Query procedural memory."""
        embedding = self.embed(query)
        return self.procedural.query(
            query_embeddings=[embedding],
            n_results=k
        )
