from langchain_chroma import Chroma

import config_data as config


class VectorStoreService:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self, k: int | None = None):
        """Return the Chroma vector retriever for compatibility with old code."""
        search_kwargs = {"k": k or config.retriever_k}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def search_with_scores(self, query: str, k: int | None = None, threshold: float | None = None):
        """Return Chroma relevance scores. Higher score means more relevant."""
        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=k or config.vector_top_k,
        )
        threshold = config.similarity_threshold if threshold is None else threshold
        if threshold is None:
            return results
        return [
            (document, score)
            for document, score in results
            if score >= threshold
        ]


if __name__ == "__main__":
    pass
