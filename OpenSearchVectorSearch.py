from typing import List, Union,Optional

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langflow import CustomComponent

class OpenSearchComponent(CustomComponent):
    """
    A custom component for implementing a Vector Store using OpenSearch.
    """
    display_name: str = "OpenSearchVector"
    description: str = "Implementation of Vector Store using OpenSearch"
    documentation = "https://python.langchain.com/v0.2/docs/integrations/vectorstores/opensearch/"

    def build_config(self):
        """
        Builds the configuration for the component.

        Returns:
        - dict: A dictionary containing the configuration options for the component.
        """
        return {
            "index_name": {"display_name": "Index Name", "value": "your_index"},
            "code": {"show": False, "display_name": "Code"},
            "documents": {"display_name": "Documents", "is_list": True},
            "embedding": {"display_name": "Embedding"},
            "opensearch_url": {
                "display_name": "OpenSearch Connection Url",
                "advanced": False,
            },
            "opensearch_user":{ "display_name": "OpenSearch Username"},
            "opensearch_pass":{ "display_name": "OpenSearch Password","advanced": True }
        }

    def build(
        self,
        embedding: Embeddings,
        opensearch_url: str,
        index_name: str,
        documents: List[Document],
        opensearch_user: Optional[str] = None,
        opensearch_pass: Optional[str] = None
    ) -> Union[VectorStore, BaseRetriever]:
        """
        Builds the Vector Store or BaseRetriever object.

        Args:
        - embedding (Embeddings): The embeddings to use for the Vector Store.
        - documents (Optional[Document]): The documents to use for the Vector Store.
        - index_name (str): The name of the OpenSearch Index.
        - opensearch_url (str): The URL for the OpenSearch.

        Returns:
        - VectorStore: The Vector Store object.
        """
        try:
            vector_store = OpenSearchVectorSearch.from_documents(
                embedding=embedding,
                documents=documents,
                index_name=index_name,
                opensearch_url=opensearch_url,
                http_auth=(opensearch_user,opensearch_pass)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build OpenSearchVector: {e}")
        return vector_store