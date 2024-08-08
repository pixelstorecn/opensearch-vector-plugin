from typing import List

from langchain_community.vectorstores import OpenSearchVectorSearch

from langflow.base.vectorstores.model import LCVectorStoreComponent
from langflow.helpers.data import docs_to_data
from langflow.io import HandleInput, IntInput, StrInput, SecretStrInput, DataInput, MultilineInput
from langflow.schema import Data

class OpenSearchComponent(LCVectorStoreComponent):
    """
    A custom component for implementing a Vector Store using OpenSearch.
    """
    
    display_name: str = "OpenSearchVector"
    description: str = "Implementation of Vector Store using OpenSearch"
    documentation = "https://python.langchain.com/v0.2/docs/integrations/vectorstores/opensearch/"
    name = "OpenSearch"
    icon = "database"

    inputs = [
        StrInput(name="opensearch_url", display_name="OpenSearch Connection String", required=True),
        StrInput(name="index_name",display_name="Index Name",required=True),
        StrInput(name="opensearch_user",display_name="OpenSearch Username",info="Enter your username."),
        SecretStrInput(name="opensearch_pass",display_name="OpenSearch Password",info="Enter your password."), 
        StrInput(name="code", display_name="Code", advanced=True),
        MultilineInput(name="search_query", display_name="Search Query"),
        DataInput(
                name="ingest_data",
                display_name="Ingest Data",
                is_list=True,
        ),
        IntInput(
                name="number_of_results",
                display_name="Number of Results",
                info="Number of results to return.",
                value=4,
                advanced=True,
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"])
    ]

    def build_vector_store(self) -> OpenSearchVectorSearch:
        documents = []

        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        try:
            vector_store = OpenSearchVectorSearch.from_documents(
                embedding=self.embedding,
                documents=documents,
                index_name=self.index_name,
                opensearch_url=self.opensearch_url,
                http_auth=(self.opensearch_user,self.opensearch_pass)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build OpenSearchVector: {e}")
        return vector_store
    
    def search_documents(self) -> List[Data]:
        vector_store = self.build_vector_store()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
            )

            data = docs_to_data(docs)
            self.status = data
            return data
        else:
            return []