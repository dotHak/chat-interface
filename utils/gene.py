from langchain_openai import OpenAIEmbeddings
from typing import Union, TypedDict, Literal, List, NotRequired
from langchain.schema import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os


class DocumentMetadata(TypedDict):
    section: NotRequired[
        Literal["doctor_speciality", "contact", "about", "services", "doctors"]
    ]
    tag: NotRequired[Union[str, None]]
    service: NotRequired[Union[str, None]]


class Gene:
    def __init__(
        self,
        all_docs: List[Document],
        should_persist: bool = False,
        should_override_persist: bool = False,
        persist_index: str = "faiss_index",
        embeddings_model: str = "text-embedding-ada-002",
        embeddings_size: int = 1536,
    ):
        self.should_persist = should_persist
        self.should_override_persist = should_override_persist
        self.persist_index = persist_index
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.embeddings_size = embeddings_size
        self.vector_store = self.load_vector_store(all_docs)

    def persist(self):
        if not self.is_index_saved() and self.should_override_persist:
            self.vector_store.save_local(self.persist_index)

    def load_vector_store(self, docs: List[Document]):

        if self.is_index_saved():
            vector_store = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            index = faiss.IndexFlatL2(self.embeddings_size)

            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            docs_ids = [doc.id for doc in docs]
            vector_store.add_documents(documents=docs, ids=docs_ids)
            if self.should_persist:
                self.persist()

        return vector_store

    def is_index_saved(self) -> bool:
        project_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        index_file = os.path.join(
            project_dir, self.persist_index, "index.faiss"
        )
        metadata_file = os.path.join(
            project_dir, self.persist_index, "index.pkl"
        )
        return os.path.exists(index_file) and os.path.exists(metadata_file)

    def search(
        self,
        query: str,
        k: int = 1,
        q_filter: Union[DocumentMetadata, None] = None,
    ):
        if q_filter:
            section = q_filter.get("section", None)
            tag = q_filter.get("tag", None)
            service = q_filter.get("service", None)

            if tag:
                results = self.vector_store.similarity_search(
                    query, k=k, filter={"tag": tag}
                )

                if results:
                    return results

            if service:
                results = self.vector_store.similarity_search(
                    query, k=k, filter={"services": service}
                )

                if results:
                    return results

            if section:
                results = self.vector_store.similarity_search(
                    query, k=k, filter={"section": section}
                )

                if results:
                    return results

        results = self.vector_store.similarity_search(query, k=k)

        return results

    def search_mmr(
        self,
        query: str,
        k: int = 5,
        q_filter: Union[DocumentMetadata, None] = None,
    ):
        if q_filter:
            section = q_filter.get("section", None)
            tag = q_filter.get("tag", None)
            service = q_filter.get("service", None)

            if tag:
                results = self.vector_store.max_marginal_relevance_search(
                    query, k=k, filter={"tag": tag}
                )

                if results:
                    return results

            if service:
                results = self.vector_store.similarity_search(
                    query, k=k, filter={"services": service}
                )

                if results:
                    return results

            if section:
                results = self.vector_store.max_marginal_relevance_search(
                    query, k=k, filter={"section": section}
                )

                if results:
                    return results

        results = self.vector_store.max_marginal_relevance_search(query, k=k)

        return results

    @staticmethod
    def format(docs: List[Document]) -> str:
        return_str = ""
        for doc in docs:
            return_str += f"* {doc.page_content} [{doc.metadata}] \n\n"
        return return_str
