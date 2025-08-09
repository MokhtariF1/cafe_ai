import csv
import os
from pydoc import doc
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import tqdm

from backend import settings
from backend.settings import get_settings

settings = get_settings()
FAISS_DIR_BASE = settings.FAISS_DIR
DATA_DIR = settings.DATA_DIR


# class BusinnessCSVDataProvider:
#     def __init__(
#         self,
#         bussiness_uid: str,
#         ollama_embedding=OllamaEmbeddings(model="nomic-embed-text"),
#     ) -> None:
#         self.bussiness_uid = bussiness_uid
#         self.embed = ollama_embedding

#         self.index_bussiness()

#     def index_bussiness(self):
#         self.check_folders()

#         if os.path.exists(self.FAISS_DIR):
#             self.vectorstore = FAISS.load_local(
#                 self.FAISS_DIR, self.embed, allow_dangerous_deserialization=True
#             )
#         else:
#             csv_files = [
#                 file
#                 for file in os.listdir(f"data/{self.bussiness_uid}")
#                 if file.endswith(".csv")
#             ]
#             docs = []
#             for csv_file in tqdm.tqdm(
#                 csv_files, desc=f"indexing csv files for bussiness {self.bussiness_uid}"
#             ):
#                 with open(csv_file, encoding="utf-8") as f:
#                     reader = csv.DictReader(f)
#                     for row in reader:
#                         content = " - ".join([f"{k}: {v}" for k, v in row.items()])
#                         docs.append(Document(page_content=content))

#             self.vectorstore = FAISS.from_documents(docs, self.embed)
#             self.vectorstore.save_local(self.FAISS_DIR)

#     def check_folders(self):
#         if "data" not in os.listdir():
#             os.mkdir("data")

#         if self.bussiness_uid not in os.listdir("data"):
#             os.mkdir(f"data/{self.bussiness_uid}")

#         self.FAISS_DIR = f"data/{self.bussiness_uid}/{FAISS_DIR_BASE}"

#     def search(self, query: str) -> str:
#         results = self.vectorstore.similarity_search(query, k=3)
#         context = "\n".join([doc.page_content for doc in results])
#         return context


class BusinnessDataProvider:
    def __init__(
        self,
        bussiness_uid: str,
        ollama_embedding=OllamaEmbeddings(model="nomic-embed-text"),
    ) -> None:
        self.bussiness_uid = bussiness_uid
        self.embed = ollama_embedding
        self.FAISS_DIR = f"{DATA_DIR}/{self.bussiness_uid}/{FAISS_DIR_BASE}"
        self.check_folders()

    def check_folders(self):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)

        if self.bussiness_uid not in os.listdir(DATA_DIR):
            os.mkdir(f"{DATA_DIR}/{self.bussiness_uid}")

    def index_bussiness(self, file: Optional[str]):
        self.check_folders()

        if os.path.exists(self.FAISS_DIR):
            self.vectorstore = FAISS.load_local(
                self.FAISS_DIR, self.embed, allow_dangerous_deserialization=True
            )
        else:
            if file is None:
                files = self._get_bussiness_files()
            else:
                files = [file]
            docs = []
            for file in tqdm.tqdm(
                files, desc=f"indexing csv files for bussiness {self.bussiness_uid}"
            ):
                with open(file, encoding="utf-8") as f:
                    if file.endswith(".csv"):
                        reader = csv.DictReader(f)
                        for row in reader:
                            content = " - ".join([f"{k}: {v}" for k, v in row.items()])
                            docs.append(Document(page_content=content))
                    elif file.endswith(".txt"):
                        for line in f.readlines():
                            docs.append(line)

            if len(docs) > 0:
                self.vectorstore = FAISS.from_documents(docs, self.embed)
            self.vectorstore.save_local(self.FAISS_DIR)

    def _get_bussiness_files(self) -> list[str]:
        bussiness_files = [
            file for file in os.listdir(f"{DATA_DIR}/{self.bussiness_uid}")
        ]
        return bussiness_files

    def search(self, query: str) -> str:
        results = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        return context

    def add_index(self, filename: str):
        self.index_bussiness(filename)
