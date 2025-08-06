import os
import csv
from typing import Optional, List, Any, Mapping
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferMemory
from pydantic import Field
from openai import OpenAI

# ===== تنظیمات =====
os.environ["OPENROUTER_API_KEY"] = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL_NAME = "openrouter/horizon-beta"
CSV_FILE = "cafe_menu.csv"
FAISS_DIR = "cafe_faiss"

# ===== LLM سفارشی با حافظه =====
class OpenRouterLLM(LLM):
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        # تاریخچه قبلی مکالمه
        history = self.memory.load_memory_variables({})["history"]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "تو یک دستیار حرفه‌ای کافه هستی که مشتری‌ها را بر اساس منوی موجود راهنمایی می‌کند. محترمانه و کوتاه جواب بده."},
                {"role": "user", "content": history + "\n" + prompt}
            ]
        )

        answer = resp.choices[0].message.content
        # ذخیره در حافظه
        self.memory.chat_memory.add_user_message(prompt)
        self.memory.chat_memory.add_ai_message(answer)
        return answer

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": MODEL_NAME}

    @property
    def _llm_type(self) -> str:
        return "openrouter_llm"

# ===== Embedding با Ollama =====
embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

# ===== ساخت یا لود دیتابیس =====
if os.path.exists(FAISS_DIR):
    print("📂 لود کردن دیتابیس آماده...")
    vectorstore = FAISS.load_local(FAISS_DIR, embedding_fn, allow_dangerous_deserialization=True)
else:
    print("⚙️ ساخت دیتابیس بردار از CSV...")
    docs = []
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = " - ".join([f"{k}: {v}" for k, v in row.items()])
            docs.append(Document(page_content=content))

    vectorstore = FAISS.from_documents(docs, embedding_fn)
    vectorstore.save_local(FAISS_DIR)

# ===== اجرای چت =====
llm = OpenRouterLLM()

while True:
    query = input("\n🍵 سوال شما: ")
    if query.lower() in ["exit", "quit", "خروج"]:
        break

    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    final_prompt = f"""
منوی کافه:
{context}

سوال مشتری: {query}
"""
    answer = llm(final_prompt)
    print("🤖 پاسخ:", answer)
