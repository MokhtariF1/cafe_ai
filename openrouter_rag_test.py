import os
import csv
import json
import pickle
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI

# =============== تنظیمات OpenRouter با کتابخانه openai ===============
os.environ["OPENROUTER_API_KEY"] = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL_NAME = "deepseek/deepseek-r1:free"  # یا مدل رایگان دیگر

# =============== کلاس Embedding با LLM ===============
class OpenRouterPromptEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]
    
    def embed_query(self, text):
        return self._embed_text(text)
    
    def _embed_text(self, text):
        prompt = f"متن زیر را به یک بردار عددی 512 بُعدی تبدیل کن و فقط JSON لیست اعداد را بده:\n{text}"
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            vector = json.loads(resp.choices[0].message.content)
        except:
            vector = [0.0] * 512
        return vector

# =============== کلاس LLM برای پاسخ ===============
class OpenRouterLLM(LLM):
    def _call(self, prompt: str, stop=None, run_manager: CallbackManagerForLLMRun = None):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "تو یک دستیار حرفه‌ای کافه هستی که مشتری‌ها را بر اساس منوی موجود راهنمایی می‌کند. محترمانه و کوتاه جواب بده."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content
    
    @property
    def _identifying_params(self):
        return {"model": MODEL_NAME}
    
    @property
    def _llm_type(self):
        return "openrouter_llm"

# =============== ساخت یا لود کردن بردارها ===============
INDEX_FILE = "cafe_faiss.pkl"
embedding_fn = OpenRouterPromptEmbedding()

if os.path.exists(INDEX_FILE):
    print("📂 لود کردن دیتابیس آماده...")
    with open(INDEX_FILE, "rb") as f:
        vectorstore = pickle.load(f)
else:
    print("⚙️ ساخت دیتابیس بردار از CSV...")
    docs = []
    with open("cafe_menu.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = " - ".join([f"{k}: {v}" for k, v in row.items()])
            docs.append(Document(page_content=content))
    
    vectorstore = FAISS.from_documents(docs, embedding_fn)
    
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(vectorstore, f)

# =============== جستجو و پاسخ ===============
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
