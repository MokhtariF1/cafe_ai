import os
import csv
import pickle
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI

# ================= تنظیمات =================
os.environ["OPENROUTER_API_KEY"] = "کلید_API_خودت"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL_NAME = "openai/gpt-3.5-turbo"  # مدل LLM از OpenRouter
INDEX_FILE = "cafe_faiss.pkl"
CSV_FILE = "cafe_menu.csv"

# ================= کلاس LLM از OpenRouter =================
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

# ================= ساخت یا لود بردارها =================
embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

if os.path.exists(INDEX_FILE):
    print("📂 لود کردن دیتابیس آماده...")
    with open(INDEX_FILE, "rb") as f:
        vectorstore = pickle.load(f)
else:
    print("⚙️ ساخت دیتابیس بردار از CSV...")
    docs = []
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = " - ".join([f"{k}: {v}" for k, v in row.items()])
            docs.append(Document(page_content=content))
    
    vectorstore = FAISS.from_documents(docs, embedding_fn)
    
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(vectorstore, f)

# ================= اجرای پرسش و پاسخ =================
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
