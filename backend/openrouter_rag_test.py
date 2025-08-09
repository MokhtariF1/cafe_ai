import os

from typing import Optional, List, Any, Mapping

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from pydantic import Field
from openai import OpenAI

from backend.bussiness_providers import BusinnessCSVDataProvider
from backend.settings import get_settings

settings = get_settings()



MODEL_NAME = "openrouter/horizon-beta"
CSV_FILE = "cafe_menu.csv"
FAISS_DIR = "faiss"

    
# ===== LLM سفارشی با حافظه =====
class OpenRouterLLM(LLM):
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)

    def _call(self, prompt: str, bussiness_uid: str, client: OpenAI, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        history = self.memory.load_memory_variables({})["history"]

        business_custom_prompt = self._get_business_prompt(bussiness_uid)

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"تو یک دستیار حرفه‌ای کافه هستی که مشتری‌ها را بر اساس منوی موجود راهنمایی می‌کند. محترمانه و کوتاه جواب بده.\n{business_custom_prompt}"},
                {"role": "user", "content": history + "\n" + prompt}
            ]
        )

        answer = resp.choices[0].message.content
        # ذخیره در حافظه
        self.memory.chat_memory.add_user_message(prompt)
        self.memory.chat_memory.add_ai_message(answer)
        return answer
    
    def _get_business_prompt(self, bussiness_uid: str):
        data_provider = BusinnessCSVDataProvider(bussiness_uid)
        return str(data_provider)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": MODEL_NAME}

    @property
    def _llm_type(self) -> str:
        return "openrouter_llm"


# ===== اجرای چت =====
llm = OpenRouterLLM()

while True:
    query = input("\n🍵 سوال شما: ")
    if query.lower() in ["exit", "quit", "خروج"]:
        break
    
    
    final_prompt = f"""
سوال مشتری: {query}
"""
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)
answer = llm(prompt=final_prompt, bussiness_uid="cafe", client=client)
print("🤖 پاسخ:", answer)
