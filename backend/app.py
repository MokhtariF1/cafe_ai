from fastapi import FastAPI, UploadFile
from openai import OpenAI

from backend.bussiness_providers import BusinnessDataProvider
from backend.openrouter import OpenRouterLLM
from backend.settings import get_settings


settings = get_settings()

app = FastAPI(title=settings.APP_TITLE, version=f"v{settings.APP_VERSION}")
llm = OpenRouterLLM()
client = OpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
)


@app.post("/{bussiness_uid}/chat")
def chat(bussiness_uid: str, query: str):
    return llm(prompt=query, bussiness_uid=bussiness_uid, client=client)


@app.post("/{bussiness_uid}/index")
def index(bussiness_uid: str):
    BusinnessDataProvider(bussiness_uid=bussiness_uid)
    return {"status": "ok"}


@app.post("/{bussiness_uid}/index/add-file")
def add_index(bussiness_uid: str, file: UploadFile):
    provider = BusinnessDataProvider(bussiness_uid=bussiness_uid)
    with open(f"{settings.DATA_DIR}/{bussiness_uid}/{file.filename}", "wb") as f:
        f.writelines(file.file.readlines())
    provider.add_index
    return {"status": "ok"}
