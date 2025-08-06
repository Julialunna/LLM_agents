from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatDeepInfra
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_PROVIDER_MAP ={
    "deepinfra": ChatDeepInfra, 
    "google": ChatGoogleGenerativeAI
}

MODEL_CONFIGS = [
    {
    "key_name": "gemini_flash",
    "provider": "google", 
    "model_name": "models/gemini-1.5-flash", 
    "temperature": 1.0
    },
    {
        "key_name": "meta_llama_3",
        "provider": "deepinfra",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "temperature": 0.0
    },
    {
        "key_name": "meta_llama_4",
        "provider": "deepinfra",
        "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "temperature": 0.0
    },
]

def _create_chat_model(model_name: str, provider: str, temperature: float | None=None):
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Provedor não suportado: {provider}. Provedores suportados são: {list(_PROVIDER_MAP.keys())}")
    model_class = _PROVIDER_MAP[provider]
    params = {"model": model_name}
    if temperature is not None:
        params["temperature"] = temperature
    return model_class(**params,  max_tokens=2048)

models = {}

for config in MODEL_CONFIGS:
    models[config["key_name"]] = _create_chat_model(
        model_name = config["model_name"], 
        provider = config["provider"],
        #criado diferentem pois temperature pode ser none, caso fosse feito como acima daria erro
        temperature = config.get("temperature")
    )