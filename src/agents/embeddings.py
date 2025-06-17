from langchain_ollama import OllamaEmbeddings

from config.settings_config import get_settings


def get_lang_store_embeddings():
    embedddings = OllamaEmbeddings(
        model=get_settings().lang_store_embeddings_model,
    )

    return embedddings, get_settings().lang_store_embeddings_model_dims
