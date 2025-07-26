import pandas as pd
import os
import asyncio
import nest_asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from lightrag.rerank import custom_rerank, RerankModel
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import setup_logger
from lightrag.utils import TokenTracker

# Setup logger
setup_logger("lightrag", level="DEBUG")

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDbjYAPstou6ljyq26Z-gmTP05_AnyS4xs")

# Working directory setup
WORKING_DIR = "./TA_Storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

token_tracker = TokenTracker()

# LLM Model function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    client = genai.Client(api_key=gemini_api_key)
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=5000, temperature=0, top_k=10),
    )

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

    token_counts = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    token_tracker.add_usage(token_counts)

    return response.text

# Initialize LightRAG
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        entity_extract_max_gleaning=1,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        embedding_cache_config={"enabled": True, "similarity_threshold": 0.90},
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
        vector_storage="FaissVectorDBStorage",
        rerank_model_func=my_rerank_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# Function to process the Excel file and run queries
def process_excel_and_run_queries():
    # Read the Excel file with questions
    df = pd.read_excel("data_uji.xlsx")  # Assuming file name is 'data_uji.xlsx'

    # Create a list to hold results
    results = []

    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    for index, row in df.iterrows():
        question = row["Pertanyaan"]
        answers = {
            "Pertanyaan": question,
            "Jawaban Naive": rag.query(question, param=QueryParam(mode="naive", only_need_context=True)),
            "Jawaban Local": rag.query(question, param=QueryParam(mode="local")),
            "Jawaban Global": rag.query(question, param=QueryParam(mode="global")),
            "Jawaban Hybrid": rag.query(question, param=QueryParam(mode="hybrid")),
            "Total token retrieval": token_tracker.get_total_usage(),  # Sum of all tokens
        }
        results.append(answers)

    # Create a DataFrame from results
    result_df = pd.DataFrame(results)

    # Save the results to an Excel file or CSV
    result_df.to_csv("hasil_uji.csv", index=False)  # Save to CSV
    # result_df.to_excel("hasil_uji.xlsx", index=False)  # Uncomment to save as Excel

# Main function to trigger the process
if __name__ == "__main__":
    process_excel_and_run_queries()
