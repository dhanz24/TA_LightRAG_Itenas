# pip install -q -U google-genai to use gemini as a client
import pandas as pd
import csv
import os
import asyncio
import numpy as np
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

setup_logger("lightrag", level="DEBUG")

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "./TA_Storage"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

token_tracker = TokenTracker()


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=gemini_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=5000, temperature=0, top_k=10
            ),
        )
        
        # 4. Get token counts with null safety
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )

        token_counts = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        token_tracker.add_usage(token_counts)
        token_tracker.add_prompt(combined_prompt)

        # Return the response text
        return response.text

    except genai.errors.ServerError as e:
        if e.status_code == 503:
            print(f"Model is overloaded. Error: {e.message}")
            # Return empty response if server is overloaded, or customize behavior
            return "Error: Server Overloaded"
        else:
            raise  # Re-raise the exception for other errors
    

async def llm_model_func_Groq(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


async def my_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
    """Custom rerank function with all settings included"""
    return await custom_rerank(
        query=query,
        documents=documents,
        model="jina-reranker-v2-base-multilingual",
        base_url="https://api.jina.ai/v1/rerank",
        api_key=os.getenv("RERANK_BINDING_API_KEY"),
        top_n=top_n or 10,
        **kwargs,
    )


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        entity_extract_max_gleaning=1,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        embedding_cache_config={"enabled": False, "similarity_threshold": 0.90},
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
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.3  # Your desired threshold
        }
    )

    # rag = LightRAG(
    #     working_dir=WORKING_DIR,
    #     enable_llm_cache=False,
    #     enable_llm_cache_for_entity_extract=False,
    #     embedding_cache_config={"enabled": False, "similarity_threshold": 0.90},
    #     llm_model_func=llm_model_func_Groq,
    #     embedding_func=EmbeddingFunc(
    #             embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
    #             max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
    #             func=lambda texts: ollama_embed(
    #                 texts,
    #                 embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
    #                 host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
    #             ),
    #     ),
    #     rerank_model_func=my_rerank_func,
    # )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Read the Excel file with questions, specifying the sheet name
    df = pd.read_excel("PENELITIAN_TA.xlsx", sheet_name="Pertanyaan_sederhana")

    # Verify the column names in the sheet
    print("Columns in the sheet:", df.columns)

    # Check if 'Pertanyaan' column exists
    if 'Pertanyaan' not in df.columns:
        print("Error: 'Pertanyaan' column not found!")
        return  # Stop the execution if the column is missing

    # Create a list to hold results
    results = []

    # Context Manager Method
    with token_tracker:
        for index, row in df.iterrows():
            question = row["Pertanyaan"]  # Make sure the column name is 'Pertanyaan'

            token_tracker.reset()  # Reset token count
            
            # Run queries without unpacking token count from query
            naive_answer = rag.query(question, param=QueryParam(mode="naive"))
            # Get the token count after running queries
            naive_token_count = token_tracker.get_usage()  # Get token usage after query
            naive_prompt = token_tracker.get_prompt()
            token_tracker.reset()  # Reset token count
            local_answer = rag.query(question, param=QueryParam(mode="local"))
            local_token_count = token_tracker.get_usage()
            local_prompt = token_tracker.get_prompt()
            token_tracker.reset()
            global_answer = rag.query(question, param=QueryParam(mode="global"))
            global_token_count = token_tracker.get_usage()
            global_prompt = token_tracker.get_prompt()
            token_tracker.reset()
            hybrid_answer = rag.query(question, param=QueryParam(mode="hybrid"))
            hybrid_token_count = token_tracker.get_usage()
            hybrid_prompt = token_tracker.get_prompt()
            token_tracker.reset()

            # Store the results in a dictionary
            answers = {
                "Pertanyaan": question,
                "Jawaban Naive": naive_answer,
                "Jawaban Local": local_answer,
                "Jawaban Global": global_answer,
                "Jawaban Hybrid": hybrid_answer,
                "Naive Token Count": naive_token_count["total_tokens"],
                "Local Token Count": local_token_count["total_tokens"],
                "Global Token Count": global_token_count["total_tokens"],
                "Hybrid Token Count": hybrid_token_count["total_tokens"],

                "Naive Prompt": naive_prompt,
                "Local Prompt": local_prompt,
                "Global Prompt": global_prompt,
                "Hybrid Prompt": hybrid_prompt,

                "Naive Prompt Token Count": naive_token_count["prompt_tokens"],
                "Local Prompt Token Count": local_token_count["prompt_tokens"],
                "Global Prompt Token Count": global_token_count["prompt_tokens"],
                "Hybrid Prompt Token Count": hybrid_token_count["prompt_tokens"],
            }
            results.append(answers)

    # Save the results to an Excel file or CSV
    result_df = pd.DataFrame(results)
    try :
        result_df.to_csv("./Tugas_Akhir/results/hasil_uji_with_tokens.csv", index=False, quoting=csv.QUOTE_ALL)  # Menggunakan QUOTE_ALL untuk mengutip semua kolom
        result_df.to_excel("./Tugas_Akhir/results/hasil_uji_with_tokens.xlsx", index=False)  # Uncomment to save as Excel
    except PermissionError:
        result_df.to_csv("/Tugas_Akhir/results/hasil_uji_with_tokens_Backup.csv", index=False)
        result_df.to_excel("/Tugas_Akhir/results/hasil_uji_with_tokens_Backup.xlsx", index=False)
    except Exception as e:
        print(f"Error When Saving: {e}")


if __name__ == "__main__":
    main()
