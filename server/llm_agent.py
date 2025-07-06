import os
import sys
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from mcp.server.fastmcp import FastMCP

# --- Configuration ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found for LLMResponseAgent.")

# --- Agent State ---
mcp = FastMCP("llm-response-agent")
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
model_name = "deepseek-chat"

@mcp.tool()
def generate_answer(query: str, context: List[str]) -> str:
    """
    Generates a final, user-facing answer based on the user's query and the
    retrieved context from the vector store.

    Args:
        query: The original question from the user.
        context: A list of relevant text chunks retrieved from the documents.
    """
    print(f"[LLMAgent] Received request to generate answer for query: '{query}'", flush=True)

    context_str = "\n\n---\n\n".join(context)

    system_prompt = (
        "You are an expert Q&A assistant. Your task is to answer the user's question based *only* on the provided context."
        " If the information is not present in the context, you must state that you cannot answer the question with the given information."
        " Do not use any external knowledge. Be concise and directly address the question."
    )

    user_message = (
        f"Here is the context retrieved from the documents:\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"Based on the context above, please answer the following question:\n\n"
        f"QUESTION: {query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        print("[LLMAgent] Calling LLM to synthesize final answer...", flush=True)
        final_response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
        print("[LLMAgent] Successfully generated answer.", flush=True)
        return final_answer
    except Exception as e:
        error_message = f"An error occurred while generating the final answer: {e}"
        print(f"[LLMAgent] ERROR: {error_message}", flush=True)
        return error_message

if __name__ == "__main__":
    print("[LLMAgent] Starting up...", flush=True)
    mcp.run(transport='stdio')