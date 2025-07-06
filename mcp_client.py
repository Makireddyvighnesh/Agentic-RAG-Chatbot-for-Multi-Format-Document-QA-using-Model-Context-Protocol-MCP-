import sys
import os
import json
from typing import Optional, List, Callable, Dict, Any
from contextlib import AsyncExitStack
from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables or .env file.")

class AgentCoordinator:
    """
    Coordinates the ingestion, retrieval, and LLM response agents.
    Handles the high-level logic and communicates with agents via MCP.
    """
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.exit_stack = AsyncExitStack()
        self.ingestion_session: Optional[ClientSession] = None
        self.retrieval_session: Optional[ClientSession] = None
        self.llm_session: Optional[ClientSession] = None  
        self.is_ready = False
        self.log = log_callback if log_callback else print

        self.llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        self.model_name = "deepseek-chat"

    async def _connect_to_agent(self, script_name: str) -> ClientSession:
        script_path = os.path.join(script_name)
        self.log(f"Starting agent: {script_path}...")
        server_params = StdioServerParameters(command=sys.executable, args=[script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        response = await session.list_tools()
        self.log(f"✅ Connected to {script_name}. Tools: {[tool.name for tool in response.tools]}")
        return session

    async def startup(self):
        self.log("--- Starting Agentic RAG System ---")
        self.ingestion_session = await self._connect_to_agent("./server/ingestion_agent.py")
        self.retrieval_session = await self._connect_to_agent("./server/retrieval_agent.py")
        self.llm_session = await self._connect_to_agent("./server/llm_agent.py") 
        self.log("\nAll agents are running and connected.")

    async def process_documents(self, file_paths: List[str]) -> bool:
        self.log("\n--- 1. Ingestion Phase ---")
        try:
            ingestion_result_obj = await self.ingestion_session.call_tool("process_files", {"file_paths": file_paths})
            ingestion_payload = ingestion_result_obj.structuredContent['result']
            if ingestion_payload.get("status") != "success":
                self.log(f"Error during ingestion: {ingestion_payload.get('message')}")
                return False
            
            self.log("\n--- 2. Indexing Phase ---")
            indexing_result_obj = await self.retrieval_session.call_tool("build_index")
            indexing_payload = indexing_result_obj.structuredContent['result']
            if indexing_payload.get("status") != "success":
                self.log(f"Error during indexing: {indexing_payload.get('message')}")
                return False

            self.is_ready = True
            self.log("\n✅ System is ready for Q&A.")
            return True
        except Exception as e:
            self.log(f"An error occurred during document processing: {e}")
            self.is_ready = False
            return False

    async def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Handles a user query by orchestrating the retrieval and LLM agents.
        Returns a dictionary with the answer and the source context.
        """
        if not self.is_ready:
            message = "System is not ready. Please process documents first."
            self.log(message)
            return {"answer": message, "context": []}

        # === STEP 1: Plan which retrieval tool to use ===
        tool_list_response = await self.retrieval_session.list_tools()
        available_tools = [
            {"type": "function", "function": tool.model_dump()}
            for tool in tool_list_response.tools
        ]
        planning_system_prompt = (
            "You are a planning agent. Your job is to decide which tool to use to help answer the user's question."
            " You must select the most appropriate tool based on the tool descriptions provided."
            " Return only a tool call. Do not answer the question directly."
        )
        messages = [
            {"role": "system", "content": planning_system_prompt},
            {"role": "user", "content": query}
        ]
        self.log("\n🤖 [Step 1/3] Asking Planner LLM to select a retrieval tool...")
        try:
            planning_response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages, 
                tools=available_tools, 
                max_tokens=500
            )
            response_message = planning_response.choices[0].message
        except Exception as e:
            error_msg = f"An error occurred during the planning phase: {e}"
            self.log(error_msg)
            return {"answer": error_msg, "context": []}

        if not response_message.tool_calls:
            no_tool_msg = "The model chose not to use a tool. I can only answer questions by retrieving information from the documents."
            self.log("❌ LLM did not invoke any tool.")
            return {"answer": response_message, "context": []}

        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        try:
            tool_args = json.loads(tool_args_str)
            self.log(f"✅ Tool selected: {tool_name} | Args: {tool_args}")
        except json.JSONDecodeError:
            error_msg = "I tried to use a tool to answer your question, but there was an internal error with the tool's arguments."
            self.log(f"Error: LLM provided invalid JSON arguments: {tool_args_str}")
            return {"answer": error_msg, "context": []}

        # === STEP 2: Call the RetrievalAgent to get context ===
        self.log(f"\n🤖 [Step 2/3] Calling RetrievalAgent with tool '{tool_name}'...")
        retrieval_result_obj = await self.retrieval_session.call_tool(tool_name, arguments=tool_args)

        print("retrieval_result_obj:\n", retrieval_result_obj)
        retrieved_context = retrieval_result_obj.structuredContent.get('result', [])
        if not retrieved_context:
            self.log("RetrievalAgent returned no context.")
        else:
            self.log(f"✅ Retrieved {len(retrieved_context)} context chunks.")

        # === STEP 3: Call the LLMResponseAgent to generate the final answer ===
        self.log("\n🤖 [Step 3/3] Calling LLMResponseAgent to generate final answer...")
        try:
            final_answer_obj = await self.llm_session.call_tool(
                "generate_answer",
                {"query": query, "context": retrieved_context}
            )

            print("Answer:\n", final_answer_obj)
            final_answer = final_answer_obj.structuredContent.get('result')
            self.log("\n✅ Final Answer Generated.")
            # Return both answer and context for the UI
            return {"answer": final_answer, "context": retrieved_context}
        except Exception as e:
            error_msg = f"An error occurred while generating the final answer: {e}"
            self.log(f"Error during final answering call: {e}")
            return {"answer": error_msg, "context": []}

    async def shutdown(self):
        self.log("\nShutting down all agents...")
        await self.exit_stack.aclose()
        self.log("Cleanup complete.")