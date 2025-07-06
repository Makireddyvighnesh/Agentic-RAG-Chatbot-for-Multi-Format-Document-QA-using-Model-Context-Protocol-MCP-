import streamlit as st
import asyncio
import os
import tempfile
from mcp_client import AgentCoordinator

# --- Page Configuration ---
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

# --- Asynchronous Event Loop Management ---
def get_or_create_eventloop():
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)
    return st.session_state.event_loop
loop = get_or_create_eventloop()

# --- App State Management ---
if "coordinator" not in st.session_state:
    st.session_state.coordinator = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()

# --- Main App Logic ---
st.title("🤖 Agentic RAG Chatbot")
st.caption("Upload documents, ask questions, and get answers from your own data.")

# --- Sidebar for Document Processing ---
with st.sidebar:
    st.header("1. Document Setup")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, etc.)",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'csv', 'pptx']
    )
    if st.button("Process Documents", disabled=not uploaded_files):
        with st.status("Processing documents...", expanded=True) as status:
            file_paths = []
            for uploaded_file in uploaded_files:
                try:
                    path = os.path.join(st.session_state.temp_dir.name, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    file_paths.append(path)
                except Exception as e:
                    st.error(f"Error saving file {uploaded_file.name}: {e}")
            if file_paths:
                st.session_state.messages = []
                def log_to_status(message):
                    status.write(message)
                try:
                    if st.session_state.coordinator is None:
                        coordinator = AgentCoordinator(log_callback=log_to_status)
                        loop.run_until_complete(coordinator.startup())
                        st.session_state.coordinator = coordinator
                    else:
                        st.session_state.coordinator.log = log_to_status
                    success = loop.run_until_complete(st.session_state.coordinator.process_documents(file_paths))
                    st.session_state.is_ready = success
                    if success:
                        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                        st.success("System is ready. You can now ask questions in the chat.")
                    else:
                        status.update(label="❌ Processing Failed", state="error", expanded=True)
                        st.error("Failed to process documents. Check the logs above.")
                except Exception as e:
                    status.update(label="❌ Critical Error", state="error", expanded=True)
                    st.error(f"A critical error occurred: {e}")
                    st.session_state.is_ready = False
    st.header("2. System Status")
    if st.session_state.is_ready:
        st.success("✅ System Ready for Q&A")
    else:
        st.warning("⚠️ System not ready. Please upload and process documents.")


# --- Main Chat Interface ---
st.header("💬 Chat with your Documents")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the message
        if "context" in message and message["context"]:
             with st.expander("View Sources"):
                for i, chunk in enumerate(message["context"]):
                    st.info(f"Source {i+1}:\n\n{chunk}")


# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.is_ready:
        st.error("Please process your documents first before asking questions.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... 🧠")

            # The coordinator's log can be disabled for the main screen
            def log_to_main_screen(message):
                pass
            st.session_state.coordinator.log = log_to_main_screen

            # Call the coordinator, which now returns a dictionary
            response_data = loop.run_until_complete(
                st.session_state.coordinator.answer_query(prompt)
            )

            final_answer = response_data.get("answer", "Sorry, I encountered an error.")
            source_context = response_data.get("context", [])

            # Display the final answer
            message_placeholder.markdown(final_answer)

            # Display the sources in an expander
            if source_context:
                with st.expander("View Sources"):
                    for i, chunk in enumerate(source_context):
                        st.info(f"Source {i+1}:\n\n{chunk}")

        # Add the full response (answer + context) to the session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "context": source_context
        })