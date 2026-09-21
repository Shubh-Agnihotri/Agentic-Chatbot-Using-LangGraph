from backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
import streamlit as st
import uuid
import tempfile
import os

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state['thread_id'])

def load_conversation(thread_id):

    state = chatbot.get_state(
        config={"configurable": 
                {"thread_id": thread_id}}
            )

    return state.values.get("messages", [])

# config = {"thread_id": "1"}

st.title("Agentic Chatbot with LangGraph and LangChain")

# Initialize message history when app runs for the first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# List for storing all conversation threads IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()
                                    
# Initialize thread_id when app runs for the first time
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

add_thread(st.session_state['thread_id'])

# Sidebar for managing conversation threads
st.sidebar.title("Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:  # Display threads in reverse order (latest first)

    if st.sidebar.button(str(thread_id), key = thread_id):

        #set selected thread_id as current thread_id 
        st.session_state['thread_id'] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            temp_messages.append(
                {"role": role, "content": message.content}
            )

        st.session_state['message_history'] = temp_messages

        st.rerun()

# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# ========================= Fixed chat input with PDF upload =========================

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"]
)

user_input = None

# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name


            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )


            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            # Delete the temporary PDF after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)



# user_input = st.chat_input("Type your message here...")

if user_input:
    
    st.session_state['message_history'].append({
        "role": "user", 
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.text(user_input)

    config = {
    "configurable":{
        "thread_id": st.session_state['thread_id']
        },
        "metadata": {
            "thread_id": st.session_state['thread_id']
        },
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode = 'messages'):

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using '{tool_name}' ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using '{tool_name}' __",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool finished", state="complete", expanded=False
            )


    st.session_state['message_history'].append({
        "role": "assistant", 
        "content": ai_message
        }
    )
