# LLM_Playground/app.py

# The 'dotenv' library is removed as it's not needed for deployment.
import streamlit as st
from huggingface_hub import InferenceClient
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Playground",
    page_icon="🤖",
    layout="wide"
)

# --- Application Title ---
st.title("🤖 Conversational LLM Web App")
st.caption("Powered by Hugging Face and Streamlit")

def stream_parser(stream):
    """
    Parses the streaming output from Hugging Face API and yields the text content.
    """
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # --- MODIFIED: Use st.secrets for secure token handling ---
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        st.success("HF token loaded successfully.", icon="✅")
    else:
        st.error("HF token not found. Please add it to your Streamlit secrets.")
        st.stop()

    # Dropdown to select the model
    selected_model = st.selectbox(
        "Select a Model",
        [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta"
        ],
        index=0
    )
    
    # Sliders and text area for model parameters
    st.header("Parameter Controls")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=4096, value=512, step=8)

    st.header("System Prompt")
    system_prompt = st.text_area("Define the bot's persona and rules.", "You are a friendly and helpful assistant.")

    st.divider()

    # Button to clear the chat history
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

    # Functionality to export the conversation
    def format_chat_history_for_export(messages):
        formatted_text = ""
        for message in messages:
            role = "You" if message["role"] == "user" else "Bot"
            timestamp = message.get("timestamp", "")
            formatted_text += f"[{timestamp}] {role}:\n{message['content']}\n\n"
        return formatted_text.strip()

    if "messages" in st.session_state and st.session_state.messages:
        chat_export_data = format_chat_history_for_export(st.session_state.messages)
        st.download_button(
            label="Export Conversation",
            data=chat_export_data,
            file_name=f"conversation_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain"
        )

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Existing Chat Messages ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        if "timestamp" in message:
            st.caption(f"_{message['timestamp']}_")
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Enter your message here..."):
    timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.caption(f"_{timestamp}_")
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Bot is thinking..."):
            try:
                # The client automatically uses the token found by st.secrets
                client = InferenceClient(model=selected_model, token=hf_token)
                
                messages_with_system_prompt = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    messages_with_system_prompt.append({"role": msg["role"], "content": msg["content"]})

                response_stream = client.chat_completion(
                    messages=messages_with_system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

                parsed_stream = stream_parser(response_stream)
                response_content = st.write_stream(parsed_stream)

            except Exception as e:
                response_content = f"An error occurred: {str(e)}"
                st.error(response_content)

    # Add the full response to the session state for history
    ai_timestamp = f"{datetime.now():%Y-%m-%d %H:%M%S}"
    st.session_state.messages.append({"role": "assistant", "content": response_content, "timestamp": ai_timestamp})