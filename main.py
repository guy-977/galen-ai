import streamlit as st
import random
import time
import os
from clarifai.client.model import Model
# from dotenv import load_dotenv
# load_dotenv()


# clarifai_pat = os.getenv('CLARIFAI_PAT')
clarifai_pat = st.secrets["CLARIFAI_PAT"]
CLARIFAI_PAT = clarifai_pat

# Model parameters or mixtral
inference_params = dict(temperature=0.7, max_tokens=200, top_k = 50, top_p= 0.95)

st.title("Chatbot with Mixtral")

st.metric(label="Score", value="1/4", delta="+25%")

st.metric(label="Score", value="2/4", delta="-25%")

st.metric(label="Score", value="Low", delta="-0%")

# Add uploader function
st.sidebar.subheader("File Uploader:")
selected_file = ""

uploaded_files = st.sidebar.file_uploader("Choose files",
                                            type=["txt", "html", "css", "py", "pdf", "ipynb", "docx", "csv"],
                                            accept_multiple_files=True)
st.sidebar.metric("Number of files uploaded", len(uploaded_files))

if uploaded_files:
    file_index = st.sidebar.selectbox("Select a file to display", options=[f.name for f in uploaded_files])
    selected_file = uploaded_files[[f.name for f in uploaded_files].index(file_index)]


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Hello 👋, I am MistralAI (mixtral-8x7B-Instruct-v0_1), how can I help you?")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input():
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# We generate an answer only where there is a prompt
if prompt is not None:
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Formatting the prompt
        prompt =  "<s> [INST] " + prompt +  " [/INST]"

        model_prediction = Model("https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
        # Take the answer
        full_response = model_prediction.outputs[0].data.text.raw

        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    