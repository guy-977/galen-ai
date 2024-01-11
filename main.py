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

# Add uploader function
st.sidebar.subheader("File Uploader:")

uploaded_file = st.sidebar.file_uploader("Choose files",
                                            type=['.jpg', '.jpeg', '.png'],
                                            accept_multiple_files=False)
if uploaded_file:
    st.sidebar.image(uploaded_file, caption='Image uploaded by patient', width=200)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Hello 👋, I am MistralAI (mixtral-8x7B-Instruct-v0_1), how can I help you?")


#Input 1 patient information: sex & age
age = st.number_input('age', step=1)
sex = st.multiselect('sex', ['male', 'female', 'other'])
#input 2 patient history
patient_history = st.text_input('patient history')

# input 3 lesion characters
lesion_char = st.text_input('lesion characters')

prompt = f'the patient is  {age} years old, {sex},his history: {patient_history},the lesion characters: {lesion_char}'


# We generate an answer only where there is a prompt
if st.button('generate'):
    # Formatting the prompt
    prompt =  "<s> [INST] " + prompt +  " [/INST]"
            
    model_prediction = Model("https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    # Take the answer
    full_response = model_prediction.outputs[0].data.text.raw
    st.write(full_response)

    