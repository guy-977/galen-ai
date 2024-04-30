import streamlit as st
import tensorflow as tf
from src.models.predict_label import get_prediction
from src.models.heatmap import *
import tempfile
from src.llm import groq

st.set_page_config(page_title='Galen AI (using Mixtral-8x7B)')
st.title("Galen AI (using Mixtral-8x7B)")

# Caching the model for faster loading
@st.cache_resource
# laod the model
def load_model(model_path):
  model = tf.keras.models.load_model(model_path)
  return model
model = load_model('models/skinNet.h5')

#Chose Groq API model
groq_model = st.sidebar.multiselect('Model', ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768', ' gemma-7b-it'], max_selections=1)

# Add uploader function
st.sidebar.subheader("File Uploader:", divider='rainbow')

uploaded_file = st.sidebar.file_uploader("Choose files",
                                            type=['.jpg', '.jpeg', '.png'],
                                            accept_multiple_files=False)
prediction = None
if uploaded_file:
    st.sidebar.image(uploaded_file, caption='Image uploaded by patient', width=100)
    classification = get_prediction(uploaded_file, model)
    # CNN model result
    prediction = f"it's {classification[0][0]} with {classification[0][1]} probability"
    with st.container(border=True):
        st.sidebar.subheader('Probabilites of Classification', divider='rainbow')
        st.sidebar.metric(f':green[{classification[0][0]}]', value='{:.2f}%'.format(classification[0][1]))
        st.sidebar.metric(f':green[{classification[1][0]}]', '{:.2f}%'.format(classification[1][1]))
        st.sidebar.metric(f':green[{classification[2][0]}]', '{:.2f}%'.format(classification[2][1]))
    heatmap = make_gradcam_heatmap(uploaded_file, model, last_conv_layer_name='conv2d_2')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        save_and_display_gradcam(uploaded_file, heatmap, cam_path=temp_file_path, alpha=0.8)
        st.sidebar.image(temp_file_path, caption='Grad Cam', width=260)
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Hello 👋, I am Galen (mixtral-8x7B-Instruct-v0_1), I'm here to help you as an AI medical assistant")


#Input 1 patient information: sex & age
age = st.number_input('Age', step=1)
sex = st.multiselect('Sex', ['Male', 'Female', 'Other'], max_selections=1)

#input 2 patient history
patient_history = st.text_input('Patient history')

# input 3 lesion characters
lesion_char = st.text_input('Lesion characters')

# TO DO: add CNN model prediction in the prompt
intro = 'Hi, I need some information about the following information'
# lesion ="melanoma"
action = "Please give a definition of the disease, a short description of it, the level of danger and what are the actions to take. Answer in 4 different bullet points with the 4 different titles [Definition, Description, Threat, Actions to take] ."

if prediction is not None:
    prompt = f'<s> [INST] {intro}. I am  {age} years old, my sex is defined as{sex}. My medical history is {patient_history}. I have a {lesion_char}. According to CNN model prediction: {prediction}. {action} "[/INST]'


# We generate an answer only where there is a prompt
if st.button('Generate', type="primary"):
    try:
        # Formatting the prompt
        prompt =  "<s> [INST] " + prompt +  " [/INST]"
            
        if groq_model:
            llm_generation = groq.generate(st.secrets["GROQ_API_KEY"], prompt, model_name=groq_model[0])
        else:
            llm_generation = groq.generate(st.secrets["GROQ_API_KEY"], prompt)

        with st.container(border=True):
            st.write(llm_generation)
        
    except Exception as err:
        st.exception('You must upload an image of the skin lesion!')
        print(err)