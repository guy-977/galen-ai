import streamlit as st
import tensorflow as tf
from src.models.predict_label import get_prediction
from src.models.heatmap import *
import tempfile
from src.llm import groq

st.set_page_config(page_title='Galen AI | French')
st.title("Galen AI | French")

# Caching the model for faster loading
@st.cache_resource
# laod the model
def load_model(model_path):
  model = tf.keras.models.load_model(model_path)
  return model
model = load_model('models/skinNet.h5')

# Add uploader function
st.sidebar.subheader("Importer un fichier:")

uploaded_file = st.sidebar.file_uploader("Sélectionner un fichier",
                                            type=['.jpg', '.jpeg', '.png'],
                                            accept_multiple_files=False)

prediction = None
if uploaded_file:
    st.sidebar.image(uploaded_file, caption='Image importée par le patient', width=100)
    classification = get_prediction(uploaded_file, model)
    # CNN model result
    prediction = f"C'est {classification[0][0]} avec une probabilité de {classification[0][1]}"
    with st.container(border=True):
        st.sidebar.subheader('Probabilites of Classification', divider='rainbow')
        st.sidebar.metric(f':green[{classification[0][0]}]', value='{:.2f}%'.format(classification[0][1]))
        st.sidebar.metric(f':green[{classification[1][0]}]', '{:.2f}%'.format(classification[1][1]))
        st.sidebar.metric(f':green[{classification[2][0]}]', '{:.2f}%'.format(classification[2][1]))
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(uploaded_file, model, last_conv_layer_name='conv2d_2')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        save_and_display_gradcam(uploaded_file, heatmap, cam_path=temp_file_path, alpha=0.8)
        st.sidebar.image(temp_file_path, caption='Grad Cam', width=260)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Bonjour 👋, Je suis Galen (mixtral-8x7B-Instruct-v0_1), je suis ici pour vous aider comme assistant médical IA.")



#Input 1 patient information: sex & age
age = st.number_input('Age', step=1)
sex = st.multiselect('Sexe', ['Homme', 'Femme', 'Autre'], max_selections=1)

#input 2 patient history
patient_history = st.text_input('Antécédent médical')

# input 3 lesion characters
lesion_char = st.text_input('Caractéristique de la lésion')

# TO DO: add CNN model prediction in the prompt
intro = "Salut, j'ai besoin d'informations sur les éléments suivants."
action = "S'il vous plaît, donnez une définition de la maladie, une brève description, le niveau de danger et les actions à prendre. Répondez en 4 points avec les titres suivants : [**Définition**: La maladie est définie comme..., **Description** : Il s'agit d'une maladie caractérisée par..., **Niveau de danger**: Le niveau de danger associé à cette maladie est..., **Actions à prendre** : Pour faire face à cette maladie, il est recommandé de prendre les mesures suivantes...]."

if prediction is not None:
    # prompt = f'Le patient est agé {age} ans, {sex}, : {patient_history},the lesion characters: {lesion_char}'
    prompt = f'<s> [INST] {intro}. J ai  {age} ans, mon sexe est defini comme {sex}. Mes antecedents medicaux sont {patient_history}. J ai {lesion_char}. {prediction}. {action} "[/INST]'


# We generate an answer only where there is a prompt
if st.button('Générer', type="primary"):
    try:
        # # Formatting the prompt
        # prompt =  "<s> [INST] " + prompt +  " [/INST]"
        llm_generation = groq.generate(st.secrets["groq_api_key"], prompt)

        with st.container(border=True):
            st.write(llm_generation)        

    except Exception as err:
        st.exception('Il y a eu une erreur 😢😭')
        print(err)

    