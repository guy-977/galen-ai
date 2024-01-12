import streamlit as st
import tensorflow as tf
from clarifai.client.model import Model
# from dotenv import load_dotenv
# load_dotenv()


# clarifai_pat = os.getenv('CLARIFAI_PAT')
clarifai_pat = st.secrets["CLARIFAI_PAT"]
CLARIFAI_PAT = clarifai_pat

# Model parameters or mixtral
inference_params = dict(temperature=0.7, max_new_tokens = 200, max_tokens=100, top_k = 50, top_p= 0.95)

def get_prediction(img, Model):
  class_names = [
    'Varicelle', 'Variole de la vache', 'Herpangine', 'Bonne santé', 'Rougeole', 'Variole du singe'
  ]
  img = tf.keras.utils.load_img(img, target_size=(180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = Model.serve(img_array)
  score = tf.nn.softmax(prediction)

  return sorted([(class_names[i], 100 * score[0][i].numpy()) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)

st.set_page_config(page_title='Galen AI | French')
st.title("Galen AI | French")

# Caching the model for faster loading
@st.cache_resource
# laod the model
def load_model(exported_model_path):
    model = tf.saved_model.load(exported_model_path)
    return model
model = load_model('SkinNet')

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
            
        model_prediction = Model("https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
        
        # Take the answer
        full_response = model_prediction.outputs[0].data.text.raw
        with st.container(border=True):
            st.write(full_response)
    except Exception as err:
        st.exception('Il y a eu une erreur 😢😭')
        print(err)

    