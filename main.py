import streamlit as st
import tensorflow as tf
from clarifai.client.model import Model
# from dotenv import load_dotenv
# load_dotenv()


# clarifai_pat = os.getenv('CLARIFAI_PAT')
clarifai_pat = st.secrets["CLARIFAI_PAT"]
CLARIFAI_PAT = clarifai_pat

# Model parameters or mixtral
inference_params = dict(temperature=0.7, max_tokens=200, top_k = 50, top_p= 0.95)

def get_prediction(img, Model):
  class_names = [
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
  ]
  img = tf.keras.utils.load_img(img, target_size=(180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = Model.serve(img_array)
  score = tf.nn.softmax(prediction)

  return sorted([(class_names[i], 100 * score[0][i].numpy()) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)

st.set_page_config(page_title='Chatbot with Mixtral')
st.title("Chatbot with Mixtral")

# Caching the model for faster loading
@st.cache_resource
# laod the model
def load_model(exported_model_path):
    model = tf.saved_model.load(exported_model_path)
    return model
model = load_model('SkinNet')

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Hello 👋, I am Galen (mixtral-8x7B-Instruct-v0_1), I'm here to help you as an AI medical assistant")


#Input 1 patient information: sex & age
age = st.number_input('age', step=1)
sex = st.multiselect('sex', ['male', 'female', 'other'], max_selections=1)

#input 2 patient history
patient_history = st.text_input('patient history')

# input 3 lesion characters
lesion_char = st.text_input('lesion characters')

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
            
        model_prediction = Model("https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
        
        # Take the answer
        full_response = model_prediction.outputs[0].data.text.raw
        with st.container(border=True):
            st.write(full_response)
    except Exception as err:
        st.exception('You must upload an image of the skin lesion!')
        print(err)