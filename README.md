# galen-ai

 Demo for skin lesiosn detection using deep learning CNN model and explain predections using LLMs for end users

 ## set up
 install dependancies
 ```bash
pip install requirements.txt
```
Create `.streamlit/secrets.toml` file to store API keys
if you are going to use Groq cloud (default option), define the API key in the `secrets.toml` file as follows:
```
GROQ_API_KEY= "YOUR OWN API KEY"
```
if you are going to use Clarifai API (requries manual modification of the code) define it as follows:
```
CLARIFAI_PAT= "YOUR OWN TOKEN"
```
run the streamlit app
```bash
streamlit run main.py
```
