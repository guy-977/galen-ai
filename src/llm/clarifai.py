from clarifai.client.model import Model

def generate(clarifai_pat, prompt, inference_params=dict(temperature=0.7, max_tokens=200, top_k = 50, top_p= 0.95)):
    CLARIFAI_PAT = clarifai_pat
    model_prediction = Model("https://clarifai.com/mistralai/completion/models/mixtral-8x7B-Instruct-v0_1").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    full_response = model_prediction.outputs[0].data.text.raw

    print("Clarifai API running Successfully!!")
    return full_response