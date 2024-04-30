from groq import Groq

def generate(api_key, prompt, model_name="mixtral-8x7b-32768"):
    client = Groq(
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                }
        ],
        model=model_name,
    )
    print("Groq Cloud API running Successfully!!")
    return chat_completion.choices[0].message.content