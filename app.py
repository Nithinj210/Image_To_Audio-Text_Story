# Imports
import os
import streamlit as st
import requests
from transformers import pipeline
import openai

# Suppressing all warnings
import warnings
warnings.filterwarnings("ignore")

# Image-to-text
def img2txt(url):
    print("Initializing captioning model...")
    captioning_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    print("Generating text from the image...")
    text = captioning_model(url, max_new_tokens=20)[0]["generated_text"]
    
    print(text)
    return text
# Text-to-story
def txt2story(img_text, top_k, top_p, temperature):

    from Secrete_key import TOGETHER_API_KEY
    import os
    import json  # Used for potential logging
    import requests

    # Set API key environment variable
    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    # Create headers with authorization token
    headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}

    # Prepare data for API request
    data = {
        "model": "togethercomputer/llama-2-70b-chat",
        "messages": [
            {"role": "system", "content": '''As an experienced short story writer, write story title and then create a meaningful story influenced by provided words. 
            Ensure stories conclude positively within 100 words. Remember the story must end within 100 words''', "temperature": temperature},
            {"role": "user", "content": f"Here is input set of words: {img_text}", "temperature": temperature}
        ],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature
    }

    # Send API request and handle potential errors
    try:
        response = requests.post("https://api.together.xyz/inference", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        # Extract story text (assuming successful response)
        story = response.json()["output"]["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        # Handle request errors (e.g., network issues)
        print(f"Error making API request: {e}")
        story = None  # Indicate error or return default value
    except (KeyError, JSONDecodeError) as e:
        # Handle potential issues with response data
        print(f"Error processing API response: {e}")
        story = None  # Indicate error or return default value

    # Optional: Log API request details for debugging
    # You can uncomment and customize this section if needed
    # with open("api_requests.log", "a") as f:
    #     log_data = {"image_text": img_text, "data": data, "response": response.json()}
    #     json.dump(log_data, f)

    return story
# Text-to-speech
def txt2speech(text):
    from Secrete_key2 import HUGGINGFACEHUB_API_TOKEN
    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    print("Initializing text-to-speech conversion...")
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
    payloads = {'inputs': text}

    response = requests.post(API_URL, headers=headers, json=payloads)
    
    with open('audio_story.mp3', 'wb') as file:
        file.write(response.content)
# Streamlit web app main function
def main():
    st.set_page_config(page_title="🎨 Image-to-Audio Story 🎧", page_icon="🖼️")
    st.title("Turn the Image into Audio Story")

    # Allows users to upload an image file
    uploaded_file = st.file_uploader("# 📷 Upload an image...", type=["jpg", "jpeg", "png"])

    # Parameters for LLM model (in the sidebar)
    st.sidebar.markdown("# LLM Inference Configuration Parameters")
    top_k = st.sidebar.number_input("Top-K", min_value=1, max_value=100, value=5)
    top_p = st.sidebar.number_input("Top-P", min_value=0.0, max_value=1.0, value=0.8)
    temperature = st.sidebar.number_input("Temperature", min_value=0.1, max_value=2.0, value=1.5)

    if uploaded_file is not None:
        # Reads and saves uploaded image file
        bytes_data = uploaded_file.read()
        with open("uploaded_image.jpg", "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption='🖼️ Uploaded Image', use_column_width=True)

        # Initiates AI processing and story generation
        with st.spinner("## 🤖 AI is at Work! "):
            scenario = img2txt("uploaded_image.jpg")  # Extracts text from the image
            story = txt2story(scenario, top_k, top_p, temperature)  # Generates a story based on the image text, LLM params
            txt2speech(story)  # Converts the story to audio

            st.markdown("---")
            st.markdown("## 📜 Image Caption")
            st.write(scenario)

            st.markdown("---")
            st.markdown("## 📖 Story")
            st.write(story)

            st.markdown("---")
            st.markdown("## 🎧 Audio Story")
            st.audio("audio_story.mp3")

if __name__ == '__main__':
    main()
# Credits
st.markdown("### Credits")
st.caption('''
            Made by Nithin John 
            Utilizes Image-to-text, Text Generation, Text-to-speech Transformer Models\n
            Gratitude to Streamlit, 🤗 Spaces for Deployment & Hosting
            ''')