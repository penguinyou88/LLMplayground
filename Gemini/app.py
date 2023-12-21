import streamlit as st
import os
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

# Define the main function
def main():
    # create main title
    st.title('Google Gemini Demo')
    st.header("")
    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
        """     
        - The web app provides a demo using the latest Google Gemini Pro Multi-Model Generative Model.
        - User can select 2 modes, the vision model lets you input and image and a text prompt at the same time. 
        - An api key can be generated at https://ai.google.dev/ by logging in with your google account. 
        - Currently, it is fully free to use at a maximum of 60 calls per minute. Have fun! 
        """
        )

        st.markdown("")

    # Create a sidebar for user input
    st.sidebar.title("LLM Settings")

    # Add a radio button to choose between Gemini Pro and Gemini Pro Vision
    model_choice = st.sidebar.radio(
        "Choose a Model",
        ("Gemini Pro", "Gemini Pro Vision"),index=1,
    )
    if model_choice == "Gemini Pro Vision":
        example_load = st.sidebar.button('Load an Example',disabled=False)
    else:
        example_load = st.sidebar.button('Load an Example',disabled=True)

    # Add a slider to choose the temperature
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4)
    top_p = st.sidebar.slider("Top p", min_value=0.0, max_value=1.0, value=0.95)
    output_length = st.sidebar.number_input("Max Output Token", value=1024, max_value=2048)
    generation_config["temperature"] = temperature
    generation_config["top_p"] = top_p
    generation_config["max_output_tokens"] = output_length

    # Add a file uploader to upload an image (only for Gemini Pro Vision)
    if model_choice == "Gemini Pro Vision":
        model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
        if example_load:
            fpath = "./example/example1.jpeg"
            st.image(fpath, caption='Example Image',width=400)
            img_raw = Image.open(fpath)
            img_byte_arr = io.BytesIO()
            img_raw.save(img_byte_arr, format='PNG')
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": img_byte_arr.getvalue()#read_bytes()
                },
                ]
            user_input = st.text_input("Your Prompt:",placeholder="what did you see in the picture? what do you think the objects are made of?")
            if_submitted = st.button('Submit')
            if_submitted = True
        else:
            # Create a text input field for the user to enter their message
            image = st.file_uploader("Upload an Image")
            if image is not None:
                st.image(image, caption='User Uploaded Image',width=400)
                image_parts = [
                    {
                        "mime_type": "image/jpeg",
                        "data": image.getvalue()#read_bytes()
                    },
                    ]
            user_input = st.text_input("Your Prompt:")
            if_submitted = st.button('Submit')
    else:
        model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
        user_input = st.text_input("Your Prompt:")
        if_submitted = st.button('Submit')

    # Generate a response from the model
    if if_submitted:
      with st.status("Running Query to API with Prompt....."):
        if model_choice == "Gemini Pro":
            prompt_parts = [
            user_input
            ]
            response = model.generate_content(prompt_parts)
        else:
            prompt_parts = [
            user_input,
            image_parts[0],
            ]
            response = model.generate_content(prompt_parts)

      # Display the response to the user
      st.write(f"Bot: {response.text}")

# Run the main function
if __name__ == "__main__":
    main()