import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# model
@st.cache_resource
def load_model():
    model_id = 'prompthero/openjourney'

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device=device)

    return pipe

# app
st.set_page_config(page_title="Image Generator", page_icon="✨", layout="wide")
st.title('Image Generator')

prompt = st.text_area(
    'Enter your prompt',
    placeholder='Describe what image you want to generate',
    key='prompt')

pipe = load_model()
generate_button = st.button(
    'Generate Image',
    key='generate_button',
    use_container_width=True,
    type='primary',
)
#image generator
st.header('Generated image')
if not generate_button:
    st.write("Click the 'Generate Image button' to generate an image")
else:
    try:

        with st.spinner('Generating image...'):
            image = pipe(
                prompt=prompt).images[0]
            resized_image = image.resize((600, 400))
            st.image(resized_image, caption="Generated Image", width="stretch")

    except Exception as e:
        st.error(e)