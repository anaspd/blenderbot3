import tensorflow
import streamlit as st
import os
from streamlit_chat import message as st_message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@st.cache_resource
def get_models():
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/blenderbot-400M-distill", from_tf=True)

    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Your Friend Blender!")


def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )   

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Tap to chat with the bot",
              key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)  