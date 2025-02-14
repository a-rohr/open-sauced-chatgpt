"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from chains.chat_chain import *

if "model" not in st.session_state:
    st.session_state["model"] = assistant_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Picus ChatGPT", page_icon=":robot:")
st.header("Picus ChatGPT")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", key="input", placeholder="Enter something...")
    return input_text


user_input = get_text()

if user_input:
    output = st.session_state["model"].run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
