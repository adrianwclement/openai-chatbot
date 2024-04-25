from ConversationManager import Convsersation_Manager
import streamlit as st
import os


# Set API Key
api_key = os.environ["OPENAI_API_KEY2"]


### Streamlit Implementation ###

st.title("Streamlit Chatbot!")

# Check if key exists in session state to prevent reinitialization
if "chat" not in st.session_state:
    st.session_state["chat"] = Convsersation_Manager(api_key)

chat_manager = st.session_state["chat"]


# Sidebar Options
st.sidebar.header("Chatbot Options")
message_token_budget = st.sidebar.slider("Max Tokens per Message", min_value=10, max_value=500, value=50)
chat_temperature = st.sidebar.slider("Chat Temperature", min_value=0.0, max_value=1.0, value=0.07, step=0.01)

# Pick and set desired chat bot persona
chat_persona = st.sidebar.selectbox("Chat Persona", ["Normal", "Sassy", "Angry", "Thoughtful", "Custom"])

if chat_persona == "Normal":
    chat_manager.set_persona("normal_assistant")
elif chat_persona == "Sassy":
    chat_manager.set_persona("sassy_assistant")
elif chat_persona == "Angry":
    chat_manager.set_persona("angry_assistant")
elif chat_persona == "Thoughtful":
    chat_manager.set_persona("thoughtful_assistant")
elif chat_persona == "Custom":
    custom_message = st.sidebar.text_area("Custom chat persona")
    if st.sidebar.button("Set custom chat persona"):
        chat_manager.set_custom_system_message(custom_message)

# Reset button in sidebar
if st.sidebar.button("Reset Conversation History", on_click=chat_manager.reset_conversation_history):
    st.session_state["conversation_history"] = chat_manager.conversation_history

# Recall conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = chat_manager.conversation_history

conversation_history = st.session_state["conversation_history"]


# Get chat input from the user
user_input = st.chat_input("Write a message")

# Call the chat manager to get a response from the AI chat bot
if user_input:
    response = chat_manager.chat_completion(user_input, temperature=chat_temperature, max_tokens=message_token_budget)

# Display the conversation history
for message in conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
