import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import FAISS
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime

VECTORSTORE_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.json"
CHAT_HISTORY_PATH = "chat_history.json"

st.set_page_config(page_title="Ayurvedic ChatBot", page_icon="🌿")

def load_vectorstore():
    vectorstore = FAISS.load_local(
        VECTORSTORE_INDEX_PATH, 
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
        allow_dangerous_deserialization=True
    )
    with open(METADATA_PATH, "r") as file:
        texts = json.load(file)
    vectorstore.texts = texts
    return vectorstore

def load_chat_history():
    """Load chat history from a JSON file."""
    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Could not decode chat history file.")
            return []
    return []

def save_chat_history():
    """Save current chat to history file if it's a new session."""
    chat_history = load_chat_history()
    if "current_chat" in st.session_state and st.session_state.current_chat:
        chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.current_chat
        })
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(chat_history, f)

def update_chat_history():
    """Update the selected chat with new messages and save the history file."""
    chat_history = load_chat_history()
    if "selected_chat_index" in st.session_state and st.session_state.current_chat:
        selected_chat_index = st.session_state.selected_chat_index
        if 0 <= selected_chat_index < len(chat_history):
            chat_history[selected_chat_index]["messages"] = st.session_state.current_chat
            with open(CHAT_HISTORY_PATH, "w") as f:
                json.dump(chat_history, f)
        else:
            st.warning("Selected chat index is out of range.")
    else:
        st.warning("No chat selected or no current chat to update.")

def get_response(context, question, model):
    if "chat_session" not in st.session_state or st.session_state.chat_session is None:
        st.session_state.chat_session = model.start_chat(history=[])

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    chat_history_context = ""
    for past_query, past_response in st.session_state.current_chat:
        chat_history_context += f"User asked: {past_query}\nAI responded: {past_response}\n\n"

    prompt_template = f"""
    You are a wise Ayurvedic guide who provides clear, holistic answers for health and wellness queries rooted in Ayurveda. When explaining concepts, follow this structure:

Start with a simple definition of the Ayurvedic principle or treatment.
Use an analogy or everyday example to make the concept relatable.
Break down advice into easy-to-understand points:
Dosha: Describe which dosha (Vata, Pitta, or Kapha) is relevant and its characteristics.
Natural Remedies: Suggest safe, practical remedies using everyday herbs and ingredients.
Diet & Lifestyle: Recommend foods, routines, and practices that align with Ayurvedic principles.
Mind-Body Balance: Offer tips for balancing mental, physical, and spiritual health.
Conclude with a friendly, encouraging note that promotes wellness.
Previous Conversations: {chat_history_context}

Current Context: {context} 
User's Question: {question}

Provide answers that are compassionate, clear, and make Ayurveda accessible to all backgrounds.
    """

    try:
        response = st.session_state.chat_session.send_message(prompt_template)
        st.session_state.current_chat.append((question, response.text))
        return response.text

    except AttributeError:
        st.warning("Chat session could not be initialized. Please restart the app.")
    except Exception as e:
        st.warning(e)


def sidebar_ui():
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
    st.session_state.api_key = api_key

    st.sidebar.markdown("### Previous Chats")
    chat_history = load_chat_history()

    # Generate chat options based on the first query in each chat
    chat_options = [
        f"{i+1}. {chat['messages'][0][0]}" if chat['messages'] else f"{i+1}. [No query]" 
        for i, chat in enumerate(chat_history)
    ]
    
    # Allow selecting a previous chat
    selected_chat = st.sidebar.selectbox("Select a previous chat to view", options=["Start a New Chat"] + chat_options, index=0)
    if selected_chat == "Start a New Chat":
        st.session_state.current_chat = []  # Clear current chat history
        st.session_state.new_chat = True
        st.session_state.selected_chat = None
        st.session_state.chat_session = None
    else:
        selected_chat_index = int(selected_chat.split(".")[0]) - 1
        st.session_state.selected_chat = chat_history[selected_chat_index]["messages"]
        st.session_state.selected_chat_index = selected_chat_index
        st.session_state.current_chat = st.session_state.selected_chat.copy()
        st.session_state.new_chat = False


def working_process(generation_config):
    api_key = st.session_state.api_key

    if not api_key:
        st.warning("Please enter a valid API key in the sidebar to continue.")
        return

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return

    system_instruction = "You are a helpful document answering assistant."
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)
    vectorstore = st.session_state['vectorstore']

    # Display the current chat history
    if "current_chat" in st.session_state:
        for user_query, ai_response in st.session_state.current_chat:
            with st.chat_message("Human"):
                st.markdown(user_query)
            if ai_response:  # Check if the response is not None
                with st.chat_message("AI"):
                    st.markdown(ai_response)

    # Ensure input box is available for new queries, below the chat messages
    user_query = st.chat_input("Enter Your Query....")
    if user_query and user_query.strip():
        # Add user query to the chat history before processing it
        st.session_state.current_chat.append((user_query, None))  # Append without a response yet

        with st.chat_message("Human"):
            st.markdown(user_query)  # Show user input in chat

        try:
            # Use the relevant context from the vectorstore
            relevant_content = vectorstore.similarity_search(user_query, k=10)
            result = get_response(relevant_content, user_query, model)

            # Update the last entry with the AI response
            st.session_state.current_chat[-1] = (user_query, result)  
            
            # Display the AI response in a new block
            with st.chat_message("AI"):
                st.markdown(result)  

        except Exception as e:
            st.warning(f"Error generating response: {e}")

    # Save or update chat history depending on the chat type
    if st.session_state.new_chat:
        save_chat_history()
    else:
        update_chat_history()



    # If revisiting a previous chat, ensure the input box is still available
    if "selected_chat" in st.session_state and st.session_state.selected_chat:
        st.session_state.new_chat = False  # Set to false when selecting a previous chat

generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8000,
}

def main():
    load_dotenv()
    st.header("🌿 Ayurvedic ChatBot")

    if "vectorstore" not in st.session_state:
        with st.spinner("Loading preprocessed data..."):
            st.session_state.vectorstore = load_vectorstore()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    sidebar_ui()

    if st.session_state.vectorstore is not None:
        working_process(generation_config)

if __name__ == "__main__":
    main()



