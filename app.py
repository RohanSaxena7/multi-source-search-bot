import streamlit as st
#we will use groq to use open source models
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper #wrappers
#to be able to search from the internet we will use DuckDuckGo
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
#importing agents
from langchain.agents import AgentType, initialize_agent
#for communication of all these tools within themselves
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv #we will ask the user to input their own key hence we will NOT load the groq api key

#enabling langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

##Arxiv and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max= 250)
arxiv = ArxivQueryRun(api_wrapper= arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max= 250)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)

#duckduckgo
search = DuckDuckGoSearchRun(name= "Search")

#Creating the streamlit app
#we will use Streamlit callback handler to display the thoughts and actions of an agent in an interactive streamlit app
st.title("🔎Search with Langchain🔍")
st.write("""
         Here we are using Wikipedia💬, Arxiv📃 and DuckDuckGO🦆 tools to answer your query.\n
         We are using StreamlitCallbackHandler🤔 to display the thoughts and actions of an agent in an interactive streamlit app.
         """)

#Asking the user to input their key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type = "password")

#to make sure that the entire conversation happens with the chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role":"assistant", "content": "Hi, I am a Chatbot that can search the web. How can I help you today?"},
        
    ]
#for every message we will traverse this
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
#if there is a prompt and a api_key, we append it to session state 
# Place the chat input outside the conditional block
prompt = st.chat_input("What is the capital of India?", key="main_chat_input")

# Then conditionally process it
if api_key and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
        
    #creating llm
    llm = ChatGroq(api_key= api_key, model_name = "Llama3-8b-8192", streaming= True)
    tools = [search, arxiv, wiki]
        
    #converting tools to agents
    search_agent = initialize_agent(tools= tools, llm= llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True) #if any errors encountered, parse them
    #difference-zero_shot does not consider chat history and makes input based on current action and chat_zero_shot considers chat history to answer
        
    #conversation
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts= False) #we will not see every thought of the agent
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        #append it to messages
        st.session_state.messages.append({'role':"assistant", "content": response})
        st.write(response)