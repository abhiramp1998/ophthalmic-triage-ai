# agent.py (Definitive, Complete, and Deployable Version)

import streamlit as st
import sys
import os
import json
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
MAX_QUESTIONS = 3 # Set the maximum number of follow-up questions

# --- ROBUST PATH DEFINITIONS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_scripts")
KANSKI_PDF_PATH = os.path.join(DATA_DIR, "Kanski_Clinical_Ophthalmology.pdf")
NICE_JSON_PATH = os.path.join(DATA_DIR, "nice_nhs_ophthalmology_kb.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- KNOWLEDGE BASE PREPARATION LOGIC ---
@st.cache_resource
def create_knowledge_base():
    """
    Loads documents, chunks them, creates embeddings, and saves them to a FAISS index.
    This is designed to run once when the Streamlit app is first launched on a clean environment.
    """
    documents, json_docs_count = [], 0
    try:
        with open(KANSKI_PDF_PATH, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text: documents.append({"text": text, "metadata": {"source": "Kanski", "page": page_num + 1}})
        st.sidebar.success(f"Loaded {len(documents)} pages from Kanski PDF.")
    except Exception as e: st.sidebar.error(f"Error loading Kanski PDF: {e}")

    try:
        with open(NICE_JSON_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                if 'content' in item and item['content']:
                    documents.append({"text": item['content'], "metadata": {"source": "NICE/NHS", "title": item.get('title', 'N/A')}}); json_docs_count += 1
            st.sidebar.success(f"Loaded {json_docs_count} articles from NICE/NHS JSON.")
    except Exception as e: st.sidebar.warning(f"Could not load NICE/NHS JSON: {e}")

    if not documents: st.error("No knowledge base documents found."); st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts, all_metadata = [], []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks: split_texts.append(chunk); all_metadata.append(doc["metadata"])

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=split_texts, embedding=embeddings_model, metadatas=all_metadata)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return True

# --- AGENT DEFINITIONS ---

class RetrieverAgent:
    def __init__(self):
        self.embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings_model, allow_dangerous_deserialization=True)
    def retrieve_context(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)

class RelevanceCheckAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            Analyze the user's query. Is it related to eyes or vision problems? Respond with only the word "yes" or "no".

            USER QUERY: "{user_query}"
            """,
            input_variables=["user_query"]
        )
    def check_relevance(self, user_query):
        formatted_prompt = self.prompt.format(user_query=user_query)
        response = self.llm.invoke(formatted_prompt)
        decision = response.content.strip().lower()
        print(f"Relevance decision: {decision}")
        return "yes" in decision

class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are a triage orchestrator. Your task is to decide if enough clinical information has been gathered.
            A complete triage needs to know about: Vision Status, Pain Level, Discharge, Trauma, and Duration.
            Review the conversation history. Has each of these five topics been discussed?
            
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            
            If one or more topics are still missing, respond with ONLY the word "ask_question".
            If all five topics have been discussed, respond with ONLY the word "provide_summary".
            """,
            input_variables=["conversation_history"]
        )
    def route(self, conversation_history):
        formatted_prompt = self.prompt.format(conversation_history=conversation_history)
        response = self.llm.invoke(formatted_prompt)
        decision = response.content.strip().replace("\"", "")
        print(f"Router decision: {decision}")
        return decision

class QuestionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an AI Triage Assistant. Your job is to ask the next follow-up question.
            Read the conversation history to see what has already been asked. Do not repeat questions.
            Your question must be simple and easy for a non-medical person to understand.

            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>

            Based on the history, what is the single best, simple question to ask next?
            """,
            input_variables=["conversation_history"]
        )
    def generate_question(self, conversation_history):
        formatted_prompt = self.prompt.format(conversation_history=conversation_history)
        response = self.llm.invoke(formatted_prompt)
        return response.content

class SummaryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an AI Triage Assistant. Your job is to provide a final summary.
            Carefully read the entire conversation history.
            
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            
            Based ONLY on the information in the conversation history, provide a Triage Summary, an Urgency Recommendation (URGENT, SEMI-URGENT, or ROUTINE), and a Justification.
            Format your response exactly as follows with markdown for bolding. Do not add any other text.

            **TRIAGE SUMMARY:**
            [Your summary here]

            **URGENCY RECOMMENDATION:**
            [URGENT/SEMI-URGENT/ROUTINE]

            **JUSTIFICATION:**
            [Your justification here]
            """,
            input_variables=["conversation_history"]
        )
    def generate_summary(self, conversation_history):
        formatted_prompt = self.prompt.format(conversation_history=conversation_history)
        response = self.llm.invoke(formatted_prompt)
        return response.content

# --- THE MAIN STREAMLIT APPLICATION ---

st.set_page_config(page_title="Emergency Triage Assistant", page_icon="👩‍⚕️")
st.title("Emergency Triage Assistant")

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an AI Triage Assistant. Please describe your eye symptoms."}]
    st.session_state.question_count = 0
    st.session_state.finished = False
    st.rerun()

if not os.path.exists(FAISS_INDEX_PATH):
    with st.sidebar, st.spinner("Setting up knowledge base for the first time..."):
        create_knowledge_base()

# Load agents using the securely stored API key
@st.cache_resource
def load_agents():
    # Check if the secret key is available in Streamlit's secrets
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("Google API Key not found. Please add it to your Streamlit Secrets.")
        st.stop()
        
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
        retriever = RetrieverAgent()
        relevance_checker = RelevanceCheckAgent(llm)
        router = RouterAgent(llm)
        question_generator = QuestionAgent(llm)
        summary_generator = SummaryAgent(llm)
        return retriever, relevance_checker, router, question_generator, summary_generator
    except Exception as e:
        st.error(f"Failed to initialize models. Error: {e}")
        return None, None, None, None, None

# Load all agents
retriever, relevance_checker, router, question_generator, summary_generator = load_agents()

# Initialize session state if not already done
if "messages" not in st.session_state:
    reset_conversation()

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

# Main orchestrator logic
if not st.session_state.get("finished", False):
    if prompt := st.chat_input("Describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # On the very first user message, check for relevance
                if len(st.session_state.messages) == 2:
                    if not relevance_checker.check_relevance(prompt):
                        response = "I am an ophthalmology triage assistant and can only help with eye-related problems. Please restart the conversation with an eye symptom."
                        st.session_state.finished = True # End conversation
                    else:
                        st.session_state.retrieved_docs = retriever.retrieve_context(prompt, k=5)
                        history = "\n".join([f"- {msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
                        response = question_generator.generate_question(history)
                        st.session_state.question_count += 1
                else: # This is a follow-up answer, so we route it
                    history = "\n".join([f"- {msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
                    
                    if st.session_state.question_count >= MAX_QUESTIONS:
                        next_step = "provide_summary"
                    else:
                        next_step = router.route(history)
                    
                    if "ask_question" in next_step:
                        response = question_generator.generate_question(history)
                        st.session_state.question_count += 1
                    else: # Includes "provide_summary" and any fallback
                        response = summary_generator.generate_summary(history)
                        st.session_state.finished = True
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if st.session_state.get("finished", False):
    st.button("Start New Triage", on_click=reset_conversation)