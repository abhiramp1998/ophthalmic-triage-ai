# agent.py (Definitive, Complete, and Verified Version for Deployment)

import streamlit as st
import sys
import os
import json
import PyPDF2
import speech_recognition as sr
import re
import base64
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
MAX_QUESTIONS = 8 # Set the maximum number of follow-up questions

# --- ROBUST PATH DEFINITIONS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_scripts")
KANSKI_PDF_PATH = os.path.join(DATA_DIR, "Kanski_Clinical_Ophthalmology.pdf")
NICE_JSON_PATH = os.path.join(DATA_DIR, "nice_nhs_ophthalmology_kb.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- KNOWLEDGE BASE PREPARATION LOGIC ---
@st.cache_resource
def create_knowledge_base():
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

class QueryRefinementAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an expert in clinical ophthalmology. Your task is to take a full conversation history
            and transform it into a highly specific, detailed query for a medical knowledge base search.
            Focus on synthesizing all symptoms, risk factors, and user statements into a clinical query.
            For example, if the history mentions "red eyes", "itching", and "contact lenses", a good refined query would be:
            "Clinical evaluation, differential diagnosis, and management of red, itchy eyes in a contact lens wearer, considering allergic conjunctivitis and infectious keratitis."

            CONVERSATION HISTORY:
            {conversation_history}

            REFINED & DETAILED QUERY FOR KNOWLEDGE BASE:
            """,
            input_variables=["conversation_history"]
        )
    def refine_query(self, conversation_history):
        formatted_prompt = self.prompt.format(conversation_history=conversation_history)
        response = self.llm.invoke(formatted_prompt)
        refined_query = response.content.strip()
        print(f"Refined Query: {refined_query}")
        return refined_query

class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are a senior clinical triage orchestrator. Your task is to analyze a conversation and decide if it should continue or be summarized.

            Review the conversation history for key clinical information (Vision Status, Pain, Discharge, Trauma, and Duration).
            
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            
            DECISION LOGIC:
            1.  First, assess your confidence. Do you have a clear clinical picture?
            2.  If the conversation reveals a clear, high-urgency emergency (e.g., chemical injury, sudden total vision loss, severe pain with vision loss), your confidence is high. Respond with ONLY "provide_summary".
            3.  If the conversation reveals a clear, low-urgency issue (e.g., mild itching with no other symptoms), your confidence is high. Respond with ONLY "provide_summary".
            4.  If the situation is ambiguous or key details are still missing (e.g., pain is mentioned but severity is unknown; redness is mentioned but vision status is unknown), your confidence is low. You must gather more information. Respond with ONLY "ask_question".

            Your response must be either "provide_summary" or "ask_question".
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
            
            IMPORTANT SAFETY RULE: If the conversation mentions a chemical injury (e.g., "bleach", "acid", "chemical"), severe trauma, or a sudden and total loss of vision, the URGENCY RECOMMENDATION MUST be URGENT.

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

# --- BROWSER-BASED TEXT-TO-SPEECH FUNCTION ---
def speak(text: str):
    """
    Generates and plays audio using the browser's built-in SpeechSynthesis API.
    """
    # Clean the text for JavaScript: escape backticks, single quotes, newlines, and asterisks
    safe_text = text.replace('`', '\\`').replace("'", "\\'").replace("\n", " ").replace("*", "")
    js_code = f"""
        <script>
            // Check if speech synthesis is already running, if so, cancel it
            if (speechSynthesis.speaking) {{
                speechSynthesis.cancel();
            }}
            // Create a new utterance
            const utterance = new SpeechSynthesisUtterance(`{safe_text}`);
            utterance.lang = 'en-US';
            // Speak the text
            speechSynthesis.speak(utterance);
        </script>
    """
    st.components.v1.html(js_code, height=0, scrolling=False)

# --- THE MAIN STREAMLIT APPLICATION ---
st.set_page_config(page_title="Ophthalmic Triage AI", page_icon="👁️")
st.title("👁️ Ophthalmic Triage AI")
st.subheader("An AI-powered assistant for assessing eye-related symptoms.")
st.caption("This is an academic proof-of-concept and does not provide medical advice.")

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an AI assistant designed to help with the initial triage of eye symptoms. Please describe your main concern.", "play_audio": False}]
    st.session_state.question_count = 0
    st.session_state.finished = False
    st.session_state.is_recording = False
    st.session_state.transcribed_text = None
    st.session_state.input_mode = "text"
    st.session_state.turn = "user"
    st.session_state.retrieved_docs = []
    st.rerun()

with st.sidebar:
    st.header("About OphthalmicTriageAI")
    st.markdown("""
    This application is a dissertation project demonstrating a multi-agent RAG 
    (Retrieval-Augmented Generation) system for clinical triage. 
    The AI's knowledge is grounded in clinical texts to ensure its questions 
    and summaries are relevant and safe.
    """)
    st.warning("**Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")

if not os.path.exists(FAISS_INDEX_PATH):
    with st.sidebar, st.spinner("Setting up knowledge base for the first time..."):
        create_knowledge_base()

api_key = st.secrets.get("GOOGLE_API_KEY")

@st.cache_resource
def load_agents(api_key):
    if not api_key: st.error("Google API Key not found. Please add it to your Streamlit Secrets."); st.stop()
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)
        retriever = RetrieverAgent()
        relevance_checker = RelevanceCheckAgent(llm)
        query_refiner = QueryRefinementAgent(llm)
        router = RouterAgent(llm)
        question_generator = QuestionAgent(llm)
        summary_generator = SummaryAgent(llm)
        return retriever, relevance_checker, query_refiner, router, question_generator, summary_generator
    except Exception as e:
        st.error(f"Failed to initialize models. Error: {e}"); return None, None, None, None, None, None

retriever, relevance_checker, query_refiner, router, question_generator, summary_generator = load_agents(api_key)

if "messages" not in st.session_state:
    reset_conversation()

# Message Display Loop with Audio Logic
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("play_audio_autoplay", False):
            speak(message["content"])
            st.session_state.messages[i]["play_audio_autoplay"] = False

# Main Orchestrator Logic
prompt = None

if st.session_state.turn == "user" and not st.session_state.get("finished", False):
    if st.session_state.get("is_recording"):
        with st.spinner("🔴 Recording... Speak now!"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=10, phrase_time_limit=30)
                    st.info("Transcribing...")
                    text = r.recognize_google(audio)
                    st.session_state.transcribed_text = text
                except Exception as e: st.warning("Could not process audio. Please try again.")
            st.session_state.is_recording = False
            st.rerun()
    else:
        if st.session_state.get("transcribed_text"):
            prompt = st.session_state.transcribed_text
            st.session_state.transcribed_text = None
            st.session_state.input_mode = "voice"
        else:
            col1, col2 = st.columns([7, 1])
            with col1:
                text_prompt = st.chat_input("Describe your symptoms...")
                if text_prompt:
                    prompt = text_prompt
                    st.session_state.input_mode = "text"
            with col2:
                if st.button("🎤", key="speak_button"):
                    st.session_state.is_recording = True
                    st.rerun()

elif st.session_state.turn == "assistant" and not st.session_state.get("finished", False):
    if st.button("▶️ Continue"):
        st.session_state.turn = "user"
        st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "play_audio": False})
    
    if st.session_state.input_mode == "voice":
        st.session_state.turn = "assistant"
    
    with st.spinner("Thinking..."):
        history = "\n".join([f"- {msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        
        # Refine and Retrieve is now done at EVERY turn of the conversation.
        refined_query = query_refiner.refine_query(history)
        st.session_state.retrieved_docs = retriever.retrieve_context(refined_query, k=5)
        
        if len(st.session_state.messages) == 2: # First user message
            if not relevance_checker.check_relevance(prompt):
                response = "I am an ophthalmology triage assistant and can only help with eye-related problems. Please restart the conversation with an eye symptom."
                st.session_state.finished = True
            else:
                response = question_generator.generate_question(history)
                st.session_state.question_count += 1
        else: # Follow-up messages
            if st.session_state.question_count >= MAX_QUESTIONS:
                next_step = "provide_summary"
            else:
                next_step = router.route(history)
            
            if "ask_question" in next_step:
                response = question_generator.generate_question(history)
                st.session_state.question_count += 1
            else:
                response = summary_generator.generate_summary(history)
                st.session_state.finished = True
        
        if st.session_state.finished and "**Next Step:**" not in response:
            if "URGENT" in response:
                response += "\n\n**Next Step:** This may indicate a serious condition. Please go to your nearest Accident & Emergency (A&E) department immediately."
            elif "SEMI-URGENT" in response:
                response += "\n\n**Next Step:** Please contact an ophthalmologist or optometrist for an appointment within the next 24-48 hours."
            else:
                response += "\n\n**Next Step:** Please book a routine appointment with your optometrist at your convenience."
    
    should_autoplay = st.session_state.get("input_mode") == "voice" and not st.session_state.get("finished", False)
    st.session_state.messages.append({
        "role": "assistant", "content": response, "play_audio_autoplay": should_autoplay, "is_summary": st.session_state.get("finished", False)
    })
    st.rerun()

if st.session_state.get("finished", False):
    st.button("Start New Triage", on_click=reset_conversation)

    summary_message = ""
    for msg in reversed(st.session_state.messages):
        if msg.get("is_summary"):
            summary_message = msg["content"]; break
    
    if summary_message:
        st.markdown("---")
        if st.button("🔊 Listen to the Summary"):
            speak(summary_message)
        st.markdown("---")

    full_history = "Ophthalmic Triage AI - Conversation Summary\n" + "="*40 + "\n\n"
    for msg in st.session_state.messages:
        full_history += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

    st.download_button(label="📥 Download Triage Summary", data=full_history, file_name="triage_summary.txt", mime="text/plain")

    with st.expander("View Clinical Sources Used for this Triage"):
        if st.session_state.get("retrieved_docs"):
            for i, doc in enumerate(st.session_state.get("retrieved_docs", [])):
                st.info(f"Source {i+1}: (Kanski, Page {doc.metadata.get('page', 'N/A')})"); st.write(doc.page_content)
        else:
            st.write("No sources were retrieved for this conversation.")