# OphthalmicTriageAI - Main Application File
# This script powers the Streamlit user interface and orchestrates the multi-agent AI system.

import streamlit as st
import sys
import os
import json
import PyPDF2
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import re
import math
import html
from typing import List, Dict, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# This constant defines the maximum number of follow-up questions the assistant can ask
# before it must provide a summary. This acts as a safety rail to prevent looping conversations.
MAX_QUESTIONS = 8

# Define the file paths for the knowledge base documents. Using os.path.join ensures
# this works correctly on any operating system (Windows, macOS, Linux).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_scripts")
KANSKI_PDF_PATH = os.path.join(DATA_DIR, "Kanski_Clinical_Ophthalmology.pdf")
NICE_JSON_PATH = os.path.join(DATA_DIR, "nice_nhs_ophthalmology_kb.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")


# This function builds the knowledge base from the source documents.
# The @st.cache_resource decorator is crucial: it tells Streamlit to run this
# expensive process only once and then cache the result, so the app loads quickly
# on subsequent runs.
@st.cache_resource
def create_knowledge_base():
    documents, json_docs_count = [], 0
    try:
        with open(KANSKI_PDF_PATH, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    documents.append({"text": text, "metadata": {"source": "Kanski", "page": page_num + 1}})
        st.sidebar.success(f"Loaded {len(documents)} pages from Kanski PDF.")
    except Exception as e:
        st.sidebar.error(f"Error loading Kanski PDF: {e}")

    try:
        with open(NICE_JSON_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                if 'content' in item and item['content']:
                    documents.append({
                        "text": item['content'],
                        "metadata": {"source": "NICE/NHS", "title": item.get('title', 'N/A')}
                    })
                    json_docs_count += 1
            st.sidebar.success(f"Loaded {json_docs_count} articles from NICE/NHS JSON.")
    except Exception as e:
        st.sidebar.warning(f"Could not load NICE/NHS JSON: {e}")

    if not documents:
        st.error("No knowledge base documents found. Please check your data_scripts folder.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts, all_metadata = [], []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            split_texts.append(chunk)
            all_metadata.append(doc["metadata"])

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=split_texts, embedding=embeddings_model, metadatas=all_metadata)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return True

# --- Agent Definitions ---
# Each class below represents a specialized AI agent with a specific role in the triage process.

class RetrieverAgent:
    """This agent is responsible for searching the knowledge base."""
    def __init__(self):
        self.embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings_model, allow_dangerous_deserialization=True)
    def retrieve_context(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)

class RelevanceCheckAgent:
    """This agent acts as a gatekeeper, ensuring the user's query is on-topic."""
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
        # Hardening: require exact "yes" for higher precision
        return decision == "yes"

class QueryRefinementAgent:
    """This agent rewrites the user's simple query into a more detailed, clinical query to improve search results."""
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an expert in clinical ophthalmology. Your task is to take a full conversation history
            and transform it into a highly specific, detailed query for a medical knowledge base search.
            The history may include patient profile information like age, sex, and pre-existing conditions. Use this information.
            Focus on synthesizing all symptoms, risk factors, and user statements into a clinical query.
            For example, if the history mentions "red eyes", "itching", "contact lenses", and the user is a 25-year-old diabetic, a good refined query would be:
            "Clinical evaluation, differential diagnosis, and management of red, itchy eyes in a 25-year-old diabetic contact lens wearer, considering allergic conjunctivitis and infectious keratitis."

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
    """This agent is the main orchestrator, deciding whether to ask another question or provide a summary.
    This version is context-aware, using retrieved documents to inform its decision, as described in the dissertation."""
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are a senior clinical triage orchestrator. Your task is to analyze a conversation and retrieved clinical context, then decide if the conversation should continue or be summarized.

            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>

            <RETRIEVED_CONTEXT>
            {retrieved_context}
            </RETRIEVED_CONTEXT>

            DECISION LOGIC:
            1. First, assess your confidence based on BOTH the history and the retrieved context. Do you have a clear clinical picture?
            2. If the context or history reveals a clear, high-urgency emergency (e.g., chemical injury, sudden total vision loss, retinal detachment), your confidence is high. Respond with ONLY "provide_summary".
            3. If the context and history reveal a clear, low-urgency issue (e.g., mild itching supporting allergy), your confidence is high. Respond with ONLY "provide_summary".
            4. If the situation is still ambiguous or key details are missing, your confidence is low. You must gather more information. Respond with ONLY "ask_question".

            Your response must be either "provide_summary" or "ask_question".
            """,
            input_variables=["conversation_history", "retrieved_context"]
        )

    def route(self, conversation_history: str, retrieved_context: str) -> str:
        """Analyzes history and context to decide the next step."""
        formatted_prompt = self.prompt.format(
            conversation_history=conversation_history,
            retrieved_context=retrieved_context
        )
        response = self.llm.invoke(formatted_prompt)
        decision_raw = response.content.strip().lower()
        
        # Hardening to ensure a valid output
        if "provide_summary" in decision_raw:
            return "provide_summary"
        if "ask_question" in decision_raw:
            return "ask_question"
            
        # Fallback to asking another question if the output is ambiguous
        return "ask_question"

class QuestionAgent:
    """This agent's job is to formulate the next follow-up question in simple, user-friendly language."""
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an AI Triage Assistant. Your job is to ask the next follow-up question.
            Read the conversation history to see what has already been asked. Do not repeat questions.
            The history begins with the patient's profile. Use this context to ask more relevant questions.
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
    """This agent provides the final, structured triage summary, recommendation, and justification."""
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""
            You are an AI Triage Assistant. Your job is to provide a final summary based on the patient profile and conversation.
            Carefully read the entire conversation history, including the initial patient profile.
            
            <CONVERSATION_HISTORY>
            {conversation_history}
            </CONVERSATION_HISTORY>
            
            IMPORTANT SAFETY RULE: If the conversation mentions a chemical injury (e.g., "bleach", "acid", "chemical"), severe trauma, or a sudden and total loss of vision, the URGENCY RECOMMENDATION MUST be URGENT.

            Based ONLY on the information in the conversation history, provide a Triage Summary, an Urgency Recommendation (URGENT, SEMI-URGENT, or ROUTINE), and a Justification.
            Format your response exactly as follows with markdown for bolding. Do not add any other text.

            **TRIAGE SUMMARY:**
            [Your summary here, incorporating the patient's age, sex, and conditions where relevant.]

            **URGENCY RECOMMENDATION:**
            [URGENT/SEMI-URGENT/ROUTINE]

            **JUSTIFICATION:**
            [Your justification here, referencing the patient's profile and symptoms.]
            """,
            input_variables=["conversation_history"]
        )
    def generate_summary(self, conversation_history):
        formatted_prompt = self.prompt.format(conversation_history=conversation_history)
        response = self.llm.invoke(formatted_prompt)
        return response.content

# --- Helper Functions ---

def text_to_audio_autoplay(text: str):
    """Converts text to speech and plays it automatically for conversational flow."""
    try:
        clean_text = re.sub(r'\*\*', '', text) # Removes markdown for cleaner speech
        tts = gTTS(text=clean_text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
        st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred in text-to-speech autoplay: {e}")

def extract_urgency(markdown_text: str):
    """
    Extracts URGENCY RECOMMENDATION (URGENT/SEMI-URGENT/ROUTINE) from the model's
    structured markdown block. Returns one of the three strings or None if not found.
    This is more robust than simple string checking.
    """
    match = re.search(
        r"\*\*URGENCY RECOMMENDATION:\*\*\s*(URGENT|SEMI-URGENT|ROUTINE)",
        markdown_text,
        flags=re.IGNORECASE
    )
    return match.group(1).upper() if match else None

# --- Advanced XAI Layer Helper Functions ---
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOP = {
    "the","and","for","with","that","this","from","your","have","been","over","more","than",
    "you","are","was","were","has","had","its","into","onto","about","but","not","any","can",
    "only","may","might","will","would","should","could","please","within","next","days",
    "hours","week","weeks","day","hour","min","mins","of","to","in","on","at","by","as","or",
    "is","be","a","an","it","we","our","they","them","their","there","here","also","than",
    "then","when","what","which","who","how","why"
}

def _tokenize(txt: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(txt or "")]

def _count_words(texts: List[str]) -> Tuple[Dict[str,int], Dict[str,int], int]:
    tf, df, N = {}, {}, len(texts)
    for t in texts:
        toks = _tokenize(t)
        seen=set()
        for w in toks:
            if w in _STOP: continue
            tf[w] = tf.get(w,0)+1
            if w not in seen:
                df[w] = df.get(w,0)+1
                seen.add(w)
    return tf, df, N

def rag_triage_keywords(retrieved_docs, conversation_messages, max_terms: int = 15) -> List[str]:
    if not retrieved_docs: return []
    rag_texts = [getattr(d, "page_content", "") or "" for d in retrieved_docs]
    tf, df, N = _count_words(rag_texts)
    if N == 0: return []
    convo = " ".join((m.get("content") or "") for m in conversation_messages if isinstance(m.get("content"), str))
    convo_terms = set(_tokenize(convo))
    scores={}
    for w, tfw in tf.items():
        idf = math.log((N+1)/(df.get(w,1))) + 1.0
        s = tfw * idf
        if w in convo_terms: s *= 1.25
        if len(w) <= 3: s *= 0.5
        scores[w]=s
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w,_ in ranked if w not in _STOP][:max_terms]

def highlight_text(text: str, keywords: List[str]) -> str:
    safe_text = html.escape(text or "")
    for kw in sorted(set(keywords), key=len, reverse=True):
        pattern = re.compile(rf"(?i)\b{re.escape(kw)}\b")
        safe_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe_text)
    return f"<div class='evidence-snippet'>{safe_text}</div>"

def _sentences(txt: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (txt or "").strip())
    return [p.strip() for p in parts if p.strip()]

def extract_claims_from_summary(summary_markdown: str) -> List[str]:
    triage_match = re.search(r"\*\*TRIAGE SUMMARY:\*\*(.*?)(?=\*\*URGENCY RECOMMENDATION:\*\*|$)", summary_markdown, flags=re.S|re.I)
    justification_match = re.search(r"\*\*JUSTIFICATION:\*\*(.*?)(?=\*\*Next Step:\*\*|$)", summary_markdown, flags=re.S|re.I)
    triage_text = triage_match.group(1).strip() if triage_match else ""
    justification_text = justification_match.group(1).strip() if justification_match else ""
    return _sentences(triage_text) + _sentences(justification_text)

def _keyword_density_score(text: str, keywords: List[str]) -> float:
    if not text or not keywords: return 0.0
    tl = text.lower()
    hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", tl))
    return hits / len(keywords)

def _overlap_score(a: str, b: str) -> float:
    A = set(w for w in _tokenize(a) if w not in _STOP)
    B = set(w for w in _tokenize(b) if w not in _STOP)
    return len(A & B) / len(A | B) if A and B else 0.0

def best_evidence_for_claims(retrieved_docs, claims: List[str], keywords: List[str], top_sources: int = 2) -> List[Dict]:
    alpha, beta = 0.7, 0.3
    results = []
    for claim in claims:
        per_source = []
        for idx, d in enumerate(retrieved_docs):
            content = getattr(d, "page_content", "") or ""
            score = alpha * _overlap_score(claim, content) + beta * _keyword_density_score(content, keywords)
            if score > 0.1: # Threshold to avoid very low-relevance matches
                per_source.append({"source_index": idx, "score": score})
        
        per_source = sorted(per_source, key=lambda x: x["score"], reverse=True)[:top_sources]
        results.append({"claim": claim, "evidence": per_source})
    return results

def pretty_source_label(doc) -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source")
    if source == "Kanski":
        return f"Kanski, Page {metadata.get('page', 'N/A')}"
    elif source == "NICE/NHS":
        return f"NICE/NHS: {metadata.get('title', 'N/A')}"
    return "Unknown Source"

def reset_conversation():
    """A dedicated function to clear all session state variables and start a new triage."""
    keys_to_reset = [
        "pre_triage_complete", "user_profile", "messages", "question_count",
        "finished", "is_recording", "transcribed_text", "input_mode",
        "turn", "retrieved_docs"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- The Main Streamlit Application ---

st.set_page_config(page_title="Ophthalmic Triage AI", page_icon="👁️", layout="wide")
st.title("👁️ Ophthalmic Triage AI")
st.subheader("An AI-powered assistant for assessing eye-related symptoms.")
st.caption("This is an academic proof-of-concept and does not provide medical advice.")

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
    with st.sidebar, st.spinner("Setting up knowledge base for the first time... (This may take a minute)"):
        create_knowledge_base()

api_key = st.secrets.get("GOOGLE_API_KEY")

@st.cache_resource
def load_agents(api_key):
    if not api_key:
        st.error("Google API Key not found. Please add it to your Streamlit Secrets.")
        st.stop()
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)
        retriever = RetrieverAgent()
        relevance_checker = RelevanceCheckAgent(llm)
        query_refiner = QueryRefinementAgent(llm)
        router = RouterAgent(llm)
        question_generator = QuestionAgent(llm)
        summary_generator = SummaryAgent(llm)
        return retriever, relevance_checker, query_refiner, router, question_generator, summary_generator
    except Exception as e:
        st.error(f"Failed to initialize models. Error: {e}")
        st.stop()

retriever, relevance_checker, query_refiner, router, question_generator, summary_generator = load_agents(api_key)

# Initialize the session state if it's the first run.
if "messages" not in st.session_state:
    st.session_state.pre_triage_complete = False
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.finished = False
    st.session_state.turn = "user"
    st.session_state.retrieved_docs = []
    st.session_state.input_mode = "text"

# === MAIN APP LOGIC: Pre-Triage or Chat ===

if not st.session_state.get("pre_triage_complete"):
    st.info("Please provide some basic information before we begin.")
    with st.form("pre_triage_form"):
        age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)
        sex = st.radio("What is your sex?", ("Male", "Female", "Other", "Prefer not to say"))
        conditions = st.multiselect(
            "Do you have any of the following pre-existing medical conditions?",
            ["Diabetes", "High Blood Pressure (Hypertension)", "Glaucoma", "Cataracts", "Macular Degeneration", "Autoimmune Disease (e.g., Rheumatoid Arthritis)","None"]
        )
        submitted = st.form_submit_button("Start Triage")

        if submitted:
            st.session_state.user_profile = {"age": age, "sex": sex, "conditions": conditions}
            profile_summary = (
                f"**Patient Profile Recorded:**\n"
                f"- **Age:** {age}\n"
                f"- **Sex:** {sex}\n"
                f"- **Existing Conditions:** {', '.join(conditions) if conditions else 'None'}"
            )
            st.session_state.messages = [
                {"role": "assistant", "content": profile_summary, "is_summary": True},
                {"role": "assistant", "content": "Thank you. Now, please describe your main eye-related concern."}
            ]
            st.session_state.pre_triage_complete = True
            st.rerun()
else:
    # --- Main Chat Interface ---
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("play_audio_autoplay", False):
                text_to_audio_autoplay(message["content"])
                st.session_state.messages[i]["play_audio_autoplay"] = False

    prompt = None
    if st.session_state.turn == "user" and not st.session_state.finished:
        if st.session_state.get("is_recording"):
            with st.spinner("🔴 Recording... Speak now!"):
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    try:
                        audio = r.listen(source, timeout=10, phrase_time_limit=30)
                        st.info("Transcribing...")
                        text = r.recognize_google(audio)
                        st.session_state.transcribed_text = text
                    except Exception as e:
                        st.warning("Could not process audio. Please try again.")
                st.session_state.is_recording = False
                st.rerun()
        else:
            if st.session_state.get("transcribed_text"):
                prompt = st.session_state.transcribed_text
                del st.session_state.transcribed_text
                st.session_state.input_mode = "voice"
            else:
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    text_prompt = st.chat_input("Describe your symptoms...")
                    if text_prompt:
                        prompt = text_prompt
                        st.session_state.input_mode = "text"
                with col2:
                    if st.button("🎤", key="speak_button"):
                        st.session_state.is_recording = True
                        st.rerun()

    elif st.session_state.turn == "assistant" and not st.session_state.finished:
        if st.button("▶️ Continue Conversation"):
            st.session_state.turn = "user"
            st.rerun()

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.input_mode == "voice":
            st.session_state.turn = "assistant"

        with st.spinner("Thinking..."):
            history = "\n".join([f"- {msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
            
            is_first_turn = len(st.session_state.messages) == 3
            if is_first_turn:
                if not relevance_checker.check_relevance(prompt):
                    response = "I am an ophthalmology triage assistant and can only help with eye-related problems. Please restart the conversation with an eye symptom."
                    st.session_state.finished = True
                else:
                    refined_query = query_refiner.refine_query(history)
                    st.session_state.retrieved_docs = retriever.retrieve_context(refined_query, k=5)
                    response = question_generator.generate_question(history)
                    st.session_state.question_count += 1
            else:
                refined_query = query_refiner.refine_query(history)
                st.session_state.retrieved_docs = retriever.retrieve_context(refined_query, k=5)
                context_str = "\n---\n".join([doc.page_content for doc in st.session_state.retrieved_docs])
                
                next_step = "provide_summary" if st.session_state.question_count >= MAX_QUESTIONS else router.route(history, context_str)

                if "ask_question" in next_step:
                    response = question_generator.generate_question(history)
                    st.session_state.question_count += 1
                else:
                    response = summary_generator.generate_summary(history)
                    st.session_state.finished = True

            if st.session_state.finished:
                urgency = extract_urgency(response)
                if urgency == "URGENT":
                    response += "\n\n**Next Step:** This may indicate a serious condition. Please go to your nearest Accident & Emergency (A&E) department immediately."
                elif urgency == "SEMI-URGENT":
                    response += "\n\n**Next Step:** Please contact an ophthalmologist or optometrist for an appointment within the next 24-48 hours."
                else:
                    response += "\n\n**Next Step:** Please book a routine appointment with your optometrist at your convenience."

        should_autoplay = st.session_state.input_mode == "voice" and not st.session_state.finished
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "play_audio_autoplay": should_autoplay,
            "is_summary": st.session_state.finished
        })
        st.rerun()

    if st.session_state.finished:
        st.button("Start New Triage", on_click=reset_conversation)

        full_history = "Ophthalmic Triage AI - Conversation Summary\n" + "="*40 + "\n\n"
        for msg in st.session_state.messages:
            full_history += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
        st.download_button(label="📥 Download Triage Summary", data=full_history, file_name="triage_summary.txt", mime="text/plain")

        with st.expander("🔎 View Clinical Evidence", expanded=True):
            # CSS for theme-aware, improved styling
            st.markdown("""
            <style>
            .evidence-claim { margin-bottom: 5px; color: var(--text-color); }
            .evidence-source { font-size: 0.9em; color: var(--secondary-text-color); margin-left: 10px; }
            .evidence-snippet { background-color: var(--secondary-background-color); border-left: 4px solid var(--separator-color); padding: 10px; margin-left: 10px; margin-bottom: 10px; border-radius: 4px; }
            .evidence-snippet mark { background-color: #ffeb3b; color: black; padding: 0.1em 0.2em; border-radius: 3px; }
            </style>
            """, unsafe_allow_html=True)

            retrieved = st.session_state.get("retrieved_docs", [])
            if retrieved:
                keywords = rag_triage_keywords(retrieved, st.session_state.messages, max_terms=15)
                
                summary_msg = ""
                for m in reversed(st.session_state.messages):
                    if m.get("role") == "assistant" and m.get("is_summary"):
                        summary_msg = m.get("content") or ""
                        break
                
                claims = extract_claims_from_summary(summary_msg)
                
                if not claims:
                    st.caption("Could not extract claims from the final summary.")
                else:
                    evidence_map = best_evidence_for_claims(retrieved, claims, keywords, top_sources=2)
                    for item in evidence_map:
                        st.markdown(f"<div class='evidence-claim'>🧾 <b>Claim:</b> {html.escape(item['claim'])}</div>", unsafe_allow_html=True)
                        if not item["evidence"]:
                            st.caption("No strong supporting source found among retrieved documents.")
                        else:
                            for ev in item["evidence"]:
                                doc = retrieved[ev["source_index"]]
                                label = pretty_source_label(doc)
                                content_for_snippet = getattr(doc, "page_content", "")
                                st.markdown(f"<div class='evidence-source'><b>Source:</b> {html.escape(label)} &nbsp;&middot;&nbsp; <b>Relevance:</b> {ev['score']:.2f}</div>", unsafe_allow_html=True)
                                st.markdown(highlight_text(content_for_snippet, keywords), unsafe_allow_html=True)
                        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            else:
                st.write("No sources were retrieved for this conversation.")