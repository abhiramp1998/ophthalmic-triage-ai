# 👁️ Ophthalmic Triage AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg) ![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green.svg)

An advanced, multi-agent AI assistant for the preliminary triage of emergency eye-related symptoms. This application is a dissertation project showcasing a robust, clinically-grounded RAG (Retrieval-Augmented Generation) system.

---

## 🚀 Live Demo

**You can test the live application here:** [**https://ophthalmic-triage-ai-keq9aqp6boy6kuqbuf2jsh.streamlit.app**](https://ophthalmic-triage-ai-keq9aqp6boy6kuqbuf2jsh.streamlit.app)

*(Note: The application is hosted on Streamlit's free community cloud and may take a moment to wake up from sleep.)*

---

## 📋 About The Project

This application is a proof-of-concept designed to explore the use of Large Language Models in a safe and effective manner for clinical decision support. It acts as an intelligent assistant that conducts a multi-turn, conversational triage for users reporting eye-related symptoms.

The core principle of this project is **clinical grounding**. The AI's reasoning is not based on its general knowledge but is augmented by information retrieved in real-time from a specialized knowledge base built from authoritative sources like **Kanski’s Clinical Ophthalmology** and **NICE/NHS guidelines**. This RAG architecture ensures that the AI's questions and final assessments are both relevant and clinically sound.

### 💡 Pro-Tip: Create an Animated GIF!
The best way to showcase your app is with a short, looping video. Use a tool like Giphy Capture (macOS), LICEcap (Windows/macOS), or ScreenToGif (Windows) to record a 15-20 second clip of you interacting with the app. Upload it to your repository and add a line like `![App Demo](demo.gif)` to this README.

---

## ✨ Key Features

* **🧠 Intelligent Multi-Agent System:** A sophisticated architecture where multiple specialized AI agents collaborate to manage the conversation, retrieve information, and generate responses.
* **🗣️ Multi-Modal Interaction:** Users can interact via **typed text** or **voice commands** (Speech-to-Text).
* **🔊 Accessible Output:** The assistant provides **spoken responses** (Text-to-Speech) in voice-led conversations, improving accessibility for visually impaired users.
* **Dynamic Conversational Flow:** A `RouterAgent` intelligently decides whether to ask more clarifying questions or proceed to a summary, allowing for conversations of variable length.
* **🔬 Clinically Grounded Reasoning:** A `QueryRefinementAgent` and `RetrieverAgent` work together to find the most relevant clinical information from the knowledge base at each step of the conversation.
* **📄 Structured Clinical Summary:** The `SummaryAgent` provides a final, formatted report including a **Triage Summary**, an **Urgency Recommendation**, and a **Clinical Justification**.
* **🛡️ Robust Safety Rails:** The system includes checks for off-topic queries and a maximum question limit to prevent errors and conversational loops.
* **🗺️ Actionable Next Steps:** The final summary includes clear, safe guidance for the user based on the triage outcome.
* **📚 Explainability & Transparency:** Users can "View Clinical Sources" to see the exact text from the knowledge base that the AI used to inform its reasoning.
* **📥 Conversation Export:** The full conversation and summary can be downloaded as a `.txt` file for easy sharing with a healthcare professional.

---

## 🛠️ Technology Stack

* **Backend & Logic:** Python, LangChain
* **AI Model:** Google Gemini 1.5 Flash
* **Knowledge Base:** FAISS Vector Store
* **User Interface:** Streamlit
* **Speech I/O:** `SpeechRecognition`, `gTTS`

---

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/abhiramp1998/ophthalmic-triage-ai.git](https://github.com/abhiramp1998/ophthalmic-triage-ai.git)
    cd ophthalmic-triage-ai
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your API Key:**
    * Create a folder in the root directory named `.streamlit`.
    * Inside that folder, create a file named `secrets.toml`.
    * Add your Google Gemini API key to it in the following format:
        ```toml
        GOOGLE_API_KEY = "your_api_key_here"
        ```
4.  **Run the application:**
    ```bash
    streamlit run agent.py
    ```
The application will build the knowledge base on the first run (which may take a few minutes) and then launch in your web browser.

---

### **Disclaimer**
This tool is an academic proof-of-concept and is **not a certified medical device**. The information provided is for educational and research purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment.