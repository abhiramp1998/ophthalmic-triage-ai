# OphthalmicTriageAI 🩺

An AI-powered multi-agent chatbot for emergency ophthalmic triage, built with a Retrieval-Augmented Generation (RAG) architecture and grounded in clinical textbooks.

---

##  About the Project

This application is a sophisticated proof-of-concept developed for a dissertation project to explore the use of Large Language Models in clinical decision support. It is designed to assist in the initial triage of emergency eye-related symptoms by conducting an intelligent, multi-turn conversation with a user.

The system's knowledge base is grounded in authoritative clinical sources, including **Kanski’s Clinical Ophthalmology**, to ensure its questions and assessments are clinically relevant and safe.

*Disclaimer: This is an academic proof-of-concept and is not a certified medical device. It does not provide medical advice.*

---

##  Key Features

* **Intelligent Dialogue:** Asks a series of clarifying questions to understand symptoms fully, guided by a sophisticated `RouterAgent`.
* **Clinically Grounded:** Uses a FAISS vector database to retrieve context from trusted medical texts, ensuring responses are based on facts, not just the LLM's general knowledge.
* **Structured Summaries:** Provides a final summary with a clear urgency recommendation (Urgent, Semi-Urgent, or Routine) and a justification.
* **Robust Safety Rails:** Includes checks for off-topic queries, irrelevant user answers, and a maximum question limit to prevent conversational loops.

---

##  Technology Stack

* **Backend & Logic:** Python, LangChain
* **AI Model:** Google Gemini 1.5 Flash
* **Knowledge Base:** FAISS Vector Store
* **User Interface:** Streamlit

---

##  How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/OphthalmicTriageAI.git](https://github.com/your-username/OphthalmicTriageAI.git)
    cd OphthalmicTriageAI
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your API Key:**
    * Create a file at `.streamlit/secrets.toml`.
    * Add your Google Gemini API key to it in the following format:
        ```toml
        GOOGLE_API_KEY = "your_api_key_here"
        ```
4.  **Run the application:**
    ```bash
    streamlit run agent.py
    ```
The application will build the knowledge base on the first run and then launch in your web browser.