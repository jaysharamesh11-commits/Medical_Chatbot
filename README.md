🩺 Medical Q&A Chatbot (Using MedQuAD Dataset)
📌 Problem Statement

Accessing accurate and reliable medical information can be challenging.
This project aims to build a Medical Question-Answering Chatbot using the MedQuAD dataset, capable of retrieving relevant, verified answers to health-related queries.

📚 Dataset

Name: MedQuAD Dataset

Description: A curated medical Q&A dataset containing questions and answers extracted from trusted medical sources like NIH and NLM.

🧠 Methodology

Data Preprocessing:

Extracted question-answer pairs from MedQuAD.

Cleaned and structured the text for retrieval-based querying.

Medical Entity Recognition:

Implemented basic recognition of medical entities (e.g., symptoms, diseases, treatments) using spaCy’s medical model or pattern matching.

Information Retrieval:

Used TF-IDF / Sentence Transformers embeddings to compute semantic similarity.

Implemented a retrieval mechanism to return the most relevant answer.

Frontend with Streamlit:

Built a simple interactive UI to input medical questions and display chatbot responses.

⚙️ Tech Stack

Language: Python

Libraries: Streamlit, Scikit-learn / SentenceTransformers, Pandas, NumPy, spaCy, NLTK

Dataset: MedQuAD

Interface: Streamlit web app

🚀 How to Run Locally
# 1. Clone the repository
git clone https://github.com/yourusername/MedQuAD_Chatbot.git
cd MedQuAD_Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

📊 Results / Demo

Achieved accurate retrieval of relevant medical responses.

Example queries:

“What are the symptoms of diabetes?”

“How is asthma treated?”

[Insert screenshot of your chatbot UI here]

🧩 Future Improvements

Integrate advanced medical entity extraction models (e.g., SciSpacy).

Add conversational memory.

Deploy using Streamlit Cloud / Hugging Face Spaces.

👩‍💻 Author
Jaysha 