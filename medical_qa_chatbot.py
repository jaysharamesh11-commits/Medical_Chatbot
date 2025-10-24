import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pickle

# Medical entity patterns
MEDICAL_ENTITIES = {
    'symptoms': ['pain', 'fever', 'cough', 'fatigue', 'nausea', 'headache', 'dizziness', 
                 'bleeding', 'swelling', 'rash', 'weakness', 'vomiting', 'diarrhea'],
    'diseases': ['diabetes', 'cancer', 'hypertension', 'asthma', 'arthritis', 'alzheimer',
                 'parkinson', 'depression', 'anxiety', 'infection', 'disease', 'syndrome'],
    'treatments': ['treatment', 'therapy', 'medication', 'surgery', 'drug', 'medicine',
                   'prescription', 'procedure', 'vaccine', 'antibiotic', 'exercise']
}

class MedicalEntityRecognizer:
    """Simple rule-based medical entity recognizer"""
    
    def __init__(self):
        self.entities = MEDICAL_ENTITIES
    
    def recognize(self, text):
        """Recognize medical entities in text"""
        text_lower = text.lower()
        found_entities = {'symptoms': [], 'diseases': [], 'treatments': []}
        
        for category, terms in self.entities.items():
            for term in terms:
                if term in text_lower:
                    found_entities[category].append(term)
        
        return found_entities

class MedicalQASystem:
    """Medical Question-Answering System with retrieval mechanism"""
    
    def __init__(self):
        self.qa_data = []
        self.vectorizer = None
        self.qa_vectors = None
        self.entity_recognizer = MedicalEntityRecognizer()
    
    def parse_medquad_xml(self, xml_file):
        """Parse MedQuAD XML file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for qa_pair in root.findall('.//QAPair'):
                question = qa_pair.find('Question')
                answer = qa_pair.find('Answer')
                
                if question is not None and answer is not None:
                    q_text = question.text if question.text else ""
                    a_text = answer.text if answer.text else ""
                    
                    if q_text and a_text:
                        self.qa_data.append({
                            'question': q_text.strip(),
                            'answer': a_text.strip(),
                            'source': os.path.basename(xml_file)
                        })
        except Exception as e:
            st.warning(f"Error parsing {xml_file}: {str(e)}")
    
    def load_medquad_data(self, data_path):
        """Load all MedQuAD XML files from directory"""
        data_path = Path(data_path)
        xml_files = list(data_path.rglob('*.xml'))
        
        if not xml_files:
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, xml_file in enumerate(xml_files):
            status_text.text(f"Loading: {xml_file.name}")
            self.parse_medquad_xml(xml_file)
            progress_bar.progress((idx + 1) / len(xml_files))
        
        progress_bar.empty()
        status_text.empty()
        
        return len(self.qa_data) > 0
    
    def build_index(self):
        """Build TF-IDF index for retrieval"""
        if not self.qa_data:
            return False
        
        questions = [qa['question'] for qa in self.qa_data]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.qa_vectors = self.vectorizer.fit_transform(questions)
        return True
    
    def retrieve_answers(self, query, top_k=3):
        """Retrieve most relevant answers for a query"""
        if self.vectorizer is None or self.qa_vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.qa_vectors)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'question': self.qa_data[idx]['question'],
                    'answer': self.qa_data[idx]['answer'],
                    'similarity': float(similarities[idx]),
                    'source': self.qa_data[idx]['source']
                })
        
        return results
    
    def save_index(self, filepath='medqa_index.pkl'):
        """Save the index to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'qa_data': self.qa_data,
                'vectorizer': self.vectorizer,
                'qa_vectors': self.qa_vectors
            }, f)
    
    def load_index(self, filepath='medqa_index.pkl'):
        """Load the index from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.qa_data = data['qa_data']
                self.vectorizer = data['vectorizer']
                self.qa_vectors = data['qa_vectors']
            return True
        except Exception:
            return False

def initialize_system():
    """Initialize the QA system"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = MedicalQASystem()
        st.session_state.initialized = False

def main():
    st.set_page_config(
        page_title="Medical Q&A Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Medical Q&A System")
        st.markdown("---")
        
        st.subheader("📚 Data Management")
        
        # Try to load existing index
        if st.button("Load Saved Index"):
            if st.session_state.qa_system.load_index():
                st.session_state.initialized = True
                st.success(f"✅ Loaded {len(st.session_state.qa_system.qa_data)} Q&A pairs")
            else:
                st.error("No saved index found")
        
        st.markdown("---")
        
        # Load from directory
        data_path = st.text_input(
            "MedQuAD Data Path",
            placeholder="/path/to/MedQuAD",
            help="Enter the path to MedQuAD dataset directory"
        )
        
        if st.button("Load from Directory"):
            if data_path and os.path.exists(data_path):
                with st.spinner("Loading MedQuAD data..."):
                    if st.session_state.qa_system.load_medquad_data(data_path):
                        st.success(f"✅ Loaded {len(st.session_state.qa_system.qa_data)} Q&A pairs")
                        
                        with st.spinner("Building search index..."):
                            if st.session_state.qa_system.build_index():
                                st.session_state.initialized = True
                                st.success("✅ Index built successfully")
                                
                                # Save index
                                st.session_state.qa_system.save_index()
                                st.info("💾 Index saved for future use")
                    else:
                        st.error("No XML files found in the specified path")
            else:
                st.error("Invalid path")
        
        st.markdown("---")
        
        if st.session_state.initialized:
            st.metric("Total Q&A Pairs", len(st.session_state.qa_system.qa_data))
        
        st.markdown("---")
        st.caption("⚠️ **Disclaimer**: This is for educational purposes only. Always consult healthcare professionals for medical advice.")
    
    # Main content
    st.title("💬 Medical Question & Answer Chatbot")
    st.markdown("Ask medical questions and get answers from trusted medical sources.")
    
    if not st.session_state.initialized:
        st.info("👈 Please load the MedQuAD dataset from the sidebar to start asking questions.")
        
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. **Download MedQuAD Dataset**: Clone from [GitHub](https://github.com/abachaa/MedQuAD)
        2. **Load Data**: Enter the path to the dataset in the sidebar and click 'Load from Directory'
        3. **Ask Questions**: Once loaded, type your medical question below
        4. **Get Answers**: View relevant answers with similarity scores
        """)
        
        st.markdown("### 📝 Example Questions:")
        st.markdown("""
        - What are the symptoms of diabetes?
        - How is hypertension treated?
        - What causes migraine headaches?
        - What are the risk factors for heart disease?
        """)
    else:
        # Question input
        question = st.text_area(
            "❓ Your Medical Question:",
            height=100,
            placeholder="E.g., What are the symptoms of diabetes?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            top_k = st.selectbox("Results", [1, 2, 3, 4, 5], index=2)
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and question:
            with st.spinner("Searching for answers..."):
                # Recognize entities
                entities = st.session_state.qa_system.entity_recognizer.recognize(question)
                
                # Show recognized entities
                if any(entities.values()):
                    st.markdown("### 🏷️ Recognized Medical Entities:")
                    cols = st.columns(3)
                    
                    if entities['symptoms']:
                        with cols[0]:
                            st.info(f"**Symptoms**: {', '.join(set(entities['symptoms']))}")
                    
                    if entities['diseases']:
                        with cols[1]:
                            st.warning(f"**Diseases**: {', '.join(set(entities['diseases']))}")
                    
                    if entities['treatments']:
                        with cols[2]:
                            st.success(f"**Treatments**: {', '.join(set(entities['treatments']))}")
                
                # Retrieve answers
                results = st.session_state.qa_system.retrieve_answers(question, top_k=top_k)
                
                if results:
                    st.markdown("### 📋 Relevant Answers:")
                    
                    for idx, result in enumerate(results, 1):
                        with st.expander(
                            f"**Answer {idx}** (Similarity: {result['similarity']:.2%})",
                            expanded=(idx == 1)
                        ):
                            st.markdown(f"**❓ Question**: {result['question']}")
                            st.markdown("---")
                            st.markdown(f"**💡 Answer**: {result['answer']}")
                            st.caption(f"📄 Source: {result['source']}")
                else:
                    st.warning("No relevant answers found. Try rephrasing your question.")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if search_button and question:
            st.session_state.chat_history.append({
                'question': question,
                'results': results
            })

if __name__ == "__main__":
    main()