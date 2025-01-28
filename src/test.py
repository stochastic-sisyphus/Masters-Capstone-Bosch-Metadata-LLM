import os
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from typing import Optional, List
from vertexai import init
from vertexai.generative_models import GenerativeModel
import torch

# === Initialize Vertex AI ===
PROJECT_ID = "ids-560-project-group-1-bosch"
init(project=PROJECT_ID, location="us-central1")
gemini_model = GenerativeModel("gemini-1.5-flash")

# === Custom Wrapper for Gemini ===
class RunnableGemini(LLM):
    def __init__(self, model: GenerativeModel):
        super().__init__()
        self._model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

# === Custom Embedding Wrapper ===
class HFEmbeddingWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text).flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text).flatten().tolist() for text in texts]

    def _get_embedding(self, text: str):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

# === Initialize Pinecone ===
def initialize_pinecone():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API Key is missing!")

        pc = Pinecone(api_key=api_key)
        print(f"Available indexes: {pc.list_indexes().names()}")
        return pc
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

# === Initialize Pinecone Index ===
def get_pinecone_index(pc, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

# === Build Conversational Chain ===
def build_conversational_chain(index, llm):
    retriever = LangchainPinecone(index=index, embedding=HFEmbeddingWrapper("sentence-transformers/all-MiniLM-L6-v2")).as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt = PromptTemplate.from_template("""
    Context:
    {context}
    Question: {question}
    Answer:
    """)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        input_key="question",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
# === Streamlit Chatbot UI ===
def chatbot_ui(pc, llm):
    # Ensure selected_db is initialized in session_state
    if "selected_db" not in st.session_state:
        st.session_state.selected_db = "chunks-with-metadata-final"  # Default database selection

    # Map database names to response labels
    response_labels = {
        "chunks-with-metadata-final": "Enriched",
        "chunks-without-metadata-final": "Naive",
    }

    # Set a constant heading for the chatbot
    st.title("AWS S3 Doc Helper")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        selected_db = st.selectbox(
            "Choose a Database",
            options=["chunks-with-metadata-final", "chunks-without-metadata-final"],
            key="db_selector"
        )

        # Check if the database has changed
        if st.session_state.selected_db != selected_db:
            st.session_state.selected_db = selected_db
            st.session_state.input_query = ""  # Clear input when the database changes

        if st.button("Clear Chat"):
            st.session_state.requests = []
            st.session_state.responses = []
            st.session_state.response_dbs = []  # Clear database tracking list

    # Session state
    if "requests" not in st.session_state:
        st.session_state.requests = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "response_dbs" not in st.session_state:
        st.session_state.response_dbs = []  # Initialize database tracking

    if "input_query" not in st.session_state:
        st.session_state.input_query = ""  # Initialize input query in session state

    # Initialize the Pinecone index
    index = get_pinecone_index(pc, st.session_state.selected_db)
    chain = build_conversational_chain(index, llm)

    # Introductory line with smaller font size
    st.markdown(
        f"<p style='font-size:14px; color:gray;'>Welcome to S3 Doc Helper. How can I assist you today?</p>",
        unsafe_allow_html=True,
    )

    # Chat Interface
    for i, query in enumerate(st.session_state.requests):
        st.text_area(f"User Query {i+1}:", value=query, disabled=True, key=f"user_query_{i}")
        if i < len(st.session_state.responses):
            # Fetch the response label (Naive/Enriched) based on the database
            dbname = st.session_state.response_dbs[i]
            response_label = response_labels.get(st.session_state.response_dbs[i], dbname)
            # Include the label in the bot response
            st.text_area(
                f"Bot Response ({response_label}) {i+1}:",
                value=st.session_state.responses[i],
                height=250,
                disabled=True,
                key=f"bot_response_{i}",
            )

    # Auto-scroll to the bottom of the page
    st.markdown(
        """
        <script>
            var elem = document.documentElement;
            window.scrollTo(0, elem.scrollHeight);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # User Input
    col1, col2 = st.columns([4, 1])
    with col1:
        # Use a text area for user input
        query = st.text_area("Enter your query:", value=st.session_state.input_query, key="unique_user_query")
    with col2:
        if st.button("Submit", key="submit_button"):
            if query.strip():  # Ensure query is not empty or just spaces
                st.session_state.input_query = ""  # Reset the input field
                st.session_state.requests.append(query.strip())
                memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
                result = chain({"question": query.strip(), "chat_history": memory.load_memory_variables({}).get("chat_history", "")})
                st.session_state.responses.append(result["result"])
                # Save the current database's response label (Naive/Enriched)
                st.session_state.response_dbs.append(response_labels.get(st.session_state.selected_db, "Unknown"))

# === Main ===
def main():
    pc = initialize_pinecone()
    llm = RunnableGemini(gemini_model)

    if pc and llm:
        chatbot_ui(pc, llm)

if __name__ == "__main__":
    main()
