# Should be first import
from __future__ import annotations

# Standard library imports
import os
import time
import getpass
import asyncio
import re
from typing import Optional, List, Any, Union, Tuple, TYPE_CHECKING, Generator

# Third-party ML/AI imports
import torch
import torch.nn.functional as F
import openai
import tiktoken
import PyPDF2
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel

# LangChain imports
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore, Pinecone as LangchainPinecone

# Project-specific imports
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec

# Constants
SEMANTIC_THRESHOLD = 0.8
MAX_TOKENS = 20000
MAX_INSTANCES = 250

# Configuration
load_dotenv(find_dotenv())

# Azure OpenAI Configuration
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = "Azure"
OPENAI_API_BASE = "https://testopenaisaturday.openai.azure.com/"
OPENAI_API_VERSION = "2023-10-01-preview"

# Configure OpenAI
openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

# Vertex AI Configuration
PROJECT_ID = "ids-560-project-group-1-bosch"
REGION = "us-central1"
MODEL_ID = "text-embedding-004"
MODEL_NAME = "text-embedding-preview-0815"
DIMENSIONALITY = 256

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Template Configuration
DEFAULT_QA_TEMPLATE = """
You are an assistant helping with error debugging and code understanding. 
Use the context provided to answer the user's question. 
If there is no context, let the user know more information is needed.

Context:
{context}

User's Question: {question}
Answer:
"""

# Model Wrapper Classes
class RunnableGemini(LLM):
    """Wrapper class to make Google's Gemini model compatible with LangChain."""
    
    def __init__(self, model: GenerativeModel):
        """Initialize with a Gemini model instance."""
        super().__init__()
        self._model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Execute the model on the given prompt."""
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini model: {e}")
            return "An error occurred while generating the response."

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "google_gemini"

class HFEmbeddingWrapper(Embeddings):
    """Wrapper class to make HuggingFace embeddings compatible with LangChain."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize with a model name."""
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding.flatten().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        return self.embed_documents([text])[0]

class QASystem:
    """Main QA system for handling queries with conversation history."""
    
    def __init__(
        self,
        llm: LLM,
        embedding: Embeddings,
        index: Any,
        template: str = DEFAULT_QA_TEMPLATE
    ):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.vectorstore = LangchainPinecone(index=index, embedding=embedding)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        self.qa_chain = self._setup_qa_chain(llm, template)
        self._request_lock = asyncio.Lock()

    def _setup_qa_chain(self, llm: LLM, template: str) -> ConversationalRetrievalChain:
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

    async def query(self, question: str, metadata_filter: Optional[dict] = None) -> dict:
        async with self._request_lock:
            try:
                if metadata_filter:
                    self.retriever.search_kwargs.update({"filter": metadata_filter})
                
                response = await self.qa_chain.ainvoke({
                    "question": question,
                    "chat_history": self.memory.chat_memory.messages
                })
                
                return {
                    "answer": response["answer"],
                    "source_documents": response["source_documents"],
                    "chat_history": self.memory.chat_memory.messages
                }
            
            except Exception as e:
                print(f"Error during query: {e}")
                return {
                    "error": str(e),
                    "answer": "An error occurred while processing your question."
                }
            finally:
                if metadata_filter:
                    self.retriever.search_kwargs.pop("filter", None)

# Utility Functions
def embed_text(
    texts: list[str],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = MODEL_NAME,
    dimensionality: int | None = DIMENSIONALITY
) -> list[list[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]

def init_pinecone(dimension: int = 384) -> Pinecone:
    """Initialize Pinecone with the specified dimension."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = getpass.getpass("Enter your Pinecone API key: ")
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "langchain-test-index"
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        pc.delete_index(index_name)
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    
    return pc.Index(index_name)

def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def split_into_token_batches(
    sentences: list[str], 
    max_tokens: int = MAX_TOKENS, 
    max_instances: int = MAX_INSTANCES
) -> Generator[list[str], None, None]:
    """Split sentences into batches based on token and instance count limits."""
    enc = tiktoken.get_encoding("cl100k_base")
    current_batch = []
    current_tokens = 0
    current_instances = 0
    
    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        
        if (current_tokens + sentence_tokens > max_tokens) or (current_instances + 1 > max_instances):
            yield current_batch
            current_batch = []
            current_tokens = 0
            current_instances = 0
            
        current_batch.append(sentence)
        current_tokens += sentence_tokens
        current_instances += 1
    
    if current_batch:
        yield current_batch

def semantic_chunking(text: str, threshold: float = SEMANTIC_THRESHOLD) -> list[str]:
    """Split text into semantically coherent chunks."""
    sentences = split_text_into_sentences(text)
    chunks = []
    
    for batch in split_into_token_batches(sentences):
        embeddings = embed_text(batch, task="RETRIEVAL_DOCUMENT")
        current_chunk = [batch[0]]
        
        for i in range(1, len(batch)):
            similarity = cosine_similarity(
                [embeddings[i-1]], 
                [embeddings[i]]
            )[0][0]
            
            if similarity < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [batch[i]]
            else:
                current_chunk.append(batch[i])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_text_from_pdf(pdf_file_path: str, max_pages: Optional[int] = None) -> str:
    """Extract text from a PDF file."""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        pages = range(len(reader.pages)) if max_pages is None else range(min(max_pages, len(reader.pages)))
        
        for page_num in pages:
            page = reader.pages[page_num]
            if extracted_text := page.extract_text():
                text += extracted_text
    return text

def parse_metadata_response(response_text: str) -> tuple[list[str], list[str]]:
    """
    Parse the metadata response from Gemini to extract topics and tags.
    
    Args:
        response_text: Raw response text from Gemini
        
    Returns:
        tuple[list[str], list[str]]: Lists of topics and tags
    """
    topics = []
    tags = []
    
    try:
        # Split response into lines and process each line
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            # Skip empty lines
            if not line:
                continue
                
            # Look for topic/heading indicators
            if any(x in line for x in ['topic:', 'heading:', '##', '#']):
                # Clean up the line
                topic = line.replace('topic:', '').replace('heading:', '').replace('#', '').strip()
                if topic:
                    topics.append(topic)
                    
            # Look for tag indicators
            elif 'tags:' in line or 'tag:' in line:
                # Extract tags, handling different formats
                tag_text = line.split(':', 1)[1].strip()
                # Handle comma-separated or space-separated tags
                if ',' in tag_text:
                    tags.extend([t.strip() for t in tag_text.split(',') if t.strip()])
                else:
                    tags.extend([t.strip() for t in tag_text.split() if t.strip()])
    
    except Exception as e:
        print(f"Error parsing metadata response: {e}")
        print(f"Response text was: {response_text}")
    
    # Remove duplicates while preserving order
    topics = list(dict.fromkeys(topics))
    tags = list(dict.fromkeys(tags))
    
    return topics, tags

def generate_metadata_for_chunks(
    model: GenerativeModel,
    chunks: list[str],
    max_tokens: int = 2048
) -> list[dict]:
    """
    Generate metadata for text chunks using Gemini model.
    
    Args:
        model: Initialized Gemini model
        chunks: List of text chunks to process
        max_tokens: Maximum tokens per chunk
        
    Returns:
        list[dict]: List of dictionaries containing chunk metadata
    """
    results = []
    for i, chunk in enumerate(chunks):
        try:
            truncated_chunk = chunk[:max_tokens]
            
            prompt = f"""
            Analyze the following text and provide:
            1. 2-3 topic headings that best describe the main concepts
            2. 2-3 relevant tags for categorization
            
            Format your response as:
            Topic 1: [topic]
            Topic 2: [topic]
            Tags: [tag1], [tag2], [tag3]
            
            Text: "{truncated_chunk}"
            """
            
            response = model.generate_content(prompt)
            topics, tags = parse_metadata_response(response.text)
            
            results.append({
                "chunk_index": i,
                "chunk_text": truncated_chunk,
                "metadata": {
                    "raw_response": response.text,
                    "topics": topics,
                    "tags": tags
                }
            })
            
        except Exception as e:
            print(f"Error generating metadata for chunk {i}: {e}")
            results.append({
                "chunk_index": i,
                "chunk_text": truncated_chunk,
                "metadata": {
                    "error": str(e),
                    "raw_response": "",
                    "topics": [],
                    "tags": []
                }
            })
    
    return results

def index_chunks_with_metadata(
    index: Any,
    chunks: list[str],
    metadata: list[dict]
) -> None:
    """Index chunks and their metadata in Pinecone."""
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        embedding = embed_text([chunk], task="RETRIEVAL_DOCUMENT")[0]
        
        doc_metadata = {
            "text": chunk,
            "topics": meta.get("metadata", {}).get("topics", []),
            "tags": meta.get("metadata", {}).get("tags", []),
            "raw_response": meta.get("metadata", {}).get("raw_response", "")
        }
        
        index.upsert(
            vectors=[(f"chunk-{i}", embedding, doc_metadata)]
        )

# Add these utility functions after the existing ones:

def batch_process_documents(
    file_paths: List[str],
    max_pages_per_doc: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> dict[str, list[str]]:
    """
    Process multiple documents in batch, extracting and chunking text.
    
    Args:
        file_paths: List of paths to PDF files
        max_pages_per_doc: Maximum pages to process per document
        progress_callback: Optional callback function for progress updates
        
    Returns:
        dict[str, list[str]]: Dictionary mapping file paths to their chunks
    """
    results = {}
    
    for i, path in enumerate(file_paths):
        try:
            # Extract text
            text = extract_text_from_pdf(path, max_pages=max_pages_per_doc)
            
            # Chunk text
            chunks = semantic_chunking(text)
            
            results[path] = chunks
            
            if progress_callback:
                progress_callback(i + 1, len(file_paths), path)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            results[path] = []
    
    return results

def filter_chunks_by_metadata(
    qa_system: QASystem,
    query: str,
    metadata_filters: dict,
    top_k: int = 5
) -> List[dict]:
    """
    Search for chunks matching both semantic similarity and metadata filters.
    
    Args:
        qa_system: Initialized QA system
        query: Search query
        metadata_filters: Dictionary of metadata filters
        top_k: Number of results to return
        
    Returns:
        List[dict]: List of matching chunks with their metadata
    """
    try:
        # Update search parameters
        qa_system.retriever.search_kwargs.update({
            "k": top_k,
            "filter": metadata_filters
        })
        
        # Perform search
        docs = qa_system.retriever.get_relevant_documents(query)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)
            })
        
        return results
        
    except Exception as e:
        print(f"Error during filtered search: {e}")
        return []
    
    finally:
        # Reset search parameters
        qa_system.retriever.search_kwargs.pop("filter", None)
        qa_system.retriever.search_kwargs["k"] = 5

async def interactive_qa_session(
    qa_system: QASystem,
    exit_commands: set[str] = {"quit", "exit", "bye"}
) -> None:
    """
    Run an interactive Q&A session with the system.
    
    Args:
        qa_system: Initialized QA system
        exit_commands: Set of commands that will end the session
    """
    print("Starting Q&A session. Type 'quit', 'exit', or 'bye' to end.")
    print("You can also use metadata filters by prefixing with 'filter:'")
    
    while True:
        try:
            # Get user input
            user_input = input("\nQuestion: ").strip().lower()
            
            # Check for exit command
            if user_input in exit_commands:
                print("Ending session.")
                break
            
            # Check for filter command
            metadata_filter = None
            if user_input.startswith("filter:"):
                filter_parts = user_input[7:].split(";")
                metadata_filter = {}
                for part in filter_parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        metadata_filter[key.strip()] = value.strip()
                
                # Get the actual question
                user_input = input("Enter your question: ").strip()
            
            # Get response
            response = await qa_system.query(user_input, metadata_filter=metadata_filter)
            
            # Print answer
            print("\nAnswer:", response["answer"])
            
            # Print sources if available
            if sources := response.get("source_documents"):
                print("\nSources:")
                for i, doc in enumerate(sources, 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

# Add a main function to demonstrate usage
async def main():
    try:
        # Initialize Pinecone
        index = init_pinecone()
        
        # Initialize Gemini model
        gemini_model = GenerativeModel("gemini-1.5-flash")
        
        # Initialize QA system
        qa_system = QASystem(
            llm=RunnableGemini(gemini_model),
            embedding=HFEmbeddingWrapper(),
            index=index
        )
        
        # Process documents if provided
        pdf_paths = ["example1.pdf", "example2.pdf"]  # Update with actual paths
        if any(os.path.exists(path) for path in pdf_paths):
            # Process documents
            doc_chunks = batch_process_documents(
                pdf_paths,
                max_pages_per_doc=None,
                progress_callback=lambda i, total, path: print(f"Processing {i}/{total}: {path}")
            )
            
            # Generate metadata and index for each document
            for path, chunks in doc_chunks.items():
                if chunks:
                    metadata = generate_metadata_for_chunks(gemini_model, chunks)
                    index_chunks_with_metadata(index, chunks, metadata)
                    print(f"Indexed {len(chunks)} chunks from {path}")
        
        # Start interactive session
        await interactive_qa_session(qa_system)
        
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        print("Session ended.")

# Main execution
if __name__ == "__main__":
    asyncio.run(main())

# Add after the existing embed_text function:

def embed_code_block(query: str) -> List[float]:
    """Embed a code block query."""
    return embed_text(
        texts=[query],
        task="CODE_RETRIEVAL_QUERY",
        model_name=MODEL_NAME,
        dimensionality=DIMENSIONALITY
    )[0]

def embed_code_retrieval(code_snippets: List[str]) -> List[List[float]]:
    """Embed code snippets for retrieval."""
    return embed_text(
        texts=code_snippets,
        task="RETRIEVAL_DOCUMENT",
        model_name=MODEL_NAME,
        dimensionality=DIMENSIONALITY
    )

def process_text_with_metadata(
    text: str,
    model: GenerativeModel,
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> List[dict]:
    """Process text with metadata generation."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create document
    document = Document(page_content=text)
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    
    # Split into chunks
    chunks = text_splitter.split_documents([document])
    
    # Generate metadata
    metadata = generate_metadata_for_chunks(
        model=model,
        chunks=[chunk.page_content for chunk in chunks]
    )
    
    return metadata

# Add these additional semantic chunking functions:

def semantic_chunking_with_titles(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    min_chunk_size: int = 100
) -> List[dict]:
    """Split text into chunks while preserving title context."""
    from unstructured.chunking.title import chunk_by_title
    
    chunks = []
    current_title = None
    
    # First split by titles
    title_chunks = chunk_by_title(text)
    
    for chunk in title_chunks:
        if chunk.type == ElementType.TITLE:
            current_title = chunk.text
        else:
            # Further split large chunks semantically
            if len(chunk.text) > min_chunk_size:
                sub_chunks = semantic_chunking(chunk.text, threshold)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        "title": current_title,
                        "text": sub_chunk
                    })
            else:
                chunks.append({
                    "title": current_title,
                    "text": chunk.text
                })
    
    return chunks

def semantic_chunking_with_code(
    text: str,
    code_pattern: str = r'```[\s\S]*?```|`[\s\S]*?`',
    threshold: float = SEMANTIC_THRESHOLD
) -> List[dict]:
    """Split text while preserving code blocks."""
    # Find all code blocks
    code_blocks = re.finditer(code_pattern, text)
    code_spans = [(m.start(), m.end(), m.group()) for m in code_blocks]
    
    chunks = []
    last_end = 0
    
    for start, end, code in code_spans:
        # Process text before code block
        if start > last_end:
            text_chunk = text[last_end:start].strip()
            if text_chunk:
                text_chunks = semantic_chunking(text_chunk, threshold)
                chunks.extend([{"type": "text", "content": chunk} for chunk in text_chunks])
        
        # Add code block as its own chunk
        chunks.append({"type": "code", "content": code.strip('`')})
        last_end = end
    
    # Process remaining text
    if last_end < len(text):
        text_chunk = text[last_end:].strip()
        if text_chunk:
            text_chunks = semantic_chunking(text_chunk, threshold)
            chunks.extend([{"type": "text", "content": chunk} for chunk in text_chunks])
    
    return chunks

# Add these metadata processing functions:

def extract_code_metadata(
    code_chunk: str,
    model: GenerativeModel
) -> dict:
    """Extract metadata specific to code chunks."""
    prompt = f"""
    Analyze this code and provide:
    1. Programming language
    2. Main functionality
    3. Key components/libraries used
    4. Complexity level (Simple/Medium/Complex)
    
    Code:
    {code_chunk}
    """
    
    try:
        response = model.generate_content(prompt)
        return {
            "raw_response": response.text,
            "language": extract_language_from_response(response.text),
            "functionality": extract_functionality_from_response(response.text),
            "components": extract_components_from_response(response.text),
            "complexity": extract_complexity_from_response(response.text)
        }
    except Exception as e:
        print(f"Error extracting code metadata: {e}")
        return {}

def extract_language_from_response(response: str) -> str:
    """Extract programming language from response."""
    try:
        # Look for language indicators
        language_patterns = [
            r"language:\s*(\w+)",
            r"programming language:\s*(\w+)",
            r"written in\s*(\w+)",
        ]
        
        for pattern in language_patterns:
            if match := re.search(pattern, response.lower()):
                return match.group(1)
        
        return "unknown"
    except Exception:
        return "unknown"

def extract_functionality_from_response(response: str) -> str:
    """Extract main functionality description."""
    try:
        # Look for functionality indicators
        patterns = [
            r"functionality:\s*(.+?)(?:\n|$)",
            r"main function:\s*(.+?)(?:\n|$)",
            r"purpose:\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, response, re.IGNORECASE):
                return match.group(1).strip()
        
        return ""
    except Exception:
        return ""

def extract_components_from_response(response: str) -> List[str]:
    """Extract key components/libraries."""
    try:
        # Look for component indicators
        patterns = [
            r"components:\s*(.+?)(?:\n|$)",
            r"libraries:\s*(.+?)(?:\n|$)",
            r"dependencies:\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, response, re.IGNORECASE):
                components = match.group(1).strip()
                return [c.strip() for c in re.split(r'[,;]', components) if c.strip()]
        
        return []
    except Exception:
        return []

def extract_complexity_from_response(response: str) -> str:
    """Extract complexity level."""
    try:
        # Look for complexity indicators
        if match := re.search(r"complexity:?\s*(\w+)", response, re.IGNORECASE):
            complexity = match.group(1).lower()
            if complexity in ['simple', 'medium', 'complex']:
                return complexity
        return "unknown"
    except Exception:
        return "unknown"

# Add these QA system extensions:

class ExtendedQASystem(QASystem):
    """Extended QA system with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_history = []
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> dict:
        """Load common error patterns and solutions."""
        return {
            r"importerror: no module named '(\w+)'": {
                "type": "import_error",
                "solution": "Install the missing package using pip: pip install {}"
            },
            r"syntaxerror: invalid syntax": {
                "type": "syntax_error",
                "solution": "Check for missing parentheses, quotes, or incorrect indentation"
            },
            r"indentationerror: ": {
                "type": "indentation_error",
                "solution": "Fix the indentation to use consistent spaces/tabs"
            }
        }
    
    def analyze_error(self, error_message: str) -> dict:
        """Analyze error message for known patterns."""
        for pattern, info in self.error_patterns.items():
            if match := re.search(pattern, error_message.lower()):
                solution = info["solution"]
                if "{}" in solution and match.groups():
                    solution = solution.format(*match.groups())
                return {
                    "type": info["type"],
                    "pattern_matched": pattern,
                    "solution": solution
                }
        return {}
    
    async def debug_code(self, code: str, error_message: str) -> dict:
        """Debug code with error message context."""
        context = f"""
        Code:
        {code}
        
        Error:
        {error_message}
        """
        
        # First check for known patterns
        if analysis := self.analyze_error(error_message):
            return {
                "analysis": analysis,
                "suggestion": analysis["solution"]
            }
        
        # If no known pattern, use QA chain
        response = await self.query(
            f"Debug this code error: {error_message}",
            metadata_filter={"type": "code"}
        )
        
        return {
            "analysis": {"type": "unknown"},
            "suggestion": response["answer"]
        }
    
    def add_code_example(self, code: str, metadata: dict) -> None:
        """Add a code example to the history."""
        self.code_history.append({
            "code": code,
            "metadata": metadata,
            "timestamp": time.time()
        })
    
    def get_similar_examples(self, code: str, top_k: int = 3) -> List[dict]:
        """Find similar code examples from history."""
        if not self.code_history:
            return []
        
        # Get embedding for query code
        query_embedding = embed_code_block(code)
        
        # Get embeddings for historical code
        historical_embeddings = embed_code_retrieval([h["code"] for h in self.code_history])
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], historical_embeddings)[0]
        
        # Get top-k similar examples
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            {
                "code": self.code_history[i]["code"],
                "metadata": self.code_history[i]["metadata"],
                "similarity": similarities[i]
            }
            for i in top_indices
        ]
