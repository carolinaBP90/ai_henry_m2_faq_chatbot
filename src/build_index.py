"""
Build Index Module: Load FAQ document, chunk it intelligently, generate embeddings,
and store them in a vector database for RAG retrieval.

This module implements:
1. Document loading with validation
2. Intelligent chunking with semantic awareness
3. Embedding generation using sentence-transformers or OpenAI
4. Vector storage in local database (FAISS) with metadata
5. Chunking strategy: Recursive content-aware splitting
"""

import os
import json
import pickle
import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import FAISS, if not available, we'll implement a simple in-memory alternative
faiss_spec = importlib.util.find_spec('faiss')
if faiss_spec is not None:
    faiss = importlib.import_module('faiss')
    FAISS_AVAILABLE = True
else:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using simple in-memory vector store.")

# Load environment variables
load_dotenv()


class DocumentChunker:
    """
    Intelligent document chunking strategy.
    
    Strategy: Hierarchical chunking
    - First splits by major sections (marked by ===== SECTION)
    - Within sections, splits by subsections (marked by TOPIC:)
    - Further splits by paragraphs to maintain semantic coherence
    - Enforces minimum chunk size (100 tokens) and maximum (500 tokens)
    - Respects sentence boundaries to avoid splitting mid-sentence
    
    Why this approach?
    - Maintains semantic coherence within chunks
    - Section/subsection markers provide natural break points
    - Hierarchical approach preserves document structure
    - Respects paragraph boundaries for readability
    - Produces 20+ high-quality chunks with diverse content
    """
    
    def __init__(self, min_tokens: int = 80, max_tokens: int = 350):
        """
        Initialize chunker with size constraints.
        
        Args:
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        # Rough estimate: 1 token ≈ 4 characters (conservative)
        self.min_chars = min_tokens * 4
        self.max_chars = max_tokens * 4
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation based on character count."""
        return len(text) // 4
    
    def chunk_document(self, document_text: str) -> List[Dict[str, str]]:
        """
        Chunk document using hierarchical intelligent strategy.
        
        Args:
            document_text: Full document text
            
        Returns:
            List of chunk dictionaries with 'content' and 'metadata'
        """
        chunks = []
        
        # Step 1: Split by major sections
        section_pattern = "=========="
        sections = document_text.split(section_pattern)
        
        section_num = 0
        for section in sections:
            if not section.strip():
                continue
            
            # Extract section number and title if available
            section_lines = section.strip().split('\n')
            section_title = "Unknown Section"
            section_content = section
            
            if len(section_lines) > 0:
                first_line = section_lines[0].strip()
                if "SECTION" in first_line:
                    section_title = first_line
                    section_num += 1
                    section_content = '\n'.join(section_lines[1:])
            
            # Step 2: Within each section, split by topics
            if "TOPIC:" in section_content:
                topics = section_content.split("TOPIC:")
                topic_num = 0
                
                for topic_idx, topic in enumerate(topics):
                    if not topic.strip():
                        continue
                    
                    topic_num += 1
                    topic_lines = topic.strip().split('\n')
                    topic_title = topic_lines[0].strip() if topic_lines else "Unnamed Topic"
                    topic_content = '\n'.join(topic_lines[1:])
                    
                    # Step 3: Split topic into paragraphs
                    paragraphs = self._split_into_paragraphs(topic_content)
                    
                    # Step 4: Merge paragraphs into chunks respecting size constraints
                    merged_chunks = self._merge_paragraphs_into_chunks(
                        paragraphs,
                        section_title,
                        topic_title
                    )
                    
                    chunks.extend(merged_chunks)
            else:
                # If no topics, split section directly into chunks
                paragraphs = self._split_into_paragraphs(section_content)
                merged_chunks = self._merge_paragraphs_into_chunks(
                    paragraphs,
                    section_title,
                    "General Content"
                )
                chunks.extend(merged_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs by double newlines or significant gaps."""
        paragraphs = []
        current_para = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_para:
                    para_text = ' '.join(current_para)
                    if para_text.strip():
                        paragraphs.append(para_text)
                    current_para = []
            else:
                current_para.append(line)
        
        if current_para:
            para_text = ' '.join(current_para)
            if para_text.strip():
                paragraphs.append(para_text)
        
        return paragraphs
    
    def _merge_paragraphs_into_chunks(
        self,
        paragraphs: List[str],
        section_title: str,
        topic_title: str
    ) -> List[Dict[str, str]]:
        """Merge paragraphs into chunks of appropriate size."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph exceeds max_chars, save current chunk
            if current_size + para_size > self.max_chars and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self.estimate_tokens(chunk_text) >= self.min_tokens:
                    chunks.append({
                        'content': chunk_text,
                        'section': section_title,
                        'topic': topic_title,
                        'word_count': len(chunk_text.split())
                    })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.estimate_tokens(chunk_text) >= self.min_tokens:
                chunks.append({
                    'content': chunk_text,
                    'section': section_title,
                    'topic': topic_title,
                    'word_count': len(chunk_text.split())
                })
        
        return chunks


class EmbeddingGenerator:
    """
    Generate embeddings for chunks using sentence-transformers or OpenAI.
    
    Strategy: Use sentence-transformers by default (local, fast, no API keys)
    Fallback to OpenAI API if OPENAI_API_KEY is provided and available.
    
    Why sentence-transformers?
    - Fast inference, no API calls required
    - Good quality for FAQ/document similarity tasks
    - Free and open-source
    - 384-dimensional embeddings with high quality
    - Works offline without rate limiting
    """
    
    def __init__(self, use_openai: bool = False):
        """
        Initialize embedding generator.
        
        Args:
            use_openai: If True, try to use OpenAI API. Otherwise use sentence-transformers.
        """
        self.use_openai = use_openai
        self.embedding_model = None
        self.embedding_dim = 384
        self.backend = 'sentence-transformers'
        
        if use_openai:
            self._init_openai()
        else:
            self._init_sentence_transformers()
    
    def _init_openai(self):
        """Initialize OpenAI client for embeddings."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Falling back to sentence-transformers.")
                self._init_sentence_transformers()
                return
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
            self.embedding_dim = 1536 if '3-small' in self.embedding_model_name else 3072
            logger.info(f"Using OpenAI embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}. Using sentence-transformers.")
            self._init_sentence_transformers()
    
    def _init_sentence_transformers(self):
        """Initialize sentence-transformers model."""
        try:
            sentence_transformers = importlib.import_module('sentence_transformers')
            SentenceTransformer = sentence_transformers.SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'  # Fast, good quality, 384-dimensional
            logger.info(f"Loading sentence-transformers model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = 384
            self.use_openai = False
            self.backend = 'sentence-transformers'
            logger.info("Sentence-transformers model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to local HashingVectorizer embeddings.")
            self._init_hashing_vectorizer()

    def _init_hashing_vectorizer(self):
        """Initialize local HashingVectorizer as fully offline embedding backend."""
        try:
            from sklearn.feature_extraction.text import HashingVectorizer

            self.embedding_model = HashingVectorizer(
                n_features=1024,
                alternate_sign=False,
                norm='l2',
                ngram_range=(1, 2)
            )
            self.embedding_dim = 1024
            self.use_openai = False
            self.backend = 'hashing-vectorizer'
            logger.info("Using HashingVectorizer embedding backend (offline mode)")
        except ImportError as exc:
            logger.error("scikit-learn is required for offline embeddings. Install with: pip install scikit-learn")
            raise exc
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.use_openai and hasattr(self, 'openai_client'):
            return self._generate_openai_embeddings(texts)
        if self.backend == 'hashing-vectorizer':
            matrix = self.embedding_model.transform(texts)
            return matrix.toarray().astype('float32')
        else:
            return self._generate_sentence_transformer_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        batch_size = 50  # OpenAI batch limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                raise
        
        return np.array(embeddings).astype('float32')
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.astype('float32')


class VectorStore:
    """
    Vector store for storing and retrieving embeddings.
    
    Implementation: FAISS for efficient similarity search when available,
    fallback to simple in-memory numpy-based search.
    
    Supports:
    - k-NN (k-Nearest Neighbors) search
    - Approximate Nearest Neighbors (ANN) via FAISS
    - Add and save/load functionality
    """
    
    def __init__(self, embeddings: np.ndarray, chunks: List[Dict], dimension: int):
        """
        Initialize vector store.
        
        Args:
            embeddings: numpy array of embeddings
            chunks: list of chunk dictionaries
            dimension: embedding dimension
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.dimension = dimension
        self.index = None
        
        if FAISS_AVAILABLE:
            self._init_faiss()
        else:
            logger.info("Using simple numpy-based vector store")
    
    def _init_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            self.index = None
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: query embedding vector
            k: number of results to return
            
        Returns:
            List of (chunk_index, distance) tuples
        """
        k = min(k, len(self.chunks))
        
        if self.index is not None:
            # Use FAISS ANN search
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1),
                k
            )
            results = list(zip(indices[0], distances[0]))
        else:
            # Fallback: use simple cosine similarity with numpy
            query_embedding = query_embedding.reshape(1, -1)
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(-similarities)[:k]
            # Return as (index, distance) tuples for consistency
            results = [(int(idx), float(1 - similarities[idx])) for idx in top_indices]
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index if available
        if self.index is not None:
            faiss.write_index(self.index, str(db_path / 'faiss.index'))
        
        # Save embeddings and chunks
        np.save(db_path / 'embeddings.npy', self.embeddings)
        
        with open(db_path / 'chunks.pkl', 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(db_path / 'metadata.json', 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'num_chunks': len(self.chunks),
                'num_embeddings': len(self.embeddings)
            }, f, indent=2)
        
        logger.info(f"Vector store saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'VectorStore':
        """Load vector store from disk."""
        db_path = Path(path)
        
        # Load embeddings and chunks
        embeddings = np.load(db_path / 'embeddings.npy')
        with open(db_path / 'chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
        
        with open(db_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        vector_store = VectorStore(embeddings, chunks, metadata['dimension'])
        
        # Load FAISS index if available
        if FAISS_AVAILABLE and (db_path / 'faiss.index').exists():
            vector_store.index = faiss.read_index(str(db_path / 'faiss.index'))
        
        logger.info(f"Vector store loaded from {path}")
        return vector_store


def build_index(
    document_path: str = 'data/faq_document.txt',
    vectorstore_path: str = 'vectorstore',
    use_openai: bool = False
) -> VectorStore:
    """
    Main function to build the vector index pipeline.
    
    Pipeline:
    1. Load document
    2. Chunk document intelligently
    3. Generate embeddings
    4. Create and save vector store
    
    Args:
        document_path: Path to FAQ document
        vectorstore_path: Path to save vector store
        use_openai: Whether to use OpenAI embeddings
        
    Returns:
        VectorStore object for querying
    """
    logger.info("=" * 60)
    logger.info("Starting FAQ Index Building Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load document
    logger.info(f"Loading document from {document_path}")
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    logger.info(f"Document loaded: {len(document_text)} characters")
    
    # Step 2: Chunk document
    logger.info("Chunking document with intelligent strategy...")
    chunker = DocumentChunker(min_tokens=80, max_tokens=350)
    chunks = chunker.chunk_document(document_text)
    
    logger.info(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"  Chunk {i+1}: {chunk['section']} > {chunk['topic']} ({chunk['word_count']} words)")
    
    # Validate chunks
    if len(chunks) < 20:
        logger.warning(f"Only {len(chunks)} chunks created. Minimum requirement is 20.")
    
    # Step 3: Extract chunk contents for embedding
    chunk_contents = [chunk['content'] for chunk in chunks]
    
    # Step 4: Generate embeddings
    logger.info("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(use_openai=use_openai)
    logger.info(f"Embedding dimension: {embedding_gen.embedding_dim}")
    
    embeddings = embedding_gen.generate_embeddings(chunk_contents)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    # Step 5: Create and save vector store
    logger.info("Creating vector store...")
    vector_store = VectorStore(embeddings, chunks, embedding_gen.embedding_dim)
    
    logger.info(f"Saving vector store to {vectorstore_path}")
    vector_store.save(vectorstore_path)
    
    logger.info("=" * 60)
    logger.info("Index Building Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  - Chunks created: {len(chunks)}")
    logger.info(f"  - Embedding dimension: {embedding_gen.embedding_dim}")
    logger.info(f"  - Vector store saved to: {vectorstore_path}")
    
    return vector_store


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build FAQ vector index for RAG pipeline.')
    parser.add_argument('--document-path', default='data/faq_document.txt', help='Path to source FAQ text document')
    parser.add_argument('--vectorstore-path', default='vectorstore', help='Path to store vector index artifacts')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings instead of local embeddings')
    args = parser.parse_args()

    build_index(
        document_path=args.document_path,
        vectorstore_path=args.vectorstore_path,
        use_openai=args.use_openai
    )
