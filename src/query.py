"""
Query Pipeline Module: Accept user questions, perform vector search,
retrieve relevant chunks, generate responses with LLM, and return structured JSON.

This module implements:
1. Question embedding and vector search (k-NN/ANN via FAISS)
2. Chunk retrieval with metadata and relevance scoring
3. LLM-based response generation
4. Response evaluator agent (optional) for quality assessment
5. Structured JSON output with user_question, system_answer, chunks_related
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import from build_index for reuse
try:
    from .build_index import EmbeddingGenerator, VectorStore
except ImportError:
    from build_index import EmbeddingGenerator, VectorStore


class RAGQueryEngine:
    """
    RAG Query Engine: Combines vector search with LLM for response generation.
    
    Pipeline:
    1. Convert user question to embedding
    2. Vector search for k most relevant chunks (k-NN via FAISS or numpy)
    3. Apply hybrid relevance scoring (semantic + lexical)
    4. Retrieve context from top chunks
    5. Generate response using LLM
    6. Format response as structured JSON
    
    Why this architecture?
    - Vector search (k-NN/ANN) fast retrieves semantically similar chunks
    - Hybrid scoring combines semantic and lexical similarity
    - LLM augmentation enables nuanced, coherent answers
    - Structured JSON output ensures transparency and auditability
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_gen: EmbeddingGenerator,
        use_openai: bool = False,
        top_k: int = 5
    ):
        """
        Initialize RAG Query Engine.
        
        Args:
            vector_store: VectorStore with indexed chunks and embeddings
            embedding_gen: EmbeddingGenerator for query embedding
            use_openai: Whether to use OpenAI for output generation
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedding_gen = embedding_gen
        self.use_openai = use_openai
        self.top_k = top_k
        self.llm_client = None
        
        if use_openai:
            self._init_openai_llm()
    
    def _init_openai_llm(self):
        """Initialize OpenAI client for response generation."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Using rule-based responses.")
                return
            
            self.llm_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI LLM client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}. Using rule-based responses.")
    
    def query(self, user_question: str, top_k: Optional[int] = None) -> Dict:
        """
        Process a user question through the RAG pipeline.
        
        Args:
            user_question: User's question text
            top_k: Override default number of chunks to retrieve
            
        Returns:
            Dictionary with user_question, system_answer, chunks_related, metadata
        """
        if not user_question or not user_question.strip():
            return {
                'user_question': user_question,
                'system_answer': 'Error: Empty question provided.',
                'chunks_related': [],
                'metadata': {
                    'retrieval_method': 'k-NN (FAISS) if available, else cosine similarity',
                    'chunks_used': 0,
                    'error': 'Empty question'
                }
            }
        
        k = top_k or self.top_k
        
        # Step 1: Embed the question
        question_embedding = self.embedding_gen.generate_embeddings([user_question])[0]
        logger.info(f"Question embedded: {question_embedding.shape}")
        
        # Step 2: Vector search for relevant chunks
        relevant_chunks = self._retrieve_chunks(question_embedding, k)
        logger.info(f"Retrieved {len(relevant_chunks)} chunks")
        
        # Step 3: Generate response
        system_answer = self._generate_response(user_question, relevant_chunks)
        
        # Step 4: Format output
        return {
            'user_question': user_question,
            'system_answer': system_answer,
            'chunks_related': relevant_chunks,
            'metadata': {
                'retrieval_method': 'k-NN (FAISS) if available, else cosine similarity',
                'chunks_used': len(relevant_chunks),
                'top_k_requested': k
            }
        }
    
    def _retrieve_chunks(
        self,
        question_embedding: np.ndarray,
        k: int
    ) -> List[Dict]:
        """
        Retrieve top-k relevant chunks using vector similarity.
        
        Args:
            question_embedding: Embedded question vector
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunk dictionaries with relevance scores
        """
        # Perform k-NN search
        results = self.vector_store.search(question_embedding, k)
        
        relevant_chunks = []
        for rank, (chunk_idx, distance) in enumerate(results, 1):
            if chunk_idx >= len(self.vector_store.chunks):
                continue
            
            chunk = self.vector_store.chunks[chunk_idx]
            
            # Convert distance to relevance score (0-100)
            # For L2 distance, closer is better (lower distance)
            # For cosine similarity, normalize to 0-100 range
            relevance_score = max(0, 100 - (distance * 50))
            
            relevant_chunks.append({
                'rank': rank,
                'content': chunk['content'],
                'section': chunk.get('section', 'Unknown'),
                'topic': chunk.get('topic', 'Unknown'),
                'relevance_score': round(relevance_score, 2),
                'word_count': chunk.get('word_count', 0)
            })
        
        return relevant_chunks
    
    def _generate_response(
        self,
        user_question: str,
        relevant_chunks: List[Dict]
    ) -> str:
        """
        Generate response using retrieved chunks and optional LLM.
        
        Strategy:
        - If OpenAI configured: Use GPT to synthesize coherent answer from chunks
        - Otherwise: Use rule-based response combining chunk contents
        
        Args:
            user_question: User's original question
            relevant_chunks: List of relevant chunks with content
            
        Returns:
            Generated response string
        """
        if not relevant_chunks:
            return "I could not find relevant information in the FAQ database to answer your question. Please contact support."
        
        # Combine chunk contents into context
        context_parts = [
            f"- [{chunk['section']} > {chunk['topic']}]: {chunk['content'][:200]}..."
            for chunk in relevant_chunks
        ]
        context = "\n".join(context_parts)
        
        if self.llm_client:
            return self._generate_response_with_llm(user_question, context, relevant_chunks)
        else:
            return self._generate_response_rule_based(user_question, context, relevant_chunks)
    
    def _generate_response_with_llm(
        self,
        user_question: str,
        context: str,
        relevant_chunks: List[Dict]
    ) -> str:
        """Generate response using OpenAI LLM."""
        try:
            prompt = f"""You are a helpful HR support assistant. Answer the following question based on the provided FAQ content.

User Question: {user_question}

Relevant FAQ Content:
{context}

Instructions:
- Answer directly and concisely
- Use information from the FAQ content
- If the FAQ content doesn't answer the question, say so
- Provide a helpful response (max 200 words)

Answer:"""
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful HR support assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response with LLM: {e}")
            return self._generate_response_rule_based(user_question, context, relevant_chunks)
    
    def _generate_response_rule_based(
        self,
        user_question: str,
        context: str,
        relevant_chunks: List[Dict]
    ) -> str:
        """Generate response using rule-based approach."""
        # Build response from chunk contents
        response_parts = [
            f"Based on our FAQ documentation, here's what I found:\n"
        ]
        
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            relevance = chunk['relevance_score']
            response_parts.append(
                f"\n{i}. [Relevance: {relevance}%] {chunk['section']} - {chunk['topic']}"
            )
            # Add first 150 chars of chunk content
            content_preview = chunk['content'][:150]
            response_parts.append(f"   {content_preview}...")
        
        response_parts.append(
            f"\n\nFor more detailed information, please contact our support team or visit the knowledge base."
        )
        
        return "".join(response_parts)


class ResponseEvaluator:
    """
    Quality Evaluator Agent for RAG Responses.
    
    Evaluates response quality on multiple dimensions:
    - Relevance: Do retrieved chunks answer the user question?
    - Completeness: Does response cover all important aspects?
    - Accuracy: Is the information correct and precise?
    - Coherence: Is the response well-structured and understandable?
    
    Returns a score (0-10) and detailed justification.
    
    Why this evaluator?
    - Provides automated QA for RAG system
    - Identifies low-quality responses for human review
    - Tracks system quality over time
    - Enables continuous improvement
    """
    
    def __init__(self, use_openai: bool = False):
        """
        Initialize evaluator.
        
        Args:
            use_openai: Whether to use LLM-based evaluation
        """
        self.use_openai = use_openai
        self.llm_client = None
        
        if use_openai:
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI for LLM-based evaluation."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI LLM client initialized for evaluator")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI for evaluator: {e}")
    
    def evaluate(
        self,
        user_question: str,
        system_answer: str,
        chunks_related: List[Dict]
    ) -> Dict:
        """
        Evaluate response quality.
        
        Args:
            user_question: Original user question
            system_answer: Generated system answer
            chunks_related: Chunks used to generate answer
            
        Returns:
            Dictionary with score (0-10) and detailed justification
        """
        if self.llm_client:
            return self._evaluate_with_llm(user_question, system_answer, chunks_related)
        else:
            return self._evaluate_rule_based(user_question, system_answer, chunks_related)
    
    def _evaluate_with_llm(
        self,
        user_question: str,
        system_answer: str,
        chunks_related: List[Dict]
    ) -> Dict:
        """Evaluate using LLM-based scoring."""
        try:
            chunk_context = "\n".join([
                f"- [{c['section']}] {c['content'][:100]}..."
                for c in chunks_related[:3]
            ])
            
            prompt = f"""Evaluate the quality of this FAQ support response on a scale of 0-10.

Question: {user_question}
Answer: {system_answer}
Supporting Information:
{chunk_context}

Evaluate on:
1. Relevance: Does it answer the question?
2. Completeness: Are all important points covered?
3. Accuracy: Is the information correct?
4. Clarity: Is it well-explained?

Respond with JSON:
{{
    "score": <0-10>,
    "relevance": <0-10>,
    "completeness": <0-10>,
    "accuracy": <0-10>,
    "clarity": <0-10>,
    "justification": "<brief explanation>"
}}"""
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse JSON response
            import json as json_module
            result_text = response.choices[0].message.content
            # Extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json_module.loads(result_text[json_start:json_end])
                return result
        except Exception as e:
            logger.warning(f"Error in LLM evaluation: {e}. Using rule-based evaluation.")
        
        return self._evaluate_rule_based(user_question, system_answer, chunks_related)
    
    def _evaluate_rule_based(
        self,
        user_question: str,
        system_answer: str,
        chunks_related: List[Dict]
    ) -> Dict:
        """Rule-based evaluation using heuristics."""
        # Check if answer is empty
        if not system_answer or len(system_answer.strip()) < 10:
            return {
                'score': 1,
                'relevance': 1,
                'completeness': 1,
                'accuracy': 5,
                'clarity': 2,
                'justification': 'Answer is too short or empty'
            }
        
        # Heuristic scoring
        answer_len = len(system_answer.split())
        avg_chunk_score = np.mean([c.get('relevance_score', 50) for c in chunks_related]) if chunks_related else 50
        
        # Calculate scores based on heuristics
        relevance = min(10, int(avg_chunk_score / 10)) if chunks_related else 5
        completeness = min(10, int(answer_len / 30))  # More words = more complete
        accuracy = min(10, 7 + len(chunks_related))  # Higher with more sources
        clarity = min(10, 5 + (1 if len(system_answer.split('.')[:3]) < 15 else 2))
        
        overall_score = int(np.mean([relevance, completeness, accuracy, clarity]))
        
        return {
            'score': overall_score,
            'relevance': relevance,
            'completeness': completeness,
            'accuracy': accuracy,
            'clarity': clarity,
            'justification': f'Based on {len(chunks_related)} relevant chunks with avg relevance {avg_chunk_score:.1f}%. Answer quality assessed as good.' if overall_score >= 7 else f'Response quality needs improvement. Score: {overall_score}/10'
        }


def load_and_query(
    question: str,
    vectorstore_path: str = 'vectorstore',
    use_openai: bool = False,
    evaluate: bool = True
) -> Dict:
    """
    Complete pipeline: Load vector store and query with evaluation.
    
    Args:
        question: User question
        vectorstore_path: Path to saved vector store
        use_openai: Whether to use OpenAI APIs
        evaluate: Whether to evaluate response quality
        
    Returns:
        Complete response dictionary with metadata
    """
    logger.info(f"Loading vector store from {vectorstore_path}")
    
    # Load vector store
    vector_store = VectorStore.load(vectorstore_path)
    
    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator(use_openai=use_openai)
    
    # Create query engine
    rag_engine = RAGQueryEngine(
        vector_store=vector_store,
        embedding_gen=embedding_gen,
        use_openai=use_openai,
        top_k=5
    )
    
    # Query
    logger.info(f"Processing question: {question}")
    result = rag_engine.query(question)
    
    # Evaluate (optional)
    if evaluate:
        logger.info("Evaluating response quality")
        evaluator = ResponseEvaluator(use_openai=use_openai)
        evaluation = evaluator.evaluate(
            user_question=question,
            system_answer=result['system_answer'],
            chunks_related=result['chunks_related']
        )
        result['evaluation'] = evaluation
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query FAQ RAG system.')
    parser.add_argument('--question', required=True, help='User question to query the RAG system')
    parser.add_argument('--vectorstore-path', default='vectorstore', help='Path to vector index artifacts')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI for response generation/evaluation')
    parser.add_argument('--no-evaluate', action='store_true', help='Disable evaluator agent')
    parser.add_argument('--output-json', default='', help='Optional path to save output JSON')
    args = parser.parse_args()

    result = load_and_query(
        question=args.question,
        vectorstore_path=args.vectorstore_path,
        use_openai=args.use_openai,
        evaluate=not args.no_evaluate
    )

    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    print(formatted)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted, encoding='utf-8')
        logger.info(f"Result saved to {output_path}")
