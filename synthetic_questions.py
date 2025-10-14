import os
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Set
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('synthetic_questions.log')
    ]
)
logger = logging.getLogger(__name__)


class SyntheticQuestionGenerator:
    def __init__(self, google_api_key: str = None):
        """Initialize the question generator with Google API key."""
        logger.info("Initializing SyntheticQuestionGenerator...")
        
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        logger.info("Loading Hugging Face embeddings model (all-MiniLM-L6-v2)...")
        # Use free Hugging Face embeddings - lightweight model for speed
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("✅ Embeddings model loaded successfully")
        
        logger.info("Initializing Google Gemini LLM...")
        # Use Google Gemini for text generation
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )
        logger.info("✅ Google Gemini LLM initialized successfully")
        
        self.vector_store = None
        self.documents = []
        
    def retry_with_backoff(self, func, max_retries=3, base_delay=20):
        """Retry function with exponential backoff for rate limits."""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                error_msg = str(e)
                if "quota_exceeded" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries:
                        # Extract retry delay from error message if available
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        if "retry in" in error_msg:
                            try:
                                # Extract the delay from error message
                                delay_str = error_msg.split("retry in ")[1].split("s")[0]
                                delay = float(delay_str) + 1  # Add 1 second buffer
                            except:
                                pass
                        
                        logger.warning(f"   ⏱️ Rate limit hit, waiting {delay:.1f}s before retry {attempt+1}/{max_retries}")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"   ❌ Max retries exceeded for rate limiting")
                        raise
                else:
                    # Non-rate-limit error, don't retry
                    raise
        return None
        
    def load_pdfs(self, pdf_directory: str) -> List[Dict]:
        """Load all PDF files from the specified directory."""
        logger.info(f"Scanning directory '{pdf_directory}' for PDF files...")
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        all_documents = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"📄 Loading PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
                logger.info(f"   ✅ Successfully loaded {len(documents)} pages from {pdf_path.name}")
                
                # Add metadata about the source file
                for doc in documents:
                    doc.metadata['source_file'] = pdf_path.name
                    
                all_documents.extend(documents)
                
            except Exception as e:
                logger.error(f"   ❌ Failed to load {pdf_path.name}: {str(e)}")
                continue
        
        logger.info(f"📚 Total documents loaded: {len(all_documents)} pages from {len(pdf_files)} PDFs")
        self.documents = all_documents
        return all_documents
    
    def create_vector_store(self) -> FAISS:
        """Create an in-memory FAISS vector store from the loaded documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_pdfs() first.")
        
        logger.info("🔄 Splitting documents into chunks for better processing...")
        # Split documents into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_documents = text_splitter.split_documents(self.documents)
        logger.info(f"   ✅ Split {len(self.documents)} documents into {len(split_documents)} chunks")
        
        logger.info("🧮 Creating FAISS vector store with embeddings...")
        logger.info("   (This may take a few minutes for large document sets)")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
        logger.info(f"   ✅ Vector store created successfully with {len(split_documents)} document chunks")
        
        return self.vector_store
    
    def process_documents_with_questions(self, questions_per_pdf: int = 10, progress_file: str = "synthetic_questions_progress.csv") -> List[Dict]:
        """Process each document: summarize -> generate questions -> save immediately."""
        logger.info("📝 Starting document processing (summarize + generate questions)...")
        all_questions = []
        
        # Clear progress file at start
        if Path(progress_file).exists():
            Path(progress_file).unlink()
            logger.info(f"   Cleared previous progress file: {progress_file}")
        
        # Group documents by source file
        logger.info("   Grouping documents by source file...")
        docs_by_file = {}
        for doc in self.documents:
            source_file = doc.metadata.get('source_file', 'unknown')
            if source_file not in docs_by_file:
                docs_by_file[source_file] = []
            docs_by_file[source_file].append(doc)
        
        logger.info(f"   Found {len(docs_by_file)} unique source files")
        
        # Process each file: summarize -> generate questions -> save
        for i, (source_file, docs) in enumerate(docs_by_file.items(), 1):
            logger.info(f"\n📄 Processing document {i}/{len(docs_by_file)}: {source_file}")
            logger.info(f"   Pages to process: {len(docs)}")
            
            try:
                # STEP A: Summarize the document
                logger.info("   🤖 Summarizing document...")
                
                def summarize_docs():
                    summary_chain = load_summarize_chain(
                        self.llm,
                        chain_type="map_reduce",
                        verbose=False
                    )
                    result = summary_chain.invoke({"input_documents": docs})
                    # Handle different response formats
                    if isinstance(result, dict) and 'output_text' in result:
                        return result['output_text']
                    elif hasattr(result, 'content'):
                        return result.content
                    else:
                        return str(result)
                
                summary = self.retry_with_backoff(summarize_docs)
                logger.info(f"   ✅ Summary completed ({len(summary)} characters)")
                
                # STEP B: Generate questions from the summary immediately
                logger.info(f"   ❓ Generating {questions_per_pdf} questions from summary...")
                questions = self.generate_questions_from_summary(
                    summary, 
                    source_file, 
                    questions_per_pdf
                )
                
                # STEP C: Create question objects and save immediately
                doc_questions = []
                for question in questions:
                    question_obj = {
                        'question': question,
                        'source_file': source_file,
                        'summary': summary[:200] + "..." if len(summary) > 200 else summary
                    }
                    doc_questions.append(question_obj)
                    all_questions.append(question_obj)
                
                # STEP D: Save to CSV immediately
                if doc_questions:
                    self.append_questions_to_csv(doc_questions, progress_file)
                    logger.info(f"   💾 Saved {len(doc_questions)} questions to CSV")
                    logger.info(f"   📊 Total questions so far: {len(all_questions)}")
                else:
                    logger.warning(f"   ⚠️ No questions generated for {source_file}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process {source_file}: {str(e)}")
                continue
        
        logger.info(f"\n📋 Completed processing {len(docs_by_file)} documents")
        logger.info(f"📊 Total questions generated: {len(all_questions)}")
        return all_questions
    
    def generate_questions_from_summary(self, summary: str, source_file: str, num_questions: int = 5) -> List[str]:
        """Generate questions from a document summary."""
        logger.info(f"❓ Generating {num_questions} questions from {source_file}")
        logger.info(f"   Summary length: {len(summary)} characters")
        
        question_prompt = PromptTemplate(
            input_variables=["summary", "num_questions"],
            template="""
            Based on the following document summary, generate {num_questions} diverse and meaningful questions that test understanding of the key concepts, facts, and insights from the document.
            
            The questions should be:
            - Clear and well-formed
            - Cover different aspects of the content
            - Range from factual recall to analytical thinking
            - Be answerable based on the document content
            
            Document Summary:
            {summary}
            
            Generate exactly {num_questions} questions, one per line:
            """
        )
        
        formatted_prompt = question_prompt.format(
            summary=summary,
            num_questions=num_questions
        )
        
        logger.info("   🤖 Calling Google Gemini to generate questions...")
        try:
            def generate_response():
                return self.llm.invoke(formatted_prompt)
            
            response = self.retry_with_backoff(generate_response)
            # Handle response - it might be a message object
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.info(f"   ✅ Received response from LLM ({len(response_text)} characters)")
            
            # Parse questions from response
            questions = [q.strip() for q in response_text.split('\n') if q.strip() and not q.strip().isdigit()]
            
            # Filter out empty questions and numbering artifacts
            clean_questions = []
            for q in questions:
                # Remove leading numbers/bullets
                q = q.lstrip('0123456789.-) ').strip()
                if len(q) > 10 and q.endswith('?'):
                    clean_questions.append(q)
            
            final_questions = clean_questions[:num_questions]
            logger.info(f"   ✅ Generated {len(final_questions)} clean questions for {source_file}")
            
            return final_questions
            
        except Exception as e:
            logger.error(f"   ❌ Failed to generate questions for {source_file}: {str(e)}")
            return []
    
    def deduplicate_questions(self, questions: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """Remove duplicate or very similar questions using TF-IDF and cosine similarity."""
        logger.info(f"🔍 Starting deduplication of {len(questions)} questions...")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        
        if not questions:
            logger.info("   No questions to deduplicate")
            return questions
        
        # Extract question texts
        question_texts = [q['question'] for q in questions]
        logger.info("   Creating TF-IDF vectors for similarity analysis...")
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform(question_texts)
            
            logger.info("   Calculating cosine similarity matrix...")
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            logger.info("   Identifying duplicate questions...")
            # Find questions to keep (avoid duplicates)
            to_keep = []
            removed_indices = set()
            
            for i, question in enumerate(questions):
                if i in removed_indices:
                    continue
                    
                to_keep.append(question)
                
                # Mark similar questions for removal
                similar_count = 0
                for j in range(i + 1, len(questions)):
                    if j not in removed_indices and similarity_matrix[i][j] > similarity_threshold:
                        removed_indices.add(j)
                        similar_count += 1
                
                if similar_count > 0:
                    logger.info(f"   Question {i+1} had {similar_count} similar questions removed")
            
            logger.info(f"   ✅ Deduplication complete: {len(questions)} -> {len(to_keep)} questions")
            logger.info(f"   Removed {len(questions) - len(to_keep)} duplicate/similar questions")
            return to_keep
            
        except Exception as e:
            logger.error(f"   ❌ Deduplication failed: {str(e)}")
            logger.info("   Returning original questions without deduplication")
            return questions
    
    def export_to_csv(self, questions: List[Dict], output_file: str = "synthetic_questions.csv"):
        """Export questions to CSV file."""
        logger.info(f"💾 Exporting {len(questions)} questions to CSV...")
        logger.info(f"   Output file: {output_file}")
        
        try:
            df = pd.DataFrame(questions)
            df.to_csv(output_file, index=False)
            
            # Log some statistics
            unique_sources = df['source_file'].nunique() if 'source_file' in df.columns else 0
            avg_summary_length = df['summary'].str.len().mean() if 'summary' in df.columns else 0
            
            logger.info(f"   ✅ Successfully exported to {output_file}")
            logger.info(f"   Questions: {len(questions)}")
            logger.info(f"   Unique source files: {unique_sources}")
            logger.info(f"   Average summary length: {avg_summary_length:.0f} characters")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to export CSV: {str(e)}")
    
    def append_questions_to_csv(self, questions: List[Dict], output_file: str = "synthetic_questions_progress.csv"):
        """Append questions to CSV file incrementally."""
        logger.info(f"💾 Appending {len(questions)} questions to progress CSV...")
        
        try:
            file_exists = Path(output_file).exists()
            
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                if questions:
                    fieldnames = questions[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header only if file is new
                    if not file_exists:
                        writer.writeheader()
                        logger.info(f"   Created new progress file: {output_file}")
                    
                    # Write the questions
                    for question in questions:
                        writer.writerow(question)
                    
                    logger.info(f"   ✅ Appended {len(questions)} questions to {output_file}")
                    
        except Exception as e:
            logger.error(f"   ❌ Failed to append to CSV: {str(e)}")
    
    def generate_synthetic_questions(self, pdf_directory: str, questions_per_pdf: int = 10, 
                                   similarity_threshold: float = 0.8, output_file: str = "synthetic_questions.csv"):
        """Main method to generate synthetic questions from PDFs."""
        
        logger.info("🚀 Starting synthetic question generation process...")
        logger.info(f"   PDF directory: {pdf_directory}")
        logger.info(f"   Questions per PDF: {questions_per_pdf}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Output file: {output_file}")
        
        start_time = logger.info("Process started")
        
        try:
            # Load PDFs
            logger.info("\n📚 STEP 1: Loading PDFs...")
            self.load_pdfs(pdf_directory)
            
            # Create vector store
            logger.info("\n🧮 STEP 2: Creating vector store...")
            self.create_vector_store()
            
            # Process documents: summarize + generate questions + save immediately
            logger.info("\n📝 STEP 3: Processing documents (summarize + generate questions)...")
            progress_file = "synthetic_questions_progress.csv"
            all_questions = self.process_documents_with_questions(questions_per_pdf, progress_file)
            
            # Deduplicate questions
            logger.info("\n🔍 STEP 4: Deduplicating questions...")
            unique_questions = self.deduplicate_questions(all_questions, similarity_threshold)
            
            # Export final deduplicated results to CSV
            logger.info("\n💾 STEP 5: Exporting final results to CSV...")
            self.export_to_csv(unique_questions, output_file)
            
            logger.info(f"\n🎉 PROCESS COMPLETE!")
            logger.info(f"   Final result: {len(unique_questions)} unique questions")
            logger.info(f"   Final output saved to: {output_file}")
            logger.info(f"   Progress file: {progress_file}")
            logger.info(f"   Log saved to: synthetic_questions.log")
            
            return unique_questions
            
        except Exception as e:
            logger.error(f"\n💥 PROCESS FAILED: {str(e)}")
            logger.error("   Check the log file for detailed error information")
            raise


def main():
    """Main function to run the synthetic question generator."""
    
    # Configuration
    PDF_DIRECTORY = "pdfs"  # Directory containing PDF files
    QUESTIONS_PER_PDF = 10  # Number of questions to generate per PDF
    SIMILARITY_THRESHOLD = 0.8  # Threshold for deduplication (0.0 = no duplicates, 1.0 = identical)
    OUTPUT_FILE = "synthetic_questions.csv"
    
    logger.info("="*60)
    logger.info("🎯 SYNTHETIC QUESTION GENERATOR")
    logger.info("="*60)
    
    try:
        # Initialize generator
        logger.info("🔧 Initializing generator...")
        generator = SyntheticQuestionGenerator()
        
        # Generate questions
        questions = generator.generate_synthetic_questions(
            pdf_directory=PDF_DIRECTORY,
            questions_per_pdf=QUESTIONS_PER_PDF,
            similarity_threshold=SIMILARITY_THRESHOLD,
            output_file=OUTPUT_FILE
        )
        
        logger.info("="*60)
        logger.info(f"✅ SUCCESS! Generated questions saved to: {OUTPUT_FILE}")
        logger.info(f"📊 Total unique questions: {len(questions)}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"💥 ERROR: {e}")
        logger.error("="*60)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())