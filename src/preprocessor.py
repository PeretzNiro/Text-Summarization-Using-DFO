"""
Document preprocessing and feature extraction module.

This module provides:
- Document text cleaning and normalization
- Sentence segmentation and tokenization
- Feature extraction (TF-IDF, embeddings)
- Parallel batch processing capabilities
"""

from typing import List, Generator, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re
import spacy
import pickle
from tqdm import tqdm
import numpy as np
import multiprocessing
from .utils import Timer

class Document:
    """Document representation with extracted features.
    
    This class handles:
    - Text cleaning and normalization
    - Sentence segmentation
    - Feature extraction and storage
    - Embedding computation
    """
    
    def __init__(self, article_id: str, article: str, highlights: str):
        """Initialize document with text and metadata.
        
        Args:
            article_id: Unique identifier for the document
            article: Original article text
            highlights: Reference summary text
            
        Raises:
            ValueError: If article text is empty or invalid
        """
        if not article or not article.strip():
            raise ValueError("Empty article text")
            
        self.id = article_id
        self.original_text = article
        self.highlights = highlights
        self.sentences = []
        self.features = defaultdict(dict)
        self.sentence_embeddings = {}
        self.spacy_doc = None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize article text.
        
        Performs:
        - Removal of location prefixes
        - CNN prefix removal
        - Special character handling
        - Whitespace normalization
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        # Remove location and CNN prefixes
        text = re.sub(r'\([^)]*\)\s*--\s*', '', text)
        text = re.sub(r'\(CNN\)\s*', '', text)
        
        # Handle special cases
        text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '. ', text)  # Fix abbreviations
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        return text.strip()

    def validate_features(self) -> bool:
        """Validate that document has all required features.
        
        Returns:
            bool: True if all required features are present
        """
        required_features = {
            'position', 'length', 'tfidf', 'noun_phrases',
            'verb_phrases', 'proper_nouns', 'cosine_sim',
            'big_vector_sim'
        }
        has_all_features = all(feat in self.features for feat in required_features)
        
        # Also check that each feature has values for all sentences
        if has_all_features:
            n_sentences = len(self.sentences)
            for feature in required_features:
                if len(self.features[feature]) != n_sentences:
                    return False
                    
        return has_all_features

    def get_feature_vector(self) -> np.ndarray:
        """Get normalized feature matrix."""
        if not self.validate_features():
            raise ValueError("Document features are incomplete or invalid")
            
        n_sentences = len(self.sentences)
        n_features = len(self.features)
        feature_matrix = np.zeros((n_sentences, n_features))
        
        for feat_idx, (feat_name, feat_values) in enumerate(self.features.items()):
            for sent_idx, value in feat_values.items():
                feature_matrix[sent_idx, feat_idx] = value
                
        return feature_matrix

class DocumentBatchProcessor:
    """Parallel document processor for efficient batch processing.
    
    Features:
    - Multi-threaded document processing
    - TF-IDF feature extraction
    - Spacy-based linguistic feature extraction
    - Progress tracking
    """
    
    def __init__(self, batch_size: int = 32, n_jobs: int = -1, data_path: Optional[str] = None):
        """Initialize the batch processor.
        
        Args:
            batch_size: Number of documents to process in parallel
            n_jobs: Number of worker threads (-1 for all available)
            data_path: Optional path to dataset for TF-IDF initialization
            
        Raises:
            RuntimeError: If spaCy model loading fails
        """
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("Loaded spaCy model successfully")
        except Exception as e:
            print(f"Error loading spaCy model: {str(e)}")
            raise
        
        # Initialize TF-IDF
        if data_path:
            try:
                # Load all documents first to fit TF-IDF
                print("Loading documents for TF-IDF initialization...")
                loader = CNNDMDataLoader(data_path)
                all_texts = []
                
                # Create progress bar for TF-IDF initialization
                for batch in tqdm(loader.load_batch(batch_size=100),
                            desc="Loading documents for TF-IDF",
                            unit="batch"):
                    texts = [doc.original_text for doc in batch]
                    all_texts.extend(texts)
                
                print(f"Fitting TF-IDF on {len(all_texts)} documents...")
                self.tfidf = TfidfVectorizer()
                self.tfidf.fit(all_texts)
                print("TF-IDF initialization completed successfully")
                
            except Exception as e:
                print(f"Error initializing TF-IDF: {str(e)}")
                print("Falling back to batch-only TF-IDF")
                self.tfidf = None
        else:
            self.tfidf = None

    def process_batch(self, documents: List[Document]) -> List[Document]:
        """Process a batch of documents in parallel."""
        with Timer("Batch processing"):
            try:
                # If TF-IDF not initialized, initialize with documents
                if self.tfidf is None:
                    all_sentences = []
                                  
                    # Add progress bar for initial document processing
                    with tqdm(documents, 
                            desc="Preparing documents for TF-IDF", 
                            unit="doc",
                            position=1, 
                            leave=False) as pbar:
                        for doc in pbar:
                            doc.spacy_doc = self.nlp(doc.original_text)
                            doc.sentences = [sent.text.strip() for sent in doc.spacy_doc.sents]
                            all_sentences.extend(doc.sentences)
                    
                    if not all_sentences:
                        raise ValueError("No sentences found in documents")
                    
                    self.tfidf = TfidfVectorizer(max_features=1000)
                    self.tfidf.fit(all_sentences)

                processed_docs = []
                
                # Create progress bar for document processing
                with tqdm(total=len(documents), 
                        desc="Processing documents", 
                        unit="doc",
                        position=1, 
                        leave=False) as pbar:
                    
                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        # Submit preprocessing jobs
                        future_to_doc = {
                            executor.submit(self._process_single_document, doc): doc 
                            for doc in documents
                        }
                        
                        # Collect results
                        for future in future_to_doc:
                            try:
                                processed_doc = future.result()
                                if processed_doc and processed_doc.validate_features():
                                    processed_docs.append(processed_doc)
                                pbar.update(1)
                            except Exception:
                                pbar.update(1)
                                continue
                
                return processed_docs
                
            except Exception as e:
                raise

    def _process_single_document(self, doc: Document) -> Document:
        """Process a single document."""
        try:
            # Process with spaCy
            doc.spacy_doc = self.nlp(doc.original_text)
            
            # Extract and store sentences
            doc.sentences = [sent.text.strip() for sent in doc.spacy_doc.sents]
            
            if not doc.sentences:
                raise ValueError(f"No sentences found in document {doc.id}")
            
            # Extract features
            self._extract_features(doc)
            
            return doc
            
        except Exception as e:
            raise

    def _extract_features(self, doc: Document) -> None:
        """Extract features from document.
        
        Args:
            doc: Document to process
        """
        try:
            n_sents = len(doc.sentences)
            if n_sents == 0:
                raise ValueError("Document contains no sentences")
                
            # Position features
            for i in range(n_sents):
                doc.features['position'][i] = 1 - (i / n_sents)
            
            # Length features
            lengths = [len(sent.split()) for sent in doc.sentences]
            max_len = max(lengths) if lengths else 1
            for i, length in enumerate(lengths):
                doc.features['length'][i] = length / max_len
            
            # TF-IDF features
            tfidf_matrix = self.tfidf.transform(doc.sentences)
            tfidf_array = tfidf_matrix.toarray()
            for i, sent_vector in enumerate(tfidf_array):
                doc.features['tfidf'][i] = np.mean(sent_vector)
            
            # Linguistic features
            try:
                for i, sent in enumerate(doc.spacy_doc.sents):
                    # Noun phrases - only if parser is available
                    if not self.nlp.has_pipe('parser'):
                        doc.features['noun_phrases'][i] = 0
                    else:
                        doc.features['noun_phrases'][i] = len(list(sent.noun_chunks))
                    
                    # Verb phrases and proper nouns - use basic POS counting if tagger available
                    if not self.nlp.has_pipe('tagger'):
                        doc.features['verb_phrases'][i] = 0
                        doc.features['proper_nouns'][i] = 0
                    else:
                        doc.features['verb_phrases'][i] = len([t for t in sent if t.pos_ == "VERB"])
                        doc.features['proper_nouns'][i] = len([t for t in sent if t.pos_ == "PROPN"])
            except Exception as e:
                print(f"Error extracting linguistic features for document {doc.id}: {str(e)}")
                # Set default values if linguistic feature extraction fails
                for i in range(n_sents):
                    doc.features['noun_phrases'][i] = 0
                    doc.features['verb_phrases'][i] = 0
                    doc.features['proper_nouns'][i] = 0
            
            # Get sentence embeddings using spaCy
            embeddings = np.array([sent.vector for sent in doc.spacy_doc.sents])
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            normalized_embeddings = embeddings / norms
            
            # Cosine similarity features
            similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
            for i in range(n_sents):
                doc.features['cosine_sim'][i] = np.mean(similarities[i])
            
            # Document vector similarity
            doc_vector = np.mean(normalized_embeddings, axis=0)
            doc_vector_norm = np.linalg.norm(doc_vector)
            if doc_vector_norm > 0:
                doc_vector = doc_vector / doc_vector_norm
                for i, sent_embedding in enumerate(normalized_embeddings):
                    doc.features['big_vector_sim'][i] = np.dot(sent_embedding, doc_vector)
            else:
                print(f"Zero norm document vector in document {doc.id}")
                for i in range(n_sents):
                    doc.features['big_vector_sim'][i] = 0.0
            
            # Normalize all features to [0,1] range
            self._normalize_features(doc)

        except Exception as e:
            print(f"Error extracting features for document {doc.id}: {str(e)}")
            raise

    def _normalize_features(self, doc: Document) -> None:
        for feature_name, feature_values in doc.features.items():
            values = list(feature_values.values())
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:  # Only normalize if there's a range
                for idx in feature_values:
                    doc.features[feature_name][idx] = (
                        feature_values[idx] - min_val
                    ) / (max_val - min_val)
            else:
                # If all values are the same, set them to 0.5 or keep original
                for idx in feature_values:
                    doc.features[feature_name][idx] = 0.5  # or feature_values[idx]

class CNNDMDataLoader:
    """CNN/DailyMail dataset loader with preprocessing.
    
    Features:
    - Efficient batch loading
    - Pickle file handling
    - Text cleaning and normalization
    - Progress tracking
    """
    
    def __init__(self, data_path: str):
        """Initialize the data loader.
        
        Args:
            data_path: Path to dataset directory
            
        Raises:
            FileNotFoundError: If dataset path or pickle files not found
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        self.pickle_files = list(self.data_path.glob("*.pkl"))
        if not self.pickle_files:
            raise FileNotFoundError(f"No pickle files found in {data_path}")

    def load_batch(self, batch_size: int) -> Generator[List[Document], None, None]:
        """Load and yield batches of documents."""
        try:
            # Load pickle file
            pickle_file = next(self.data_path.glob("*.pkl"))
            with open(pickle_file, 'rb') as f:
                articles_data = pickle.load(f)
            
            current_batch = []
            for article in articles_data:
                try:
                    # Create document
                    doc = Document(
                        article['id'],
                        article['article'],
                        article['highlights']
                    )
                    current_batch.append(doc)
                    
                    # Yield batch when full
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
                        
                except Exception:
                    continue
            
            # Yield remaining documents
            if current_batch:
                yield current_batch
                
        except Exception as e:
            raise

    def _parse_story_file(self, file_path: Path) -> Tuple[str, str]:
        """Parse CNN/DailyMail story file.
        
        Args:
            file_path: Path to story file
            
        Returns:
            Tuple of (article_content, highlights)
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into article and highlights
            parts = content.split('@highlight')
            if len(parts) < 2:
                raise ValueError(f"Invalid story file format in {file_path}")
            
            article = parts[0].strip()
            highlights = ' '.join([part.strip() for part in parts[1:]])
            
            return article, highlights
            
        except Exception as e:
            raise