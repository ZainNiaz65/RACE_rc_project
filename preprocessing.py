import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix, lil_matrix
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

class RACEPreprocessor:
    def __init__(self, use_tfidf=False, max_vocab_size=5000):
        self.use_tfidf = use_tfidf
        self.max_vocab_size = max_vocab_size
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_vocab_size, stop_words='english') if use_tfidf else None
        self.vocab = None
        self.vocab_size = 0
        
    def clean_text(self, text):
        """Lowercase, remove punctuation, remove extra spaces"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_handcrafted_features(self, article, question, option):
        """Extract lexical features for traditional ML"""
        features = []
        
        # Length features
        article_words = len(article.split())
        question_words = len(question.split())
        option_words = len(option.split())
        
        features.append(min(article_words / 500, 1.0))  # Normalized
        features.append(min(question_words / 50, 1.0))
        features.append(min(option_words / 20, 1.0))
        
        # Overlap features
        article_words_set = set(article.split())
        question_words_set = set(question.split())
        option_words_set = set(option.split())
        
        if question_words > 0:
            overlap = len(question_words_set & article_words_set) / question_words
            features.append(min(overlap, 1.0))
        else:
            features.append(0.0)
        
        if option_words > 0:
            overlap = len(option_words_set & article_words_set) / option_words
            features.append(min(overlap, 1.0))
        else:
            features.append(0.0)
        
        if question_words > 0:
            overlap = len(option_words_set & question_words_set) / question_words
            features.append(min(overlap, 1.0))
        else:
            features.append(0.0)
        
        # Position feature
        pos = article.find(option[:50]) if len(option) > 0 else -1
        features.append(pos / max(len(article), 1) if pos != -1 else 1.0)
        
        # Binary features
        features.append(1 if any(word in article for word in option.split()[:3]) else 0)
        features.append(1 if option_words <= 5 else 0)  # Short option
        
        return np.array(features, dtype=np.float32)
    
    def build_vocabulary(self, texts, max_samples=50000):
        """Build vocabulary from texts for One-Hot encoding"""
        print("Building vocabulary...")
        word_counts = {}
        
        # Sample texts to build vocabulary
        sample_size = min(len(texts), max_samples)
        indices = np.random.choice(len(texts), sample_size, replace=False)
        
        for idx in indices:
            words = texts[idx].split()[:200]  # Limit to first 200 words per text
            for word in words:
                if word not in STOPWORDS and len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_words[:self.max_vocab_size])}
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        return self.vocab
    
    def text_to_sparse_matrix(self, texts, max_samples=None):
        """Convert texts to sparse matrix using vocabulary"""
        if max_samples:
            texts = texts[:max_samples]
        
        # Use LIL format for efficient construction
        n_samples = len(texts)
        data = []
        row_ind = []
        col_ind = []
        
        for i, text in enumerate(texts):
            if i % 5000 == 0:
                print(f"  Processing text {i}/{n_samples}")
            
            words = text.split()[:200]  # Limit words per document
            word_counts = {}
            
            for word in words:
                if word in self.vocab:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Normalize by document length
            doc_len = len(words)
            for word, count in word_counts.items():
                if doc_len > 0:
                    tf = count / doc_len
                    data.append(tf)
                    row_ind.append(i)
                    col_ind.append(self.vocab[word])
        
        # Create sparse CSR matrix
        sparse_matrix = csr_matrix((data, (row_ind, col_ind)), 
                                    shape=(n_samples, self.vocab_size))
        
        return sparse_matrix
    
    def create_option_level_data(self, df, max_samples=None):
        """
        Convert dataframe to option-level training data.
        Returns sparse matrix to save memory.
        """
        print("Creating option-level data...")
        
        X_texts = []
        X_handcrafted = []
        y = []
        
        if max_samples:
            df = df.head(max_samples)
        
        # First pass: collect all combined texts
        all_combined = []
        handcrafted_list = []
        
        for idx, row in df.iterrows():
            article = self.clean_text(row['article'])
            question = self.clean_text(row['question'])
            correct_answer = row['answer']
            
            # Handle different column naming conventions
            options = {}
            if 'A' in row:
                options = {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']}
            elif 'option_A' in row:
                options = {'A': row['option_A'], 'B': row['option_B'], 'C': row['option_C'], 'D': row['option_D']}
            else:
                # Fallback: use answer as only option
                options = {'A': correct_answer}
            
            for label, option_text in options.items():
                if not isinstance(option_text, str) or not option_text:
                    option_clean = ""
                else:
                    option_clean = self.clean_text(option_text)
                
                # Combine text
                combined_text = f"{article} {article} {question} {option_clean}"
                all_combined.append(combined_text)
                
                # Handcrafted features
                handcrafted = self.extract_handcrafted_features(article, question, option_clean)
                handcrafted_list.append(handcrafted)
                
                # Label
                y.append(1 if label == correct_answer else 0)
        
        print(f"Total examples created: {len(all_combined)}")
        
        # Build vocabulary if using One-Hot encoding
        if not self.use_tfidf and self.vocab is None:
            self.build_vocabulary(all_combined, max_samples=30000)
        
        # Convert texts to sparse features
        print("Converting texts to feature matrix...")
        if self.use_tfidf and self.tfidf_vectorizer:
            X_text_features = self.tfidf_vectorizer.fit_transform(all_combined)
        else:
            X_text_features = self.text_to_sparse_matrix(all_combined)
        
        # Convert handcrafted features to numpy array
        X_handcrafted = np.array(handcrafted_list, dtype=np.float32)
        
        print(f"Text features shape: {X_text_features.shape}")
        print(f"Handcrafted features shape: {X_handcrafted.shape}")
        
        # Combine sparse and dense features
        X_combined = hstack([X_text_features, csr_matrix(X_handcrafted)])
        
        y = np.array(y)
        
        # Calculate class weights
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) > 1:
            class_weights = compute_class_weight('balanced', classes=unique, y=y)
            class_weight_dict = {unique[i]: class_weights[i] for i in range(len(unique))}
        else:
            class_weight_dict = {0: 1.0, 1: 1.0}
        
        print(f"Final matrix shape: {X_combined.shape}")
        print(f"Memory usage: {X_combined.data.nbytes / 1024**2:.2f} MB")
        
        return X_combined, y, class_weight_dict
    
    def save(self, path):
        """Save preprocessor"""
        joblib.dump({
            'use_tfidf': self.use_tfidf,
            'max_vocab_size': self.max_vocab_size,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size
        }, path)
    
    def load(self, path):
        """Load preprocessor"""
        data = joblib.load(path)
        self.use_tfidf = data['use_tfidf']
        self.max_vocab_size = data['max_vocab_size']
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']


def create_cosine_similarity_features(df, vectorizer):
    """Create cosine similarity features between article, question, and options"""
    features = []
    
    for idx, row in df.iterrows():
        article = row['article']
        question = row['question']
        options = []
        
        if 'A' in row:
            options = [row['A'], row['B'], row['C'], row['D']]
        elif 'option_A' in row:
            options = [row['option_A'], row['option_B'], row['option_C'], row['option_D']]
        
        correct = row['answer']
        correct_idx = ord(correct) - ord('A') if correct in ['A','B','C','D'] else 0
        
        # Vectorize
        article_vec = vectorizer.transform([article])
        question_vec = vectorizer.transform([question])
        option_vecs = vectorizer.transform(options)
        
        # Similarities
        q_article_sim = cosine_similarity(question_vec, article_vec)[0][0]
        
        for i, opt_vec in enumerate(option_vecs):
            opt_article_sim = cosine_similarity(opt_vec, article_vec)[0][0]
            opt_question_sim = cosine_similarity(opt_vec, question_vec)[0][0]
            
            features.append({
                'q_article_sim': q_article_sim,
                'opt_article_sim': opt_article_sim,
                'opt_question_sim': opt_question_sim,
                'is_correct': 1 if i == correct_idx else 0
            })
    
    return pd.DataFrame(features)