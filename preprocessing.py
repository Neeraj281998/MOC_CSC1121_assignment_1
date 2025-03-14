import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('punkt', quiet=True)      # Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


stemmer = PorterStemmer()            # Initialize stemmers
lemmatizer = WordNetLemmatizer()     # Initialize lemmatizers


STOP_WORDS = set(stopwords.words('english'))                 # Create stopwords list with domain-specific terms
TECHNICAL_STOPWORDS = {
    'fig', 'figure', 'eq', 'equation', 'ref', 'reference', 'et', 'al',
    'paper', 'result', 'results', 'method', 'methods', 'using', 'use',
    'show', 'shown', 'found', 'given', 'consider', 'considered', 'since'
}
STOP_WORDS.update(TECHNICAL_STOPWORDS)

def preprocess_text(text, use_stemming=True, use_lemmatization=False, remove_numbers=True):  # Text preprocessing  - text, use_stemming, use_lemmatization, remove_numbers
    
    if not text:
        return []
        
    
    text = text.lower() # Convert to lowercase
    
    
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)  # Remove numbers if exist
    
    # Handling special characters

    text = re.sub(r'[/\\]', ' ', text)  # Replacing slashes with spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # Replacing punctuation with spaces
    
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)  # Split hyphenated words
    
    
    tokens = word_tokenize(text)   # Tokenize
    
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]   # Remove stopwords and short tokens
    

    if use_stemming:                                        # Apply stemming 
        tokens = [stemmer.stem(word) for word in tokens]
    elif use_lemmatization:                                 # Apply lemmatization 
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens  # Returning List of processed tokens

def preprocess_query(query, use_stemming=True, use_lemmatization=False):
   
    # Expand common abbreviations and technical terms
    query_expansions = {
        "aero": "aerodynamic",
        "eq": "equation",
        "eqs": "equations",
        "temp": "temperature",
        "eval": "evaluation",
        "calc": "calculation",
        "exp": "experimental",
        "2d": "two dimensional",
        "3d": "three dimensional"
    }
    
    expanded_query = query.lower()
    for abbr, expansion in query_expansions.items():
        expanded_query = re.sub(r'\b' + abbr + r'\b', abbr + " " + expansion, expanded_query)
    
    
    return preprocess_text(expanded_query, use_stemming, use_lemmatization, remove_numbers=False) # Process with standard pipeline

