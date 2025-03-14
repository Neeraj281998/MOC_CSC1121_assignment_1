from collections import Counter, defaultdict
from math import log, exp
import numpy as np

# BM25 Parameters 

K1 = 1.2                    # Term frequency saturation parameter
B = 0.75                    # Length normalization parameter
EPSILON = 0.25              # Smoothing parameter for document length

# Language Model Parameters

LAMBDA_PARAM = 0.2          # Jelinek-Mercer smoothing parameter
ALPHA = 0.7                 # Dirichlet smoothing parameter
MU = 2000                   # Dirichlet prior parameter

# Weights

TF_IDF_WEIGHT = 0.2
BM25_WEIGHT = 0.5
LMJM_WEIGHT = 0.3

def compute_tf_idf(documents, inverted_index, doc_lengths, term_doc_freq):         # VSM with  TF-IDF scores 
    
    N = len(documents)
    tf_idf = defaultdict(float)

    for term, doc_positions in inverted_index.items():                              # Calculating IDF with smoothing to avoid division by zero
        
        df = len(doc_positions)
        idf = log((N + 1) / (df + 0.5))  # Smoothed IDF
        
        for doc_id, positions in doc_positions.items():
            
            tf = 1.0 + log(len(positions))                                           # Calculating log normalization TF 
            
           
            doc_len_factor = 1.0 / log(1 + doc_lengths.get(doc_id, 1))              # Apply document length normalization
            
            # Calculate and store score
            tf_idf[(doc_id, term)] = tf * idf * doc_len_factor                       # Calculate and store score
    
    return dict(tf_idf)

def compute_bm25(documents, inverted_index, doc_lengths, term_doc_freq): #BM25
   
    N = len(documents)
    avg_doc_len = sum(doc_lengths.values()) / max(N, 1)
    bm25_scores = defaultdict(float)

    for term, doc_positions in inverted_index.items():
        
        df = len(doc_positions)
        idf = log((N - df + 0.5) / (df + 0.5) + 1.0)              # Calculating IDF with BM25 formula
        
        for doc_id, positions in doc_positions.items():
           
            tf = len(positions)                                      # Term frequency 
            
            
            doc_len = doc_lengths.get(doc_id, 0)                    # Document length normalization 
            length_norm = 1.0 - B + B * (doc_len / avg_doc_len) if avg_doc_len > 0 else 1.0
            
            
            proximity_boost = 1.0
            if len(positions) > 1:                                  # Proximity boost if term appears multiple times close together
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps and min(gaps) < 5:  # Terms appear within 5 words
                    proximity_boost = 1.2
            
            # BM25 formula with term saturation

            numerator = tf * (K1 + 1)
            denominator = tf + K1 * length_norm
            
            # Calculate final score

            score = idf * (numerator / max(denominator, EPSILON)) * proximity_boost
            bm25_scores[(doc_id, term)] = score
    
    return dict(bm25_scores)

def compute_lmjm(documents, inverted_index, doc_lengths, term_doc_freq):      # Language Model scores with Jelinek-Mercer smoothing
    
    
    
    total_tokens = sum(doc_lengths.values())            # Calculating collection statistics
    collection_prob = {}
    
    
    for term, doc_positions in inverted_index.items():  # Calculating collection probability for each term
        term_count = sum(len(positions) for positions in doc_positions.values())
        collection_prob[term] = term_count / max(total_tokens, 1)
    
    
    lm_scores = defaultdict(float)      # Calculating language model scores
    
    for term, doc_positions in inverted_index.items():
        coll_prob = collection_prob.get(term, 0.0)
        
        for doc_id, positions in doc_positions.items():
            doc_len = doc_lengths.get(doc_id, 0)
            if doc_len == 0:
                continue
                
            
            tf = len(positions)
            doc_prob = tf / doc_len     # Calculating document probability
            
           
            smoothed_prob = (1 - LAMBDA_PARAM) * doc_prob + LAMBDA_PARAM * coll_prob         # Jelinek-Mercer smoothing
            
            
            lm_scores[(doc_id, term)] = log(max(smoothed_prob, 1e-10))          # Storing log probability to prevent underflow and allow addition
    
    return dict(lm_scores)
