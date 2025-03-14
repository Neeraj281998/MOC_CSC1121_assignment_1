from collections import defaultdict
import heapq
from preprocessing import preprocess_query

def score_aggregation(query_tokens, scoring_model, max_results=20):  #Score aggregation

        # query_tokens: Preprocessed query tokens
        # scoring_model: Dictionary mapping (doc_id, term) pairs to scores
        # max_results: Maximum number of results to return
    
    query_terms = set(query_tokens)  # Get unique query terms
    
   
    doc_scores = defaultdict(float)  # Calculating document scores
    
    
    doc_term_matches = defaultdict(set) # Track which terms matched in each document for term coverage scoring
    
    
    for term in query_terms:   # Aggregate scores for document
        for (doc_id, t), score in scoring_model.items():
            if term == t:
                doc_scores[doc_id] += score
                doc_term_matches[doc_id].add(term)
    
    
    for doc_id, matched_terms in doc_term_matches.items(): # Boost documents that match more query terms with term coverage 
        coverage_ratio = len(matched_terms) / max(len(query_terms), 1)
        doc_scores[doc_id] *= (1.0 + coverage_ratio) 
    
 
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_results] #List of (doc_id, score) pairs sorted by score
