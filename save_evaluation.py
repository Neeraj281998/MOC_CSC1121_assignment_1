import os
import re
from collections import defaultdict
import numpy as np
from Scoring_term import score_aggregation
from preprocessing import preprocess_query

def save_results(queries, scoring_model, filename):
    """
    Save ranked results in TREC format with enhanced error handling
    """
    # Ensure the directory exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filename, "w") as f:
        for q_id, query_tokens in queries.items():
            # Use the preprocessed query tokens directly
            ranked_results = score_aggregation(query_tokens, scoring_model, max_results=100)
            
            if ranked_results:
                for rank, (doc_id, score) in enumerate(ranked_results):
                    f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.6f} B25-improved\n")
            else:
                print(f"No results for query {q_id}: {' '.join(query_tokens)}")

