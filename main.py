import os
from indexing import load_documents, load_queries, build_inverted_index
from ranking import compute_tf_idf, compute_bm25, compute_lmjm
from Scoring_term import score_aggregation
from save_evaluation import save_results
from preprocessing import preprocess_query


if not os.path.exists("output"):                                                              # Create output directory
    os.makedirs("output")


documents = load_documents("cran.all.1400.xml")                                               # Load datasets
queries = load_queries("cran.qry.xml")
qrels = "cranqrel.trec.txt"


inverted_index, doc_lengths, term_doc_freq = build_inverted_index(documents)                  # Creating Inverted index with document frequency and position information


# Compute ranking models ( VSM with TF-IDF , BM25 , Language Model with Jelinek-Mercer )

tf_idf_scores = compute_tf_idf(documents, inverted_index, doc_lengths, term_doc_freq)
bm25_scores = compute_bm25(documents, inverted_index, doc_lengths, term_doc_freq)
lmjm_scores = compute_lmjm(documents, inverted_index, doc_lengths, term_doc_freq)

# Save results in TXT file 
save_results(queries, tf_idf_scores, "output/tfidf_results.txt")
save_results(queries, bm25_scores, "output/bm25_results.txt")
save_results(queries, lmjm_scores, "output/lmjm_results.txt")


#Query
query = "properties of aerodynamic models"
processed_query = preprocess_query(query)
print("\nQuery:", query)

# Printing TOP 5 result from each Model
print("TF-IDF :", score_aggregation(processed_query, tf_idf_scores)[:5])
print("BM25 :", score_aggregation(processed_query, bm25_scores)[:5])
print("LMJM :", score_aggregation(processed_query, lmjm_scores)[:5])
