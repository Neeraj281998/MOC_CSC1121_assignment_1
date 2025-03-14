import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import os
from preprocessing import preprocess_text

def load_documents(file_path):
     
    
    try:                                           
        tree = ET.parse(file_path)                          # Document loading
        root = tree.getroot()
    except ET.ParseError as e:                              #Error Handling 
        print(f"Error parsing XML: {e}")
        return {}
    except FileNotFoundError:                               #File not found Error
        print(f"File not found: {file_path}")
        return {}

    documents = {}

    for doc in root.findall('doc'):
        doc_id = doc.find('docno').text.strip()
        
        
        title_elem = doc.find('title')                                                               # Extract Title fields
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""        
        
        author_elem = doc.find('author')                                                             # Extract Author fields
        author = author_elem.text.strip() if author_elem is not None and author_elem.text else ""
        
        text_elem = doc.find('text')                                                                 # Extract Text fields
        text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        
       
        combined_text = f"{title} {title} {title} {author} {text}"               # Combine content and weighting title more heavily by duplicating three times
        
        
        documents[doc_id] = preprocess_text(combined_text)                       # Preprocess the combined text

    return documents

def load_queries(file_path):
    
    
    try:
        tree = ET.parse(file_path)                       # Query Document loading
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")                 #Error Handling 
        return {}
    except FileNotFoundError:
        print(f"File not found: {file_path}")            #Query File not found Error
        return {}

    queries = {}

    for top in root.findall('top'):
        num = top.find('num').text.strip()
        title = top.find('title').text.strip() if top.find('title') is not None else ""
        
        # Preprocessed query
        queries[num] = preprocess_text(title)

    return queries

def build_inverted_index(documents):          # Generating Inverted index and Returns: inverted_index, doc_lengths, term_doc_freq
    
     
    inverted_index = defaultdict(lambda: defaultdict(list))
    doc_lengths = {}
    term_freq = defaultdict(Counter)  # Frequency in each document
    
  
    for doc_id, tokens in documents.items():     # Generating Inverted index
        doc_lengths[doc_id] = len(tokens)
        
        
        for position, token in enumerate(tokens):  # Track positions for each term
            inverted_index[token][doc_id].append(position)
            term_freq[doc_id][token] += 1
    
   
    term_doc_freq = {term: len(doc_positions) for term, doc_positions in inverted_index.items()}   # Calculate Term frequency 
    
    return dict(inverted_index), doc_lengths, term_doc_freq
