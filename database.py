from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import get_embedding

DATA_DIR = "path/to/directory"

def load_docs(data_dir):
    
    """ A simple function that loads data from all pdfs inside a directory.

    Args:
        DATA_DIR (path): Path to the directory where pdf documents are stored.

    Returns:
        list(tuple): returns a list of tuples where each tuple contains "page content" of the document and
        some meta deta like "source" and "page number" i.e., [Document(page_content = "Tax information ...",
        metadata = {"source": "doc_1.pdf", "page": 10})].
    """
    
    # create an instance of document loader using the pypdfdirectoryloader from langchain
    document_loader = PyPDFDirectoryLoader(data_dir)
    
    return document_loader.load()

def split_docs_into_chunks(documents, chunk_size = 500, chunk_overlap = 50):
    
    """ A function that split a document into chunks of specific sizes like 500 characters.

    Args:
        documents (list(tuple)): a list of tuples where each tuple contains "page content" of the document and
        some meta deta like "source" and "page number".

    Returns:
        list(tuples): similar to the format in which data came but any document larger than chunk_size has been split
        into multiple chunks.
    """
    
    # create an instance of recursive character text splitter from langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False,
    )
    
    return text_splitter.split_documents(documents)    

def get_chunk_ids(chunks):

    """ A function that creates a unique identifier for each chunk in the database like "docs/doc_name:page_num:chunk_id".
    A chunk id of "docs/doc_1.pdf:10:5" refers to the 5th chunk on page 10 of doc_1.pdf. 

    Returns:
        list(tuple): returns a list of tuples where each tuple contains "page content" of the document and
        some meta deta like "source", "page number", and "chunk_id"
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # get source and page number from metadata to create a current page id
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def get_vector_db(load_saved = False):
    
    """ A function that creates a new vector database or loads an existing database based on db-name.

    Args:
        load_saved (bool, optional): Indicates whether to create a new database or load an existing one. Defaults to False.

    Returns:
        vector_db: a FAISS vector database
    """
    
    # load an existing vector database - replace db-name with a valid name
    if load_saved:
        # use model_name = "sentence-transformers/all-MiniLM-l6-v2" for English language embeddings in 384 dimensions
        db = FAISS.load_local("db_name", get_embedding(model_name = "danielheinz/e5-base-sts-en-de")) # 768 dimensions
        
        return db
    
    # load documents from .pdf files
    documents = load_docs(data_dir = DATA_DIR)
    # create document chunks for better retrieval
    chunks = split_docs_into_chunks(documents = documents)
    # create unique ids for all chunks
    chunks_with_ids = get_chunk_ids(chunks = chunks)
    
    # store data in a vector database like FAISS
    db = FAISS.from_documents(chunks_with_ids, get_embedding(model_name = "sentence-transformers/all-MiniLM-l6-v2"))
    # db.save_local("db_name") # uncomment only if you want to save your vector database locally
    
    return db

if __name__ == "__main__":
    get_vector_db()