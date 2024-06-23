from langchain_huggingface import HuggingFaceEmbeddings

# create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

def get_embedding(model_name: str):
    
    """ Get a hugging face embedding based on the model name specified.

    Args:
        model_name (str): a hugging face pre-trained embedding model like 'sentence-transformers/all-MiniLM-l6-v2'

    Returns:
        model: embedding model to embed pdf data
    """
    
    # initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,     
        model_kwargs = model_kwargs, 
        encode_kwargs = encode_kwargs 
    )
    
    return embeddings