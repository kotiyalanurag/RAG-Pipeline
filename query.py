import argparse

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from database import get_vector_db

# Use a German prompt template to get answers in German

PROMPT_TEMPLATE = """
Beantworten Sie die Frage nur basierend auf dem folgenden Kontext:

{context}

---

Beantworten Sie die Frage anhand des obigen Kontexts: {question}
"""

# # Use an English prompt template to get answers in English

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

def get_answer(question: str):
    
    """ A function that retrives similar documents from our faiss database and pass an enhanced query
    through Ollama minstral model to receive a coherent and concide answer to the base query.

    Args:
        question (str): a question asked by the user.

    Returns:
        str: formatted text response from the minstral model that contains source material for the answer.
    """
    
    # load or create a new faiss vector database
    db = get_vector_db()

    # initialise Ollama llama3 model -> need to run ollama serve from terminal before using Ollama
    model = Ollama(model = "llama3")
    
    # retrieve top 3 relevant document chunks from the database based on user's query 
    results = db.similarity_search_with_score(question, k = 3)
    context = " ".join([doc.page_content for doc, _ in results])
    
    # create a prompt template for Ollama
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # create a proper prompt using langchains LLM prompt
    prompt = prompt_template.format(context = context, question = question)
    
    # get sources of the relevant docs in this case the unique id we created for the chunks
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # get a response to the enhanced query from Ollama
    response_text = model.invoke(prompt)
    
    # format response so that it contains model's answer as well the source documents it used to generate that answer
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    print(formatted_response)
    
    return response_text

def main():
    
    """ A function that takes in the user's query as an argument and displays Ollama's response on the terminal. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', type = str, default = 'Welche Arten von Fonds werden angeboten?', help = 'Enter a question')
    args = parser.parse_args()
    
    # get the query from the relevant args
    query = args.question
    
    # get Ollama's response to the query (RAG)
    get_answer(question = query)
    
    
if __name__ == "__main__":
    main()