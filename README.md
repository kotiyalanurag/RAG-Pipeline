<h1 align=center> LLAMA3 RAG Pipeline

![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/faiss-1.8.0-blue) ![](https://img.shields.io/badge/langchain-0.2.5-blue) ![](https://img.shields.io/badge/PyMuPDF-1.24.5-blue) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

<p align = left>A simple application that uses faiss vector databases to store chunk embeddings from pdf documents and retrieves top-3 closest embeddings to a user query. The retrieved queries are combined into a single context and sent to Llama 3 along with the original user query to generate a coherent response based on the context.</p>

<p align="center">
  <img src = assets/rag-intuition.png max-width = 100% height = '400' />
</p>

## Project Structure

```html
|- docs
  |-- doc_1.pdf
  |-- doc_2.pdf
|- notebook
  |-- rag_de.ipynb
  |-- rag_en.ipynb
|- database.py
|- embeddings.py
|- query.py
```

## How to get started?

Open terminal and run the following commands one by one

```html
mkdir Project
```
Now switch to the newly created Project directory using

```html
cd Project
```
Now clone the github repository inside your working directory using

```html
git clone https://github.com/kotiyalanurag/RAG-Pipeline.git
```
Create a python virtual environment using

```html
python -m venv env
```

Activate the environment using

```html
source env/bin/activate
```

Now, load all the dependencies for the project using 

```html
pip install -r requirements.txt
```
## How to install Ollama?

Ollama helps us to run LLMs locally. Apart from the package dependencies you need to have Ollama installed on your system using this [link](https://github.com/ollama/ollama).

Once, you're done with the installations run the following on your terminal. Replace model with model of your choice like llama2, llama3, minstral etc.

```html
ollama run model
```

Running this for the first time will take a couple of minutes since it downloads the model on your local system. If you already have the model then you can start an Ollama server using

```html
ollama serve
```

Now we're ready to start our application.

## Hyperparameters

For the faiss vector database, chunk size and chunk overlap make a significant difference when retrieving relevant documents based on a query.

```python
# tried with 250, 500, and 750 
chunk_size = 500

# usually set it to 10% of chunk size
chunk_overlap = 50
```

The embedding models play an important role as generated vector quality completely depends upon the embedding model used.

```python
# for german language embeddings in 768 dimensions
model_name = "danielheinz/e5-base-sts-en-de" 

# for english language embeddings in 384 dimensions
model_name = "sentence-transformers/all-MiniLM-l6-v2"
```

## How to run the application from CLI?

The model is capable of generating responses in both English and German. However, certain changes need to be made. If you want to generate responses in German then you only need to set the data path on line 6 inside database.py. 

For English response, make sure the data path inside database.py line 6 is set to the directory where your pdf files are stored. Then make sure you're using the correct embedding model inside database.py lines 97 and 109. Lastly, check that you're using the correct template based on language inside query.py lines 8-30. Also the pdf files should have data in English for this version to work. 

Now, open terminal and type the following, where you replace "insert-your-query-here" with your query in German or English.

```python
python query.py -q "insert-your-query-here"
```
## Context + Query Template for Llama3

Here is what the combined query plus context template fed to Llama3 looks like. Note that the context is retrieved from the vector database based on a similarity search with the query.

```html
Human: 
Beantworten Sie die Frage nur basierend auf dem folgenden Kontext:

Ihr Mindestanteil an jedem einzelnen von Ihnen ausgewählten Fonds beträgt 1 Prozent.  
5.5 Welche Arten von Fonds bieten wir an?  
Die von Ihnen gewählten Fonds ordnen wir dem Fondsvermögen in Ihrem Vertrag zu. Sie können aus drei Fondsarten auswählen, St andard Life Fonds, Managed Portfolios sowie Publikumsfonds verschiedener 
Fondsgesellschaften. 5 Ihr Fondsvermögen .............................................................................................................................4
5.1 Was ist Ihr Fondsvermögen?.................................................................................................................4
5.2 Wie berechnen wir Ihr Fondsvermögen?...............................................................................................4 5.3 Was gilt unter außergewöhnlichen Umständen für die Berechnung von Vermögenswerten? ..............4
5.4 Wie investieren wir Ihren Einmalbeitrag? ..............................................................................................4
5.5 Welche Arten von Fonds bieten wir an?................................................................................................5

---

Beantworten Sie die Frage anhand des obigen Kontexts: Welche Arten von Fonds werden angeboten?
```

## Some Responses using RAG

Below you can see some examples queries that were sent to the model and the response that was generated by the model based on RAG from faiss database. In addition to a response the application also cites the sources from the pdf using unique chunk 'id' that it used for answering the query. To see the complete results check out rag_de or rag_en inside notebook directory.

1. Query - "Welche Arten von Fonds werden angeboten?"

```html
Response: Nach dem gegebenen Kontext bieten wir drei Arten von Fonds an:

    1. Standard Life Fonds
    2. Managed Portfolios
    3. Publikumsfonds verschiedener Fondsgesellschaften
```
2. Query - "Wie sieht es mit den Abzügen von der Kirchensteuer aus?"

```html
Response: Nach dem vorliegenden Text erfolgt die Abgeltung von Kirchensteuern auf abgeltend besteuerte Kapitalerträge automatisch seit dem 1. Januar 2015. Dies bedeutet, dass die Kirchensteuer direkt vom Versicherer abgeführt wird und erst im Rahmen der Einkommensteuerveranlagung des Steuerpflichtigen anrechnet wird.
```

3. Query - "An wen kann ich mich wenden, wenn ich Fragen habe?"

```html
Response: Sie sollten sich als erstes an Ihren Vermittler wenden. Unsere Servicemitarbeiter sind von Montag bis Freitag von 9:00 bis 17:00 Uhr für Sie da:

   * Tel.: 0800 2214747 (kostenfrei)
   * Fax: 0800 5892821
   * E-Mail: kundenservice@standardlife.de

```

4. Query - "Wie bestimmen Sie die Höhe der Auszahlung?"

```html
Response: Die Höhe der Auszahlung wird durch das Fondsvermögen in Ihrem Vertrag zum maßgeblichen Stichtag bestimmt. Dieses ergibt sich aus der Summe aller Anteilseinheiten der Fonds in Ihrem Vertrag multipliziert mit dem jeweiligen Anteilspreis des Fonds zum Stichtag.

```

5. Query - Response: Nach dem obigen Text gilt Folgendes:

```html
Response: Nach dem obigen Text gilt Folgendes: Die Allgemeine Versicherungsbedingungen sind in der Dokumentation "BASIS_PACK_WBWB/D/1006/XIII/03/22" zu finden, insbesondere im Abschnitt "1. Allgemeine Versicherungsbedingungen .....................................................................................................1".
```
To check generated response in both German and English check out queries.txt.

## Limitations

We can see that the model generates good results. However, the response can be vague in some cases like query 5 where it pointed out to check the source document itself. This can be due to multiple reasons like 
- use of current chunk size which can be played around with some more
- lack of data cleaning that leads to a lot of special characters in some chunks 
- use of small open-source embedding models to create vector embeddings for chunks
- use of Llama3 to generate response which isn't actually trained on a lot of Deutsch text data hence the results are impressive

## How to improve this application?

There are multiple ways to improve this application. Some of them are listed below:

- use bigger embedding models like text-embedding-3-large from OpenAI (which is paid)
- make sure that the embedding model is good at generating embeddings for Deutsch (most models only give good results for English)
- play around with the number of relevant documents retrieved for each user query (current default in application is 3)
- use an LLM that is good at generating response in Deutsch and considerably bigger than Llama3 (8B)
