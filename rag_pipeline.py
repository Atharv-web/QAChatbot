import json
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from typing_extensions import TypedDict
from typing import List, Annotated
import operator

# Load environment variables
load_dotenv()

# Initialize tools
websearch_tool = TavilySearchResults(k=3)
model = ChatOllama(model='llama3.2:3b-instruct-fp16', base_url='http://localhost:11434')
json_model = ChatOllama(model='llama3.2:3b-instruct-fp16', temperature=0, format='json')
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Formatting utility
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# Load and split documents
def load_and_split_docs(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return text_splitter.split_documents(docs_list)

# Embedding and vectorstore creation
def create_vectorstore(chunks):
    index = faiss.IndexFlatL2(len(embeddings.embed_query(chunks[0].page_content)))
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vectorstore.add_documents(chunks)
    vectorstore.save_local('VectorDB')

# Load vectorstore
def load_vectorstore():
    return FAISS.load_local('VectorDB', embeddings=embeddings, allow_dangerous_deserialization=True)

saved_vector_store = load_vectorstore()
retriever = saved_vector_store.as_retriever(k=3)

# Instructions and prompts
router_instruction = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to prompt engineering and finance.
Use the vectorstore for these topics. For all else, use web-search.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore'."""

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
Grade as 'yes' if the document has semantic meaning related to the question."""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.
Return JSON with key 'binary_score' as 'yes' or 'no' indicating relevance."""

rag_prompt = """You are an assistant for question-answering tasks.
Here is the context to use to answer the question:

{context}

Now, review the user question:

{question}

Provide an answer using five sentences maximum. Keep it concise.
Answer:"""

hallucination_grader_instructions = """You are a teacher grading a quiz.
Ensure the STUDENT ANSWER is grounded in the FACTS and does not contain hallucinations.
Return JSON with 'binary_score' as 'yes' or 'no' and an explanation."""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.
Return JSON with 'binary_score' and 'explanation'."""

# Graph state definition
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    max_tries: int
    answers: int
    loop_steps: Annotated[int, operator.add]
    documents: List[Document]

# Functions for pipeline
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    docs_text = format_docs(state["documents"])
    rag_prompt_formatted = rag_prompt.format(context=docs_text, question=state["question"])
    generation = model.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": state.get("loop_step", 0) + 1}

def grade_documents(state):
    question = state["question"]
    filtered_documents = []
    web_search = "No"

    for d in state["documents"]:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = json_model.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']

        if grade.lower() == 'yes':
            filtered_documents.append(d)
        else:
            web_search = "Yes"

    return {"documents": filtered_documents, "web_search": web_search}

def web_search(state):
    question = state["question"]
    documents = state.get("documents", [])
    web_results = websearch_tool.invoke({"query": question})
    documents.append(Document(page_content="\n".join(d["content"] for d in web_results)))
    return {"documents": documents}

def route_question(state):
    route_decision = json_model.invoke([SystemMessage(content=router_instruction)] + [HumanMessage(content=state["question"])]).content
    source = json.loads(route_decision)['datasource']
    return "websearch" if source == "websearch" else "vectorstore"

def decide_generation(state):
    return "websearch" if state["web_search"] == "Yes" else "generate"

def grade_generation(state):
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(state["documents"]),
        generation=state["generation"].content
    )
    result = json_model.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    return json.loads(result.content)
