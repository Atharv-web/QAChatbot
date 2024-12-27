import json
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

websearch_tool = TavilySearchResults(k =3)

from langchain_ollama import ChatOllama,OllamaEmbeddings
model = ChatOllama(model='llama3.2:3b-instruct-fp16',base_url='http://localhost:11434')
json_model = ChatOllama(model = 'llama3.2:3b-instruct-fp16',temperature =0,format ='json')
embeddings = OllamaEmbeddings(model ='nomic-embed-text',base_url= 'http://localhost:11434')

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
import faiss
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# prompt_template = """
#         You are my chatbot assistant. You are going to have a conversation with me.
#         When asked a question provide a correct and clear answer.
#         If you dont know the answer, say you dont know the answer. 
#         Question : {question}
#         Answer:
#     """
# prompt = ChatPromptTemplate.from_template(prompt_template)

# Post processing 
def format_docs(docs):
    f_docs = []
    for doc in docs:
        f_docs.append(doc.page_content)
    return "\n".join(f_docs)

urls = ['https://en.wikipedia.org/wiki/Prompt_engineering']

def load_docs(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

# docs = load_docs(urls)
# using recursive text splitter
def split_docs(docs_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs_list)
    return chunks

# chunks = split_docs(docs)
from langchain_community.docstore.in_memory import InMemoryDocstore

def embedder(chunks):
    embedded_chunks = []
    for chunk in chunks:
        emb = embeddings.embed_query(chunk.page_content)
        embedded_chunks.append(emb)
    # return embedded_chunks[0]
    index = faiss.IndexFlatL2(len(embedded_chunks[0]))

    vectorstore = FAISS(
        embedding_function=embeddings,
        index = index,
        docstore = InMemoryDocstore(),
        index_to_docstore_id = {}
    )
    ids = vectorstore.add_documents(chunks)
    db_name= 'VectorDB'
    vectorstore.save_local(db_name)

# embedder(chunks)
database_name = 'VectorDB'
saved_vector_store = FAISS.load_local(database_name,embeddings = embeddings,allow_dangerous_deserialization = True)
retriever = saved_vector_store.as_retriever(k=3)
# result = retriever.invoke("Summarize Prompt Engineering in 50 words")
# result_output = result[1].page_content
# pribnt(result_output)

# -------------------------------------------------------------------------------------------------------------------------------------------
# ROUTER
from langchain_core.messages import HumanMessage,SystemMessage
router_instruction = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to prompt engineering.

Use the vectorstore for questions on these topics. For all else, and especially for current events use web-search.

return json with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# question = "what is prompt engineering?"
# message = [HumanMessage(content = question)]
# test_vectorstore=json_model.invoke([SystemMessage(content=router_instruction)] + message)
# print(json.loads(test_vectorstore.content))


# ------------------------------------------------------------------------------------------------------------------------------------------
#GRADING THE RETRIEVAL
doc_grader_instructions = """ You are a grader assessing relevance of a retrieved document to a user question.

If the document has semantic meaning related to the question, grade it as relevant."""


doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

Carefully and objectively assess whether the document contains atleast some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains atleast some information that is relevant to the question."""
 
question1 = "what is chain of thought prompting?"
# docs = retriever.invoke(question1)
# doc_text = docs[1].page_content
# print(doc_text)
# doc_grader_prompt_formatted = doc_grader_prompt.format(document = doc_text, question = question1)
# result = json_model.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
# print(json.loads(result.content))

# -------------------------------------------------------------------------------------------------------------------------------------------------
# GENERATE 
rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer question:

{context}

Think carefully about the above context.

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.

Use five sentences maximum and keep the answer concise.

Answer:"""

# test it
# docs = retriever.invoke(question1)
# doc_text = format_docs(docs)
# rag_prompt_formatted = rag_prompt.format(context = doc_text, question = question1)
# message = [HumanMessage(content = rag_prompt_formatted)]
# generation = model.invoke(message)
# print(generation.content)

# ----------------------------------------------------------------------------------------------------------------------------------------
# WEB SEARCHING 
# using tavily
# generation = websearch_tool.invoke("LLM agents")
# print(generation[0]['content'])
# ----------------------------------------------------------------------------------------------------------------------------------------

# ANSWER GRADING
answer_grader_instructions = """You are teacher grading a quiz.
You will be given a QUESTION and a STUDENT ANSWER.
Here is the grade criteria to follow:
(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
"""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# ----------------------------------------------------------------------------------------------------------------------------------------
# HALLUCINATION GRADING

hallucination_grader_instructions = """ You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure that the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure that the STUDENT ANSWER does not contain "hallucinated" information ouotside the scope of FACTS.

Score:

A score of yes means that the student's answer meets all the criteria. This is best score.

A score of no means tha the student's answer does not meet all the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """ FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.
Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key explanation, that contains an explanation of the score.
"""

# hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents= doc_text, generation=generation.content)
# result = json_model.invoke([SystemMessage(content = hallucination_grader_instructions)]+[HumanMessage(content=hallucination_grader_prompt_formatted)])
# print(json.loads(result.content))


import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node. 
    """

    question : str
    generation : str
    web_search : str
    max_tries : int
    answers : int
    loop_steps : Annotated[int, operator.add]
    documents : List[str] 

from langchain.schema import Document
# from langchain.graphs import END

# CREATE THE NODES FOR EACH

def retrieve(state):
    """
    Retrieve documents from vectorstore
    Args: 
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    """
    Generate answers using RAG on retrieved documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step",0)

    docs_text = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context = docs_text, question  = question)
    generation = model.invoke([HumanMessage(content = rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to this question
    If any document is not relevant, we will set a flag to run web search
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document = d.page_content, question = question)
        result = json_model.invoke([SystemMessage(content= doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']

        if grade.lower() == 'yes':
            print("Doc is relevant")
            filtered_documents.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_documents, "web_search": web_search}


def web_search(state):
    """Web search based based on the question
    
    Args:
        state(dict): The current graph state
        
    Returns:
        state(dict): Append web results to documents
    """

    question = state["question"]
    documents = state.get("documents",[])

    docs = websearch_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents":documents}

def route_question(state):
    """
    Route question to websearch or RAG
    Args:
        state(dict): The current graph state
    Returns:
        str: Next node to call
    """

    route_question = json_model.invoke([SystemMessage(content=router_instruction)]+[HumanMessage(content= state["question"])])
    source = json.loads(route_question.content)['datasource']
    if source == "websearch":
        print("---ROUTING TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTING QUESTION TO RAG---")
        return "vectorstore"
    
def decide_generation(state):
    """
    Determines whether to generate an answer or add web search
    Args:
        state(dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    question = state["question"]
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"


def grade_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question
    Args:
        state(dict): The current graph state
    Returns:
        state(dict): Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_tries = state.get("max_tries",3)
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents = format_docs(documents),generation = generation.content)
    result = json_model.invoke([SystemMessage(content=hallucination_grader_instructions)]+ [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]

    # check for hallucinations
    if grade == "yes":
        answer_grader_prompt_formatted = answer_grader_prompt.format(question = question, generation = generation.content)
        result = json_model.invoke([SystemMessage(content = answer_grader_instructions)]+ [HumanMessage(content= answer_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            return "useful"
        elif state["loop_step"] <= max_tries:
            return "not useful"
        else:
            return "max tries"
    elif state["loop_step"] <= max_tries:
        return "not supported"
    else:
        return "max tries"
    
# ----------------------------------------------------------------------------------------------
# CONTROL FLOW
from langgraph.graph import StateGraph,END

workflow = StateGraph(GraphState)
workflow.add_node("websearch",web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents",grade_documents)
workflow.add_node("generate",generate)

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore":"retrieve"
    }
)

workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve","grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_generation,
    {
        "websearch":"websearch",
        "generate":"generate"
    }
)
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max_tries": END
    }
)

graph = workflow.compile()

question = "What are the important points during presenting an idea?"
inputs = {"question":question}
for event in graph.stream(inputs,stream_mode="values"):
    print(event)



