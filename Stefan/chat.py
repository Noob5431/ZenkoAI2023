import os
import textwrap
 
import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import json
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore import InMemoryDocstore
from langchain.document_loaders import JSONLoader
from pathlib import Path
from pprint import pprint
import getpass
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import  LLMChain, RetrievalQA, SimpleSequentialChain
#from langchain.chains import LLMBashChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma



user_string="What is the name of the festival"

llm = OpenAI(temperature=0)


# path_questions="questions.json"
# f=open(path_questions,"r",encoding="UTF-8")
# questions = json.loads(f.read())
# data=""

# for i in questions:
#     data+= i["question"]+'\n'
#     data+= i["answer"]+'\n'
f=TextLoader('questions.txt',encoding="UTF-8").load()


questions_splitter= CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0,separator="\n")
questions_chunks = questions_splitter.split_documents(f)
db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())

print(type(db))
questions_embeded= OpenAIEmbeddings().embed_query(user_string)
docs = db.similarity_search_by_vector(questions_embeded)

#context for the questions
context=""
for i in range(3):
    context += docs[i].page_content


template = "use {context} combine the provided data to answer the following question: {question}"


query= PromptTemplate(template = template , input_variables=["context","question"])
# query.format(context= context, question= user_string)

chain =LLMChain(llm=llm , prompt=query )
queSys=chain.run(context= context, question = user_string)

print(queSys)

#find category for question this new agent delete if not work
category_agent =OpenAI()
category_user_question = user_string
categories = {"public transport information","pathfinding to a location","anythingh else"}

template_category = "given the following list of categories: {categories} : what category does: {question} : fit best in, tell me only the category?"
question_category = PromptTemplate(template= template_category, input_variables=["categories","question"])
chain_category = LLMChain(llm =category_agent , prompt = question_category)

category = chain_category.run(categories=categories, question = category_user_question)
print(category)

category_temp= Chroma.from_texts(category,OpenAIEmbeddings())
# db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())
temp= "\n".join(categories)
print(temp)

category_embeded= OpenAIEmbeddings().embed_query(temp)
category_final = category_temp.similarity_search_by_vector(category_embeded)

print(category_final)










