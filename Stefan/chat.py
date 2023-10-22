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



user_string="What is the name of the festival?"

llm = ChatOpenAI(temperature=0,model_name="gpt-4")


# path_questions="questions.json"
# f=open(path_questions,"r",encoding="UTF-8")
# questions = json.loads(f.read())
# data=""

# for i in questions:
#     data+= i["question"]+'\n'
#     data+= i["answer"]+'\n'

# f=TextLoader('questions.txt',encoding="UTF-8").load()

# questions_splitter= CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0,separator="\n")
# questions_chunks = questions_splitter.split_documents(f)
# questions_db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())







#find category for question this new agent delete if not work
category_agent =OpenAI()
category_user_question = user_string
categories = {"public transport information","nearest facility","anythingh else"}

template_category = "given the following list of categories: {categories} : what category does: {question} : fit best in, tell me only the category?"
question_category = PromptTemplate(template= template_category, input_variables=["categories","question"])
chain_category = LLMChain(llm =category_agent , prompt = question_category)

category = chain_category.run(categories=categories, question = category_user_question)
print(category)
# category_temp= Chroma.from_texts(category,OpenAIEmbeddings())
# db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())
temp= "".join(categories)
# print(temp)
lop= Chroma.from_texts(categories,OpenAIEmbeddings())
category_embeded = OpenAIEmbeddings().embed_query(category)
category_final = lop.similarity_search_by_vector(category_embeded)

print(category_final[0].page_content)
#end categories

test_category= category_final[0].page_content

template=""
######
def CreatePromptTrans():
    
    tr=TextLoader('transport.txt',encoding='UTF-8').load()
    transport_db = Chroma.from_documents(tr,OpenAIEmbeddings())
    transport_embeded = OpenAIEmbeddings().embed_query(user_string)
    trasport_doc =transport_db.similarity_search_by_vector(transport_embeded)
    # transport_context=""
    # for j in range(3):
    #     transport_context+= trasport_doc[i].page_content
    # print(tr)
    transport_template = "use {context} combine the provided data to answer the following question : {question}"
    transport_query=PromptTemplate(template = transport_template, input_variables=["context", "question"])
    
    chain = LLMChain(llm=llm, prompt=transport_query)
    transport_answer=chain.run(context= tr, question = user_string)
    
    return transport_answer
#######

def CreatePromptQuestions():
    
    question_template = " use {context} combine the provided data to answer the following question in friendly manner considering you are a customer relation manager add a :D at the end and then remove it:{question}"
    qt=TextLoader('questions.txt',encoding="UTF-8").load()
    questions_splitter= CharacterTextSplitter(chunk_size= 1000, chunk_overlap=200,separator="\n")
    questions_chunks = questions_splitter.split_documents(qt)
    question_db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())
    questions_embeded= OpenAIEmbeddings().embed_query(user_string)
    questions_doc = question_db.similarity_search_by_vector(questions_embeded)

    questions=""
    for i in range(3):
        questions += questions_doc[i].page_content
    question_query= PromptTemplate(template = question_template , input_variables=["context","question"])
    # query.format(context= context, question= user_string)

    chain =LLMChain(llm=llm , prompt=question_query )
    queSys=chain.run(context= questions, question = user_string)
    return queSys


######
def CreatePromptCoordinates():
    
    pass



if(test_category == "public transport information"):
    print(CreatePromptTrans())
elif(test_category == "nearest facility"):
    pass
elif(test_category == "anythingh else"):
    print(CreatePromptQuestions())




# query= PromptTemplate(template = template , input_variables=["context","question"])
# # query.format(context= context, question= user_string)

# chain =LLMChain(llm=llm , prompt=query )
# queSys=chain.run(context= context, question = user_string)

# print(queSys)




# temp_split= CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0,separator="\n")
# questions_chunks = questions_splitter.split_documents(f)
# db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())

# category_embeded= OpenAIEmbeddings().embed_query(temp)
# category_final = category_temp.similarity_search_by_vector(category_embeded)

# print(category_final)





    
   





