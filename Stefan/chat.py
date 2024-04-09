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
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory,
)
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
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain, RetrievalQA, SimpleSequentialChain

# from langchain.chains import LLMBashChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import geojson
import geopandas as gpd
from math import sqrt
import math


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


def ChatFuction(user_string: str):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    final_string = ""
    # find category for question this new agent delete if not work
    category_agent = OpenAI()
    category_user_question = user_string
    categories = {"public transport information", "nearest facility", "anythingh else"}

    template_category = "given the following list of categories: {categories} : what category does: {question} : fit best in, tell me only the category?"
    question_category = PromptTemplate(
        template=template_category, input_variables=["categories", "question"]
    )
    chain_category = LLMChain(llm=category_agent, prompt=question_category)

    category = chain_category.run(
        categories=categories, question=category_user_question
    )
    print(category)
    # category_temp= Chroma.from_texts(category,OpenAIEmbeddings())
    # db = Chroma.from_documents(questions_chunks,OpenAIEmbeddings())
    temp = "".join(categories)
    # print(temp)
    lop = Chroma.from_texts(categories, OpenAIEmbeddings())
    category_embeded = OpenAIEmbeddings().embed_query(category)
    category_final = lop.similarity_search_by_vector(category_embeded)

    print(category_final[0].page_content)
    # end categories

    test_category = category_final[0].page_content

    template = ""
    ######

    def CreatePromptTrans():
        tr = TextLoader("transport.txt", encoding="UTF-8").load()
        transport_db = Chroma.from_documents(tr, OpenAIEmbeddings())
        transport_embeded = OpenAIEmbeddings().embed_query(user_string)
        trasport_doc = transport_db.similarity_search_by_vector(transport_embeded)
        # transport_context=""
        # for j in range(3):
        #     transport_context+= trasport_doc[i].page_content
        # print(tr)
        transport_template = "use {context} combine the provided data to answer the following question : {question}"
        transport_query = PromptTemplate(
            template=transport_template, input_variables=["context", "question"]
        )

        chain = LLMChain(llm=llm, prompt=transport_query)
        transport_answer = chain.run(context=tr, question=user_string)

        return transport_answer

    #######

    def CreatePromptQuestions():
        question_template = " use {context} combine the provided data to answer the following question in friendly manner considering you are a customer relation manager add a :D at the end and then remove it:{question}"
        qt = TextLoader("questions.txt", encoding="UTF-8").load()
        questions_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separator="\n"
        )
        questions_chunks = questions_splitter.split_documents(qt)
        question_db = Chroma.from_documents(questions_chunks, OpenAIEmbeddings())
        questions_embeded = OpenAIEmbeddings().embed_query(user_string)
        questions_doc = question_db.similarity_search_by_vector(questions_embeded)

        questions = ""
        for i in range(3):
            questions += questions_doc[i].page_content
        question_query = PromptTemplate(
            template=question_template, input_variables=["context", "question"]
        )
        # query.format(context= context, question= user_string)

        chain = LLMChain(llm=llm, prompt=question_query)
        queSys = chain.run(context=questions, question=user_string)
        return queSys

    ######

    def CreatePromptCoordinates():
        f_json = open("cotete.json")
        data_json = json.load(f_json)
        f_geojson = open("geo.geojson")
        # data_geojson= gpd.read_file("geo.geojson")
        data_geojson = json.load(f_geojson)
        postion = [46.989920, 6.929978]

        for j in data_geojson["features"]:
            ligma = j["properties"]["centerpoint"]
            # x= j["geometry"].centroid.x
            vect = list(map(float, ligma.split(",")))
            x = vect[0]
            y = vect[1]
            j["distance"] = sqrt(pow((x - postion[0]), 2) + pow((y - postion[1]), 2))

        for j in range(len(data_geojson)):
            for k in range(j, len(data_geojson)):
                if (
                    data_geojson["features"][k]["distance"]
                    > data_geojson["features"][j]["distance"]
                ):
                    aux = data_geojson["features"][k]
                    data_geojson["features"][k] = data_geojson["features"][j]
                    data_geojson["features"][j] = aux

        position_string = ""
        for j in data_geojson["features"]:
            if "place_name" in j:
                position_string += (
                    j["place_name"]
                    + "\n"
                    + j["place_description"]
                    + "\n"
                    + j["properties"]["centerpoint"]
                    + "\n"
                )
        position_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        position_chunks = position_splitter.split_text(position_string)
        position_db = Chroma.from_texts(position_chunks, OpenAIEmbeddings())

        print(type(position_chunks))
        print(len(position_chunks))

        position_template = "give me the coordinates of the first stall that can fulfill the wishes of the following rquest {question} from the following list of stalls {location}.Only tell us the coordinates with a , between them .However if you do not find an apropriate stall respond with no"
        position_query = PromptTemplate(
            template=position_template, input_variables=["location", "question"]
        )

        for k in position_chunks:
            location_description = k
            chain = LLMChain(llm=llm, prompt=position_query)
            locationSys = chain.run(location=location_description, question=user_string)
            print("aici")

            if not locationSys.lower() == "no":
                temp = ""
                temp += "&origin=" + str(postion[0])
                temp += "," + str(postion[1])
                for m in range(len(locationSys.split(","))):
                    if m == 0:
                        temp += "&destination=" + locationSys.split(",")[1]
                    if m == 1:
                        temp += "," + locationSys.split(",")[0]
                print(temp)
                return temp
        return "I could not find such a location."

    if test_category == "public transport information":
        return CreatePromptTrans()

    elif test_category == "nearest facility":
        print("\n\nresponse\n\n\n\n")
        return CreatePromptCoordinates()

    elif test_category == "anythingh else":
        return CreatePromptQuestions()

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
