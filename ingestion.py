from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    #"https://media.wizards.com/2014/downloads/dnd/PlayerDnDBasicRules_v0.2_PrintFriendly.pdf",
    # "https://media.wizards.com/2014/downloads/dnd/DMBasicRulesv.0.3_PrinterFriendly.pdf",
    # "https://media.wizards.com/2014/downloads/dnd/HoardDragonQueen_Supplement_PF_v0.3.pdf",
    # "https://media.wizards.com/2014/downloads/dnd/RiseTiamatSupplementv0.2_Printer.pdf",
    # "https://media.wizards.com/2015/downloads/dnd/EE_PlayersCompanion.pdf",
    # "https://media.wizards.com/downloads/dnd/ADVLeague_PlayerGuide_TODv1.pdf",
    # "https://media.wizards.com/2015/downloads/dnd/DDALPG_EEv1Print.pdf",
    "https://media.wizards.com/2015/downloads/dnd/UA_Eberron_v1.1.pdf"
    # "https://media.wizards.com/2015/downloads/dnd/UA_Battlesystem.pdf"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()