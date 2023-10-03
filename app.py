from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def doc_preprocessing():
    loader = DirectoryLoader('data/', glob='**/*.pdf', show_progress=True)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name='dod')
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query):
    primed_query = (f"Provide a comprehensive breakdown related to '{query}'. "
                    f"The user is often seeking details on budgets, pricing, or costs. "
                    f"For budget-related inquiries, an ideal response would include figures for multiple fiscal years, "
                    f"like FY 2022, FY 2023, etc. Always present nearby mentioned budgets, pricing, or costs relevant to the query. "
                    f"Disclaimer: The provided answers are based on available information and may have variations. "
                    f"For terms with multiple definitions, provide all relevant explanations.")
    result = llm.run(primed_query)  # Assuming llm.run is equivalent to your original_retrieval_function
    return result

def main():
    st.title("Question & Answer Retrieval from PDFs")
    st.write("""
    This tool was developed by Chancee Vincent to assist in information retrieval from the following documents:
    - [Document 1](https://usg02.safelinks.protection.office365.us/?url=https%3A%2F%2Fcomptroller.defense.gov%2FPortals%2F45%2FDocuments%2Fdefbudget%2Ffy2024%2Fbudget_justification%2Fpdfs%2F03_RDT_and_E%2FOSD_PB2024.pdf&data=05%7C01%7C%7C3d94d8a8f971462e334308dbb5638175%7Cb95a24d4bf23485495ba52535e36a689%7C0%7C0%7C638303211215324281%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=cFDapCuJLsTORH6PeEOvDWdIAn%2FS1af%2BsGvWaXr0kYo%3D&reserved=0)
    - [Document 2](https://usg02.safelinks.protection.office365.us/?url=https%3A%2F%2Fcomptroller.defense.gov%2FPortals%2F45%2FDocuments%2Fdefbudget%2Ffy2024%2Fbudget_justification%2Fpdfs%2F02_Procurement%2FPB_2024_PDW_VOL_1.pdf&data=05%7C01%7C%7C3d94d8a8f971462e334308dbb5638175%7Cb95a24d4bf23485495ba52535e36a689%7C0%7C0%7C638303211215324281%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=LGbIX5KANyz%2FYUX5Hq9PDekrvEsh6oKzevLG6yCPjSc%3D&reserved=0)
    - [Document 3](https://usg02.safelinks.protection.office365.us/?url=https%3A%2F%2Fcomptroller.defense.gov%2FPortals%2F45%2FDocuments%2Fdefbudget%2Ffy2024%2Fbudget_justification%2Fpdfs%2F01_Operation_and_Maintenance%2FO_M_VOL_1_PART_1%2FOM_Volume1_Part1.pdf&data=05%7C01%7C%7C3d94d8a8f971462e334308dbb5638175%7Cb95a24d4bf23485495ba52535e36a689%7C0%7C0%7C638303211215324281%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=vUsbSXaLsmtZUHRAiGsgHTC9IixSxPfegNeb3gtUSL8%3D&reserved=0)

    **Example Queries**:
    - What department or organization is DTRA about?
    - Tell me about the DoD teleport program and the budgets involved
    ... (add more examples)
    """)

    text_input = st.text_area("Type your query:", height=150)  # Using text_area for better visibility and space
    if st.button("Retrieve Information"):
        if len(text_input) > 0:
            st.subheader("Your Query:")
            st.write(text_input)
            st.subheader("Answer:")
            answer = retrieval_answer(text_input)
            st.write(answer)

    st.write("""
    #### Note:
    This tool is optimized for extracting budgetary and organizational details from the provided documents. 
    For best results, try to be as specific as possible in your queries. If you have any feedback or require further assistance, please [contact us](mailto:Chancee.Vincent@aximgeo.com).
    """)

if __name__ == "__main__":
    main()
