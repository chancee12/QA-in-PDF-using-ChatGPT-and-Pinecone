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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name='dod2')
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query):
    primed_query = (f"Provide a comprehensive breakdown related to '{query}'.")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    
    result = qa.run(primed_query)
    # Budget and related terminologies
    budget_terms = ['budget', 'price', 'cost', 'funding', 'expense', 'financing', 'appropriation', 'enactment', 'supplemental', 'request']
    fiscal_years = ['PRIOR', 'FY 2022', 'FY 2023', 'FY 2024', 'FY 2025', 'FY 2026', 'FY 2027', 'FY 2028', 'FY-22', 'FY-23', 'FY-24', 'FY-25', 'FY-26', 'FY-27', 'FY-28']
    
    # Checking if the query has any budget-related term or FY reference
    if any(term in query.lower() for term in budget_terms) or any(year in query for year in fiscal_years):
        # If specific FY info isn't mentioned in the result
        if not any(year in result for year in fiscal_years):
            result += (" Unfortunately, the system couldn't identify specific budget figures or relevant fiscal year details in the provided context.")
    else:
        # Trimming the budget part if it's not relevant to the query, while keeping other relevant details.
        result = result.split("Regarding budget information,")[0]
    
    result += " Please note that the provided answers are based on available documents and may not capture the full context or details."
    
    return result
    

def hide_streamlit_elements():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        footer .btn {visibility: hidden;} 
        #root > div:nth-child(1) > div > div > a {display: none;}
        #root > div:nth-child(1) > div.withScreencast > div > div > header > div.st-emotion-cache-zq5wmm.ezrtsby0 {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.title("Question & Answer Retrieval from PDFs")
    st.write("""
    This tool was developed by Chancee Vincent to assist in information retrieval from the following documents:
    - [DoD RDT&E Budget Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/03_RDT_and_E/OSD_PB2024.pdf)
    - [DoD Procurement Budget Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/02_Procurement/PB_2024_PDW_VOL_1.pdf)
    - [DoD Operation and Maintenance Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/01_Operation_and_Maintenance/O_M_VOL_1_PART_1/OM_Volume1_Part1.pdf)
    - [Air Force Aircraft Procurement Vol I](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Air%20Force%20Aircraft%20Procurement%20Vol%20I.pdf)
    - [Air Force Aircraft Procurement Vol II Mods](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Air%20Force%20Aircraft%20Procurement%20Vol%20II%20Mods.pdf)
    - [Air Force Ammunition Procurement](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Air%20Force%20Ammunition%20Procurement.pdf)
    - [Air Force Missile Procurement](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Air%20Force%20Missile%20Procurement.pdf)
    - [Air Force Other Procurement](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Air%20Force%20Other%20Procurement.pdf)
    - [Space Force Procurement](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Procurement/FY24%20Space%20Force%20Procurement.pdf)
    - [Air Force RDT&E Vol I](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Research%20and%20Development%20Test%20and%20Evaluation/FY24%20Air%20Force%20Research%20and%20Development%20Test%20and%20Evaluation%20Vol%20I.pdf)
    - [Air Force RDT&E Vol II](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Research%20and%20Development%20Test%20and%20Evaluation/FY24%20Air%20Force%20Research%20and%20Development%20Test%20and%20Evaluation%20Vol%20II.pdf)
    - [Air Force RDT&E Vol IIIa](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Research%20and%20Development%20Test%20and%20Evaluation/FY24%20Air%20Force%20Research%20and%20Development%20Test%20and%20Evaluation%20Vol%20IIIa.pdf)
    - [Air Force RDT&E Vol IIIb](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Research%20and%20Development%20Test%20and%20Evaluation/FY24%20Air%20Force%20Research%20and%20Development%20Test%20and%20Evaluation%20Vol%20IIIb.pdf)
    - [Space Force RDT&E](https://www.saffm.hq.af.mil/Portals/84/documents/FY24/Research%20and%20Development%20Test%20and%20Evaluation/FY24%20Space%20Force%20Research%20and%20Development%20Test%20and%20Evaluation.pdf)
      
    **Example Queries**:
    - What is the budget allocation for the FAB-T Force Element Terminal for FY 2024?
    - Describe the Joint Hypersonic Transition Office's mission.
    - Provide a summary of the Aircraft Integration Technologies initiative.
    - How has the budget changed for the Missile Replacement Eq-Ballistic program from FY 2023 to FY 2024?
    - What are the specific changes in the Department of Defense Operation and Maintenance, Defense-Wide funding request between FY 2023 and FY 2024, including the dollar amounts?
    - Tell me about the DoD teleport program.
    - What is the budget activity 02: National Guard equipment listed as?
    - What is the Silent Knight Radar (SKR) Program?
    """)

    text_input = st.text_area("Type your query:", height=150)  # Using text_area for better visibility and space
    if st.button("Retrieve Information"):
            if len(text_input) > 0:
                st.subheader("Your Query:")
                st.write(text_input)
                st.subheader("Answer:")
                
                with st.spinner('Sifting through the information now...'):
                    answer = retrieval_answer(text_input)
                
                st.write(answer)

    st.write("""
    #### Note:
    This tool is optimized for extracting textual details from the provided documents. 
    For best results, try to be as specific as possible in your queries. If you have any feedback or require further assistance, please [contact us](mailto:Chancee.Vincent@aximgeo.com).
    """)

hide_streamlit_elements()

if __name__ == "__main__":
    main()
