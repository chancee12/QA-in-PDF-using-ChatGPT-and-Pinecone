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
    # Priming the query more specifically
    primed_query = f"Details related to '{query}' in the budget documents."

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )

    result = qa.run(primed_query)[0]  # assuming result is the first value returned

    # Budget and related terminologies
    budget_terms = ['budget', 'price', 'cost', 'funding', 'expense', 'financing', 'appropriation', 'enactment', 'supplemental', 'request']
    fiscal_years_long = ['PRIOR', 'FY 2022', 'FY 2023', 'FY 2024', 'FY 2025', 'FY 2026', 'FY 2027', 'FY 2028']
    fiscal_years_short = ['PRIOR', 'FY-22', 'FY-23', 'FY-24', 'FY-25', 'FY-26', 'FY-27', 'FY-28']

    # Check for fiscal years in query and result
    has_fy_long = any(year in query for year in fiscal_years_long) or any(year in result for year in fiscal_years_long)
    has_fy_short = any(year in query for year in fiscal_years_short) or any(year in result for year in fiscal_years_short)

    # Check if query has budget-related terms or FY reference and if result has FY info
    if any(term in query.lower() for term in budget_terms) or has_fy_long or has_fy_short:
        if not (has_fy_long or has_fy_short):
            result += " The specific budget figures or fiscal year details were not identified."
    else:
        # Trimming the budget part if it's not relevant to the query
        if "Regarding budget information," in result:
            result = result.split("Regarding budget information,")[0]

    # Feedback for user when no relevant information is found
    if len(result) < 50:
        result = "Sorry, I couldn't find relevant information based on your query."

    result += "\nPlease note that answers are derived from available documents and might not capture the entire context."
    
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
    This tool is optimized for extracting budgetary and organizational details from the provided documents. 
    For best results, try to be as specific as possible in your queries. If you have any feedback or require further assistance, please [contact us](mailto:Chancee.Vincent@aximgeo.com).
    """)

hide_streamlit_elements()

if __name__ == "__main__":
    main()
