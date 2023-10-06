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
    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name='dod3')
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
    st.title("Question & Answer Retrieval")
    st.write("This tool was developed to assist in information retrieval from the following documents:")

    with st.expander("PDF Links"):
        st.write("""
        ### DoD Links
        - [DoD RDT&E Budget Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/03_RDT_and_E/OSD_PB2024.pdf)
        - [DoD Procurement Budget Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/02_Procurement/PB_2024_PDW_VOL_1.pdf)
        - [DoD Operation and Maintenance Justification](https://comptroller.defense.gov/Portals/45/Documents/defbudget/fy2024/budget_justification/pdfs/01_Operation_and_Maintenance/O_M_VOL_1_PART_1/OM_Volume1_Part1.pdf)
        ### Air & Space Force Links
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
        ### Navy Links
        - [PMC Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/PMC_Book.pdf)
        - [SCN Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/SCN_Book.pdf)
        - [WPN Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/WPN_Book.pdf)
        - [APN BA1-4 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/APN_BA1-4_Book.pdf)
        - [APN BA5 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/APN_BA5_Book.pdf)
        - [APN BA6-7 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/APN_BA6-7_Book.pdf)
        - [OPN BA1 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/OPN_BA1_Book.pdf)
        - [OPN BA2 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/OPN_BA2_Book.pdf)
        - [OPN BA3 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/OPN_BA3_Book.pdf)
        - [OPN BA4 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/OPN_BA4_Book.pdf)
        - [OPN BA5-8 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/OPN_BA5-8_Book.pdf)
        - [PANMC Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/PANMC_Book.pdf)
        - [RDTEN BA1-3 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/RDTEN_BA1-3_Book.pdf)
        - [RDTEN BA4 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/RDTEN_BA4_Book.pdf)
        - [RDTEN BA5 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/RDTEN_BA5_Book.pdf)
        - [RDTEN BA6 Book](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/RDTEN_BA6_Book.pdf)
        - [RDTEN BA7-8 Book was not included due to size](https://www.secnav.navy.mil/fmc/fmb/Documents/24pres/RDTEN_BA7-8_Book.pdf)
        ### Army Links
        - [Aircraft Procurement Army](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Aircraft%20Procurement%20Army.pdf)
        - [Missile Procurement Army](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Missile%20Procurement%20Army.pdf)
        - [Procurement of Weapons and Tracked Combat Vehicles](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Procurement%20of%20Weapons%20and%20Tracked%20Combat%20Vehicles.pdf)
        - [Procurement of Ammunition Army](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Procurement%20of%20Ammunition%20Army.pdf)
        - [Other Procurement - BA 1 - Tactical & Support Vehicles](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Other%20Procurement%20-%20BA%201%20-%20Tactical%20&%20Support%20Vehicles.pdf)
        - [Other Procurement - BA 2 - Communications and Electronics](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Other%20Procurement%20-%20BA%202%20-%20Communications%20and%20Electronics.pdf)
        - [Other Procurement - BA 3 & 4 - Other Support Equipment](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/Procurement/Other%20Procurement%20-%20BA%203%20&%204%20-%20Other%20Support%20Equipment.pdf)
        - [RDTE Vol 1 - Budget Activity 1](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%201-Budget%20Activity%201.pdf)
        - [RDTE Vol 1 - Budget Activity 2](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%201-Budget%20Activity%202.pdf)
        - [RDTE Vol 1 - Budget Activity 3](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%201-Budget%20Activity%203.pdf)
        - [RDTE Vol 2 - Budget Activity 4A](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%204A.pdf)
        - [RDTE Vol 2 - Budget Activity 4B](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%204B.pdf)
        - [RDTE Vol 2 - Budget Activity 5A](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%205A.pdf)
        - [RDTE Vol 2 - Budget Activity 5B](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%205B.pdf)
        - [RDTE Vol 2 - Budget Activity 5C](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%205C.pdf)
        - [RDTE Vol 2 - Budget Activity 5D](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%202-Budget%20Activity%205D.pdf)
        - [RDTE Vol 3 - Budget Activity 6](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%203-Budget%20Activity%206.pdf)
        - [RDTE Vol 3 - Budget Activity 7](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%203-Budget%20Activity%207.pdf)
        - [RDTE Vol 3 - Budget Activity 8](https://www.asafm.army.mil/Portals/72/Documents/BudgetMaterial/2024/Base%20Budget/rdte/RDTE-Vol%203-Budget%20Activity%208.pdf)
        """)

    st.write("""
    **Example Queries**:
    - What is the budget allocation for the FAB-T Force Element Terminal for FY 2024?
    - Describe the Joint Hypersonic Transition Office's mission.
    - Provide a summary of the Aircraft Integration Technologies initiative.
    - How has the budget changed for the Missile Replacement Eq-Ballistic program from FY 2023 to FY 2024?
    - What are the specific changes in the Department of Defense Operation and Maintenance, Defense-Wide funding request between FY 2023 and FY 2024, including the dollar amounts?
    - Tell me about the DoD teleport program.
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
    - This tool is optimized for extracting textual details from the provided documents. However, many of the tables are unstructured images and may not be contextually understood.
    - For best results, try to be as specific as possible in your queries. If you have any feedback or require further assistance, please [contact](mailto:Chancee.Vincent@aximgeo.com).
    """)

hide_streamlit_elements()

if __name__ == "__main__":
    main()
