from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging
import warnings
import chromadb
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_groq():
    """Initialize the ChatGroq model with error handling."""
    try:
        return ChatGroq(
            model="mixtral-8x7b-32768",
            groq_api_key=os.getenv("Grok_API_KEY"),
            temperature=0,
            max_tokens=None,
            max_retries=2
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {str(e)}")
        raise

def query_model(llm, query):
    """Send a query to the model with error handling."""
    try:
        response = llm.invoke(query)
        return response.content
    except Exception as e:
        logger.error(f"Failed to query model: {str(e)}")
        raise

def load_webpage(url):
    """Load and extract content from a webpage with error handling."""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content if data else None
    except Exception as e:
        logger.error(f"Failed to load webpage {url}: {str(e)}")
        raise

def main():
    try:
        # Initialize Groq
        llm = initialize_groq()
        
        # Query the model
        result = query_model(llm, "what's the capital of illinois")
        #print("Model response:", result)
        
        # Load LinkedIn profile
        linkedin_content = load_webpage("https://jobs.nike.com/job/R-43433?from=job%20search%20funnel")
        print("\nLinkedIn profile content:", linkedin_content)
        promptt=PromptTemplate.from_template(
            """
            ###scrapped text from  a carrer website:
            {linkedin_content}
            ###instructions:
            your job is to extract the job posting and return them in the json format containing following keys:
            'role','expereince','skills','description' 
            only return the valid jason
            ###valid json (no preamble):
              
             """
        )
        chain_extract=promptt | llm
        res=chain_extract. invoke(input={'linkedin_content':linkedin_content})
        #print(res.content)
        json_parser=JsonOutputParser()
        json_res=json_parser.parse(res.content)
        #print(json_res)
        df=pd.read_csv("/Users/admin/Desktop/codebasicsgenai/my_portfolio.csv")
        client1=chromadb.PersistentClient('vectorstore')
        collec=client1.get_or_create_collection(name="portfolio")
        if not collec.count():
            for _, row in df.iterrows():
                collec.add(documents=["Techstack"],metadatas={"links": row["Links"]},ids=[str(uuid.uuid4())])
        links=collec.query(query_texts=["expereince in python"],n_results=4)
        print(links)
        prompt1=PromptTemplate.from_template(""" 
                 ### job description
                 {json_res}, 
                 ### chromadb database
                 {df}                            
                 ### instruction
                 you have been provided with a job description and a database you have to llok into the database and by seeing the job description
                 check for the appropriate skiils which matches the jd and write a cold email of no more than 60 words for the recruitter and provide 
                with the aprropriate link attached to the skill in the database
                link to the chromadb databse is {df} and the job description is {json_res}
                ### return  email (no preamble)  """)
        chain_email=prompt1|llm
        res1=chain_email.invoke({"json_res":json_res,"df":df})
        print(res1.content)


    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())