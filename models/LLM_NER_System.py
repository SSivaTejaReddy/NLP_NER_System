import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_config import IO_PATH
from data_handler import load_data, save_data
from groq import Groq
from dotenv import load_dotenv
load_dotenv() 
class NER_LLM:
    def __init__(self, model_name="llama-3.3-70b-versatile"): #llama3-70b-8192
        api_key = os.getenv("GROQ_API_KEY")  # Replace with Groq API environment variable name

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name  

    def ner_recog(self, text):
        try:
            prompt = f"""
            Think as an helpful assistance, and ectract the Organization name form the text.
            Below are the examples with their relevant text and organizations.
            Do not add any other text apart form organization name in the column.
           
            Examples:
            Text: How much revenue does KSM Castings Group GmbH generate?
            Organizations: ['KSM Castings Group GmbH']


            Text: What are the total earnings ofbouygues-es.co.uk?
            Organizations: ['BYES SOLAR UK LIMITED']


            Text: How much money does landrysseafood.com make?
            Organizations: ["Landry's Seafood Kemah", 'Inc.']


            Text: {text}
           
            Organizations:
            """  
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.3, 
                max_tokens=500
            )
            
            # Parse the response
            orgs_text = response.choices[0].message.content.strip()
            org_names = [name.strip() for name in orgs_text.split(",") if name.strip()]
            
            return org_names
        except Exception as e:
            print(f"Error processing text: {text[:50]}... Error: {str(e)}")
            return []


if __name__ == "__main__":
    print("Loading the data...")
    df = load_data(IO_PATH)
    print("Data loaded successfully")
    
    print("Initializing NER LLM model...")
    ner = NER_LLM()  
    
    tqdm.pandas(desc="Extracting ORG entities")
    df["Org_name"] = df["input"].progress_apply(lambda x: ner.ner_recog(x))
    
    print("Saving the data...")
    save_data(df, IO_PATH)
    print(f"Data saved to {IO_PATH}")
