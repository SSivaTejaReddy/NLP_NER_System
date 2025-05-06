import os
import sys
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_config import IO_PATH
from data_handler import load_data, save_data

HF_API_Keys = os.environ["HF_API_KEY"]
model = "Jean-Baptiste/roberta-large-ner-english" # bert-base-uncased , Jean-Baptiste/roberta-large-ner-english
class NER_system:
    def __init__(self, model = model, hf_api = HF_API_Keys):
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.header = {
            "Authorization" : f"Bearer {hf_api}",
        }
        
    def ner_recog(self, text):
        try:
            response = requests.post(self.api_url, headers= self.header, json ={"inputs" : text})
            response.raise_for_status()
            entities = response.json()
            org_names = [entity['word'] for entity in entities if entity.get('entity_group') == 'ORG']
            return org_names
        except Exception as e:
            print(f"NER API error for text: {text[:50]}... Error: {str(e)}")
   

if __name__ == "__main__":
    
    print("Data is loading .....")
    df = load_data(IO_PATH)

    ner = NER_system()
    
    print("Initilizing the model .....")
    tqdm.pandas(desc="Extracting ORG entities")
    df['Org_name'] = df["input"].progress_apply(lambda x: ner.ner_recog(x))

    save_data(df, IO_PATH)
    print("Data is saved .....")
