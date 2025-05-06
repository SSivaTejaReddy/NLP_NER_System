import os
import sys
from transformers import pipeline
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_config import IO_PATH
from data_handler import load_data, save_data

class NER_Transformer:
    def __init__(self, model_name="Jean-Baptiste/roberta-large-ner-english"):
        self.ner_pipeline = pipeline(
            "ner",
            model= model_name,
            aggregation_strategy="simple",
            grouped_entities = True,

        )

  
    def ner_recog(self, text):
        try:
            entities = self.ner_pipeline(text)
            org_names = [entity["word"] for entity in entities if entity["entity_group"] == "ORG"]
            return org_names
        except Exception as e:
            print(f"Error processing text: {text[:50]}... Error: {str(e)}")
            return []


if __name__ == "__main__":
    print("Loading the data...")
    df = load_data(IO_PATH)
    print("Data loaded successfully")
    
    print("Initializing NER transformer model...")
    ner = NER_Transformer()
    
    # Using tqdm for progress bar
    tqdm.pandas(desc="Extracting ORG entities")
    df["Org_name"] = df["input"].progress_apply(lambda x: ner.ner_recog(x))
    
    print("Saving the data...")
    save_data(df, IO_PATH, )
    print(f"Data saved to {IO_PATH}")

"""
Models tired
1. Jean-Baptiste/roberta-large-ner-english -> Better extraction compared to SPacy models
Outcome: Failed to cover corner cases like (['Fort Rucker']	[])

2. dslim/bert-base-NER
Outcome : This model is not better than "Jean-Baptiste/roberta-large-ner-english" 
Des: dslim/bert-base-NER was trained only on CoNLL-2003, which has very few organization-like place names.

3. dslim/bert-large-NER
outcome: bettr than bert-base-NER but not better than "Jean-Baptiste/roberta-large-ner-english"

dbmdz/bert-large-cased-finetuned-conll03-english
"""

