import os
import sys
import spacy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_config import IO_PATH
from data_handler import load_data, save_data

model = "en_core_web_lg"  # used "en_core_web_sm" model but not accurate."en_core_web_trf"
nlp = spacy.load(model)


class NER_Spacy:
    def __init__(self, text):
        self.text = text
    def ner_recog(self):
        doc = nlp(self.text)
        Org_name = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                Org_name.append(ent.text)
        return Org_name
        


if __name__ == "__main__":
    
    print("Loading the data ...")
    df = load_data(IO_PATH)
    print("Data is Loaded")

    print("Triggering the model ....")
    ns = NER_Spacy(text= df)
    """Instead of passing entire data frame pass each row other wise it popup with error as below
       "expected a string as input but not a dataframe,
       because the class is created to handle single text rows".
    """
    df["Org_name"] = df["input"].apply(lambda x : NER_Spacy(x).ner_recog())

    print("Saving the data ...")
    save_data(df, IO_PATH)
    print("Extracted Data is Saved")