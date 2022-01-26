import re
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,Wav2Vec2Processor

def clean_batch(batch):
    batch["sentence"] = re.sub("([^A-Za-zÀ-ú ])", '', batch["sentence"]).lower()
    batch["sentence"]= re.sub("([ß|þ|ð|æ])",'',batch['sentence'])
    return batch

def homologate_accents(batch):
    batch["sentence"]=re.sub("([â|ã|ä|å|à])","a",batch["sentence"])
    batch["sentence"]=re.sub("([é|ê|ë])","e",batch["sentence"])
    batch["sentence"]=re.sub("([ì|î|ï])","i",batch["sentence"])
    batch["sentence"]=re.sub("([ö|õ|ô|ò|ø])","o",batch["sentence"])
    batch["sentence"]=re.sub("ù","u",batch["sentence"])
    batch["sentence"]=re.sub("ç","c",batch["sentence"])
    
    return batch



def get_processor():

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    repo_name = "wav2vec2-xls-r-300m-spanish-small"

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor