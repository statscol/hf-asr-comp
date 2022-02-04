import re
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,Wav2Vec2Processor

def homologate_accents(batch,text_column="sentence"):
    
    batch[text_column]=re.sub("([â|ã|ä|å|à])","a",batch[text_column])
    batch[text_column]=re.sub("([é|ê|ë])","e",batch[text_column])
    batch[text_column]=re.sub("([ì|î|ï])","i",batch[text_column])
    batch[text_column]=re.sub("([ö|õ|ô|ò|ø])","o",batch[text_column])
    batch[text_column]=re.sub("ù","u",batch[text_column])
    batch[text_column]=re.sub("ç","c",batch[text_column])
    return batch

num2word={'1':'tres','2':'dos','3':'tres','4':'cuatro','5':'cinco','6':'seis','7':'siete','8':'ocho','9':'nueve','0':'cero'}

def rep_numbers_sentence(sentence:str,num2word=num2word):

    for char in sentence:
        if char in num2word:
            sentence=sentence.replace(char,num2word.get(char)+" ")
    return sentence

def clean_batch(batch,text_column="sentence"):
    batch[text_column]=rep_numbers_sentence(batch[text_column])
    batch[text_column] = re.sub("([^A-Za-zÀ-ú ])", '', batch[text_column]).lower()
    batch[text_column]= re.sub("([ß|þ|ð|æ])",'',batch[text_column])
    return batch


def get_processor():

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor