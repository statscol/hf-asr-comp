import torch
from datasets import load_dataset, load_metric, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2ForCTC
import numpy as np
from utils import clean_batch,homologate_accents,get_processor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
import argparse

REPO_NAME_BASE="facebook/wav2vec2-xls-r-300m"
REPO_OUT="jhonparra18/wav2vec2-large-xls-r-300m-spanish-large"

processor=get_processor()

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch




##apply it for every audio

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


wer_metric = load_metric("wer")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_model(tr:float,tst:float):


    common_voice_train = load_dataset("common_voice", "es", split="train+validation").train_test_split(train_size=tr)['train']
    common_voice_test = load_dataset("common_voice", "es", split="test").train_test_split(train_size=tst)['train']

    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

    common_voice_train = common_voice_train.map(clean_batch)
    common_voice_train = common_voice_train.map(homologate_accents)
    common_voice_test = common_voice_test.map(clean_batch)
    common_voice_test = common_voice_test.map(homologate_accents)

    ##preprocessing

    ##convert to 16000 hz
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16000))

    
    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        REPO_NAME_BASE, 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_encoder()


    training_args = TrainingArguments(
    output_dir=REPO_OUT,
    group_by_length=True,
    per_device_train_batch_size=18,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=20,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=800,
    eval_steps=400,
    logging_steps=400,
    learning_rate=2e-4,
    warmup_steps=300,
    save_total_limit=30,
    push_to_hub=True,
    report_to="wandb",
    run_name="wav2vec-large-noLM"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )

    ###
    trainer.train()

if __name__=='__main__':
    import wandb
    wandb.init(project="wav2vec-spanish")
    
    parser = argparse.ArgumentParser(description = 'ASR Parser')
    parser.add_argument('-tr',type=float,help="train sample ratio",dest="tr_size")
    parser.add_argument('-ts',type=float,help="test sample ratio",dest="ts_size")
    args=parser.parse_args()
    train_model(args.tr_size,args.ts_size)
    
