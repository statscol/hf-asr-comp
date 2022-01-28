# Utility for fitting ngram Language Model and appending it to Wav2Vec2 HF model

## How to use

0. Create a dataset for your language of interest, and upload to Hugging Face Hub. It must contain a "text" field, consisting of sentences in your target language. See example_dataset.ipynb for an example of how to create one.

1. Build Docker image,

   ```{console}
   docker build -t sometag .
   ```

   Alternatively, download image from DockerHub: `docker pull azuur/ngram_util`.

2. Run Docker image, with the following params:

   ```{console}
   docker run -it --name somename --rm sometag USERNAME_DATASET DATASET_NAME USERNAME_MODEL MODEL_NAME HF_TOKEN_TO_WRITE_MODEL N_TO_CALCULATE_NGRAMS
   ```

   For example,

   ```{console}
   docker run -it --name test --rm ngram_util azuur es_corpora_parliament_processed "glob-asr" "test-asr-sp-model" hf_token123abc 4
   ```

3. Wait for a while. If the process is successful, the acoustic model + ngram LM should now be visible in Hugging Face Hub.
