import subprocess
import shutil
from pathlib import Path

import typer
from huggingface_hub import Repository
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM


def main(username_to_clone: str, model_to_clone: str, n: str):
    processor = AutoProcessor.from_pretrained(f"{username_to_clone}/{model_to_clone}")

    vocab_dict = processor.tokenizer.get_vocab()

    sorted_vocab_dict = {
        k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
    }

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()), kenlm_model_path=f"{n}gram_correct.arpa"
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )

    repo = Repository(
        local_dir=model_to_clone, clone_from=f"{username_to_clone}/{model_to_clone}"
    )

    dirpath = Path(f"{model_to_clone}") / "language_model"
    if dirpath.exists() and dirpath.is_dir():
        print("Found existing ngram model in repo. Replacing with new model.")
        shutil.rmtree(dirpath)

    processor_with_lm.save_pretrained(model_to_clone)

    subprocess.call(
        [
            "kenlm/build/bin/build_binary",
            f"{model_to_clone}/language_model/{n}gram_correct.arpa",
            f"{model_to_clone}/language_model/{n}gram.bin",
        ]
    )
    subprocess.call(["rm", f"{model_to_clone}/language_model/{n}gram_correct.arpa"])

    repo.push_to_hub(commit_message="Upload lm-boosted decoder")


if __name__ == "__main__":
    typer.run(main)
