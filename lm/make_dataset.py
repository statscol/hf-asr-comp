import typer
from datasets import load_dataset

def main(username: str, dataset: str):
    dataset = load_dataset(f"{username}/{dataset}", split="train")

    with open("text.txt", "w") as file:
        file.write(" ".join(dataset["text"]))


if __name__ == "__main__":
    typer.run(main)
