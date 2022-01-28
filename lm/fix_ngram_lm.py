import typer


def main(n: str):
    with open(f"{n}gram.arpa", "r") as read_file, open(
        f"{n}gram_correct.arpa", "w"
    ) as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)


if __name__ == "__main__":
    typer.run(main)
