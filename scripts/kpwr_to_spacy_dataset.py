from pathlib import Path

import argilla as rg
from datasets import load_dataset
from spacy.lang.pl import Polish
from spacy.tokens import DocBin


def make_docbin(dataset, nlp, span_key="sc"):
    rg_dataset = rg.read_datasets(
        dataset,
        tags="ner",
        task="TokenClassification",
        metadata={"lemmas": "lemmas", "orth": "orth"},
    )
    docbin = rg_dataset.prepare_for_training(framework="spacy", lang=nlp)
    docbin_wspans = DocBin()
    for doc in docbin.get_docs(nlp.vocab):
        doc.spans[span_key] = list(doc.ents)
        assert len(doc.spans[span_key]) > 0, "No spans in doc!"
        docbin_wspans.add(doc)
    return docbin_wspans


def get_docbin_len(docbin, vocab):
    return len(set(docbin.get_docs(vocab)))


def count_tokens(docbin, vocab):
    return sum([len(doc) for doc in docbin.get_docs(vocab)])


def contains_any_label(record):
    return all([True if label != 160 else False for label in record["ner"]])


def main():
    OUT_DIR = Path("data") / "spacy_kpwr"
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset("clarin-pl/kpwr-ner").filter(function=contains_any_label)

    train_ds = dataset["train"]
    val_ds, test_ds = dataset["test"].train_test_split(test_size=0.5).values()

    nlp = Polish()
    train_docbin = make_docbin(train_ds, nlp)
    val_docbin = make_docbin(val_ds, nlp)
    test_docbin = make_docbin(test_ds, nlp)

    db_basename = "kpwr_spancat_{split}.spacy"
    train_docbin.to_disk(OUT_DIR / db_basename.format(split="train"))
    val_docbin.to_disk(OUT_DIR / db_basename.format(split="val"))
    test_docbin.to_disk(OUT_DIR / db_basename.format(split="test"))

    train_len = get_docbin_len(train_docbin, nlp.vocab)
    train_tokens = count_tokens(train_docbin, nlp.vocab)

    val_len = get_docbin_len(val_docbin, nlp.vocab)
    val_tokens = count_tokens(val_docbin, nlp.vocab)

    test_len = get_docbin_len(test_docbin, nlp.vocab)
    test_tokens = count_tokens(test_docbin, nlp.vocab)

    total_docs = train_len + val_len + test_len

    info = (
        "KPWR dataset in Spacy's DocBin format.\n"
        "Source: https://huggingface.co/datasets/clarin-pl/kpwr-ner \n"
        "Validation and test sets represent dataset's 'test' split (50-50).\n"
        "Records without any 'ner' labels were omitted.\n\n"
        f"Train docs: {train_len} ({train_len / total_docs:.2%}, {train_tokens} tokens)\n"
        f"Validation docs: {val_len} ({val_len / total_docs:.2%}, {val_tokens} tokens)\n"
        f"Test docs: {test_len} ({test_len / total_docs:.2%}, {test_tokens} tokens)\n"
    )
    with open(OUT_DIR / "info.txt", "w") as f:
        f.write(info)
    print(info)


if __name__ == "__main__":
    main()
