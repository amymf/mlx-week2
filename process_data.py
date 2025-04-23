import random
import pandas as pd
from datasets import load_dataset
from datasets import Dataset

ds = load_dataset("microsoft/ms_marco", "v1.1")

# Initialize mappings
doc_text_to_id = {}
doc_id_to_text = {}

all_documents = []


def get_all_documents(split: str):
    next_doc_id = 0
    total_passages_count = 0
    for example in ds[split]:
        passage_texts = example["passages"]["passage_text"]
        for passage_text in passage_texts:
            if (
                passage_text not in doc_text_to_id
            ):  # Add document only if it's not seen before
                doc_text_to_id[passage_text] = next_doc_id
                doc_id_to_text[next_doc_id] = passage_text
                next_doc_id += 1
            total_passages_count += 1
            all_documents.append(passage_text)


def build_triplets_batch(batch):
    output = {
        "query_id": [],
        "pos_doc_id": [],
        "neg_doc_id": [],
    }

    for query_id, passage_dict in zip(batch["query_id"], batch["passages"]):
        passage_texts = passage_dict["passage_text"]
        is_selected_flags = passage_dict["is_selected"]

        # Get a positive doc
        pos_doc_id = None
        for i, flag in enumerate(is_selected_flags):
            if flag:
                pos_doc_id = doc_text_to_id.get(passage_texts[i])
                break
        if pos_doc_id is None:
            continue

        # Create a set of current query's docs
        current_query_docs = set(passage_texts)

        # Sample a negative doc efficiently
        neg_doc_text = None
        attempts = 0
        while attempts < 10:  # Try a few times to avoid rare edge cases
            candidate = random.choice(all_documents)
            if candidate not in current_query_docs:
                neg_doc_text = candidate
                break
            attempts += 1
        if neg_doc_text is None:
            continue  # fallback if sampling fails

        neg_doc_id = doc_text_to_id[neg_doc_text]

        output["query_id"].append(query_id)
        output["pos_doc_id"].append(pos_doc_id)
        output["neg_doc_id"].append(neg_doc_id)

    return output


def get_triplets_batched(split: str):
    batched_output = ds[split].map(
        build_triplets_batch,
        batched=True,
        remove_columns=ds[split].column_names,
        desc="Building triplets",  # progress bar
    )
    return batched_output


def create_dataframes(split: str, batched_triplets):
    queries_df = pd.DataFrame(
        [(query["query_id"], query["query"]) for query in ds[split]],
        columns=["query_id", "query_text"],
    )
    documents_df = pd.DataFrame(
        list(doc_id_to_text.items()), columns=["doc_id", "doc_text"]
    )
    triplets_df = pd.DataFrame(
        batched_triplets, columns=["query_id", "pos_doc_id", "neg_doc_id"]
    )
    return queries_df, documents_df, triplets_df


def save_dataframes(split:str, queries_df, documents_df, triplets_df):
    queries_df.to_csv(f"data/queries_{split}.tsv", sep="\t", index=False)
    documents_df.to_csv(f"data/documents_{split}.tsv", sep="\t", index=False)
    triplets_df.to_csv(f"data/triplets_{split}.tsv", sep="\t", index=False)


def push_to_hugging_face(split: str, queries_df, documents_df, triplets_df):
    queries_dataset = Dataset.from_pandas(queries_df)
    documents_dataset = Dataset.from_pandas(documents_df)
    triplets_dataset = Dataset.from_pandas(triplets_df)

    queries_dataset.push_to_hub(f"amyf/ms-marco-queries-{split}")
    documents_dataset.push_to_hub(f"amyf/ms-marco-documents-{split}")
    triplets_dataset.push_to_hub(f"amyf/ms-marco-triplets-{split}")


def main(split: str):
    get_all_documents(split)
    batched_triplets = get_triplets_batched(split)
    queries_df, documents_df, triplets_df = create_dataframes(split, batched_triplets)
    save_dataframes(split, queries_df, documents_df, triplets_df)
    push_to_hugging_face(split, queries_df, documents_df, triplets_df)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MS MARCO dataset.")
    parser.add_argument(
        "--split", type=str, choices=["train", "validation", "test"], required=True
    )
    args = parser.parse_args()

    main(args.split)
