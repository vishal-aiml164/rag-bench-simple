import os
import json

def load_documents(input_dir="data/input"):
    documents = []
    doc_id = 0

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)

        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    doc_id += 1
                    documents.append({"id": doc_id, "text": text})

        elif fname.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            doc_id += 1
                            documents.append({"id": doc_id, "text": obj["text"]})
                    except json.JSONDecodeError:
                        continue

    return documents
