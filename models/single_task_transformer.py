from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

label2id = {
    "safe": 0,
    "unsafe": 1,
    "ambiguous": 2,
}
id2label = {v: k for k, v in label2id.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

print("hello world")