from transformers import BertTokenizer,TFAutoModelForTokenClassification,DataCollatorForTokenClassification,RobertaTokenizer,AutoTokenizer

tag2id={'O': 0, 'I-TIME': 1, 'I-PER': 2, 'I-LOC': 3, 'B-LOC': 4, 'B-PER': 5, 'B-TIME': 6,'B-OFI':7,'I-OFI':8}
id2tag={0: 'O', 1: 'I-TIME', 2: 'I-PER', 3: 'I-LOC', 4: 'B-LOC', 5: 'B-PER', 6: 'B-TIME',7:'B-OFI',8:'I-OFI'}


# modelpath='D:\PythonProjects\LLMs\siku-bert'
# #示用sikubet的tokenizer
# tokenizer=BertTokenizer.from_pretrained(modelpath)
# model= TFAutoModelForTokenClassification.from_pretrained(
#     pretrained_model_name_or_path=modelpath,
#     id2label=id2tag,
#     label2id=tag2id,
#     from_pt=True,
# )
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


modelpath="D:\PythonProjects\LLMs\sikuroberta"
tokenizer=BertTokenizer.from_pretrained(modelpath)
model=TFAutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=modelpath,
    id2label=id2tag,
    label2id=tag2id,
    from_pt=True,)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print(model.config)