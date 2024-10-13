import json
from datasets import Dataset
from Model import tag2id
from transformers import DataCollatorForTokenClassification
#filepath
hanfloder='dataset/han/'
train=hanfloder+'hantrainbio.json'
val=hanfloder+'hanvalbio.json'
test=hanfloder+'hantestbio.json'
files=[train,val,test]

def readjosn(file:str) :
    token=[]
    nertag=[]
    with open(file=file,mode='r',encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    for i in data:
        # token.append([simplify_to_traditional(i['stentece'])])
        token.append([i['stentece']])
        nertag.append(i['bio'])
    return [token,nertag]


def delatedata(tokens,tags):
    index_list=[]
    for token,tag in zip(tokens,tags):
        inputs = tokenizer2(token[0], is_split_into_words=True)
        if len(token[0])+2 != len(inputs.input_ids):
            index=tokens.index(token)
            index_list.append(index)
    print(index_list)
    tokenlis = [n for i, n in enumerate(tokens) if i not in index_list]
    tagslis= [n for i, n in enumerate(tags) if i not in index_list]
    return tokenlis,tagslis

def create_ds(tokens,labels):
    dic = {"tokens": tokens, "ner_tags": labels}
    newds = Dataset.from_dict(dic)
    return newds

def get_Ds(filelists):
    datalist=[]
    for file in filelists:
        data=readjosn(file)
        datalist.append(data)

    datalist = [delatedata(data[0], data[1]) for data in datalist]
    dsall=[create_ds(data[0], data[1]) for data in datalist]
    return dsall
#【train，val test】
datall=get_Ds(files)


def align_labels_with_tokens(labels,tagdic):
    changelabels = [tagdic[label] for label in labels]
    """
    siku bert只要前后加一下两个词汇标记
    """
    new_labels=[-100]+changelabels+[-100]
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer2(examples['tokens'], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i in all_labels:
        new_labels.append(align_labels_with_tokens(i,tag2id))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def saveds(dslist,filepath):
    for ds,path in zip(dslist,filepath):
        newds=ds.map(tokenize_and_align_labels,batched=True,remove_columns = ["tokens","ner_tags"])
        print(newds)
        newds.save_to_disk(path)
        print("saved over")

folder="dataset/handsRobeerta/"
train=folder+'train'
val=folder+'val'
test=folder+'test'
filepath=[train,val,test]
saveds(datall,filepath)