import json
from datasets import Dataset
from Model import tokenizer,tag2id
import opencc
converter = opencc.OpenCC('s2t.json')

allfloder='dataset/all/all_aug_medium/'
gunerindex=[4968,300,200]
train=allfloder+'trainbio.json'
val=allfloder+'valbio.json'
test=allfloder+'testbio.json'
files=[train,val,test]


def simplify_to_traditional(simplified_text):
    traditional_text = converter.convert(simplified_text)
    return traditional_text

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
        inputs = tokenizer(token[0], is_split_into_words=True)
        if len(token[0])+2 != len(inputs.input_ids):
            index=tokens.index(token)
            index_list.append(index)
    # print(index_list)
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

def get_all_datalist(pathlst):
    dataall = []
    for path in pathlst:
        data = readjosn(path)
        token = data[0]
        bio = data[1]
        dataall.append([token, bio])
    all = [delatedata(data[0], data[1]) for data in dataall]
    return all

def get_datalst_split(fpathlist,nlist):
    guner = []
    zuo = []
    for file, n in zip(fpathlist, nlist):
        data = readjosn(file)
        gtoken = data[0][:n]
        gunerbio = data[1][:n]
        guner.append([gtoken, gunerbio])

        zuotoken = data[0][n:]
        zuobio = data[1][n:]
        zuo.append([zuotoken, zuobio])
    guner = [delatedata(data[0], data[1]) for data in guner]
    zuo = [delatedata(data[0], data[1]) for data in zuo]

    return zuo,guner

# [【train,l]，val test】
def get_Ds_split(fpathlist,nlist):
    zuo,guner=get_datalst_split(fpathlist,nlist)
    gunerds = [create_ds(data[0], data[1]) for data in guner]
    zuods = [create_ds(data[0], data[1]) for data in zuo]
    return zuods,gunerds



#每次这个会变化，直接写死，datall[0][1]是nertags
# unique_tags = set(tag for doc in datall[0][1] for tag in doc)
# tag2id = {tag: id for id, tag in enumerate(unique_tags)}
# id2tag = {id: tag for tag, id in tag2id.items()}

def align_labels_with_tokens(labels,tagdic):
    czuogelabels = [tagdic[label] for label in labels]
    """
    siku bert只要前后加一下两个词汇标记
    """
    new_labels=[-100]+czuogelabels+[-100]
    return new_labels


def tokenize_and_align_labels(examples):
    # print(examples)
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
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
        # second check
        deltelist = []
        for i in range(len(newds)):
            a = len(newds[i]['input_ids'])
            b = len(newds[i]['labels'])
            if a != b:
                deltelist.append(i)
        cleands = newds.filter(lambda example, idx: idx not in deltelist, with_indices=True)
        print('delete index:')
        print(deltelist)
        print(cleands)
        cleands.save_to_disk(path)
        print("saved over")



zuo,guner=get_Ds_split(files,gunerindex)

zuodsfloder=[allfloder+'zuo/train',allfloder+'zuo/val',allfloder+'zuo/test']
gunerdsfoleder=[allfloder+'guner/train',allfloder+'guner/val',allfloder+'guner/test']
saveds(zuo,zuodsfloder)
saveds(guner,gunerdsfoleder)

all=get_Ds(files)
alldsfloder=[allfloder+'all/train',allfloder+'all/val',allfloder+'all/test']
saveds(all,alldsfloder)
print("=======ds============")

#保存成人民日报的格式
def save_data_2ndform(lst,name):

    sentences=lst[0]
    labels=lst[1]
    with open(name,mode="w",encoding="utf-8") as f:
        for sentence,label in zip(sentences,labels):
            print(sentence)
            print(label)
            for s,l in zip(sentence[0],label):
                print(s)
                print(l)
                f.write(s+' '+l+'\n')
            f.write("\n")

# 使用上面截断好的 zuo 和guner
def china_daily_form_data(datalist,floder):
    filepath=allfloder+floder+'/'
    allpath=[filepath+'train.txt',filepath+'dev.txt',filepath+'test.txt']
    save_data_2ndform(datalist[0],name=allpath[0])
    save_data_2ndform(datalist[1],name=allpath[1])
    save_data_2ndform(datalist[2],name=allpath[2])
    print('over')

zuolst,gunerlst=get_datalst_split(files,gunerindex)

all=get_all_datalist(files)
china_daily_form_data(zuolst,floder='zuo')

china_daily_form_data(gunerlst,floder="guner")

china_daily_form_data(all,floder='all')