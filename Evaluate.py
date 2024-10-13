from datasets import Dataset
from Model import data_collator,id2tag
import tensorflow as tf
import evaluate
import numpy as np
from transformers import TFAutoModelForTokenClassification
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

# all aug
floder='dataset/all/all_aug/'
han=floder+"zuo/"
# guner=floder+"guner/"

def getds(path):
    train=path+'train'
    val=path+'val'
    test=path+'test'
    trainds=Dataset.load_from_disk(train)
    valds=Dataset.load_from_disk(val)
    testds=Dataset.load_from_disk(test)
    tf_train_dataset = trainds.to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=1,
    )
    tf_val_dataset = valds.to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=1,
    )

    tf_test_dataset = testds.to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=1)
    return tf_train_dataset,tf_val_dataset,tf_test_dataset

# tf_train_dataset,tf_val_dataset,tf_test_dataset=getds(han)
tf_train_dataset,tf_val_dataset,tf_test_dataset=getds(han)


savedmodel='model/sikuroberta/all/aug/'

model=TFAutoModelForTokenClassification.from_pretrained(savedmodel)
# model=tf.keras.models.load_model(savedmodel+'tf_model.h5')
def getevaluate(ds):
    all_predictions = []
    all_labels = []
    i=0
    for batch in ds:
        logits = model.predict_on_batch(batch)["logits"]
        labels = batch["labels"]
        predictions = np.argmax(logits, axis=-1)
        for prediction, label in zip(predictions, labels):
            prediction=[id2tag[x] for x in prediction][1:-1]
            # print('pre')
            # print(prediction)
            # print(len(prediction))

            templabel=[]
            for label_idx in label:
                if label_idx == -100:
                    continue
                idx=label_idx.numpy()
                templabel.append(id2tag[idx])
            # print('rel')
            # print(templabel)
            # print(len(templabel))
            if len(prediction)!=len(templabel):
                i=i+1
                pass
            all_predictions.append(prediction)
            all_labels.append(templabel)
    print(str(i)+"data worng")
    print("evaluate")
    print(f1_score(all_labels, all_predictions))
    print(classification_report(all_labels, all_predictions,digits=4))
    print("over")

# getevaluate(tf_train_dataset)
# getevaluate(tf_val_dataset)
getevaluate(tf_test_dataset)