from transformers import create_optimizer
from datasets import Dataset
from Model import model,data_collator,id2tag,tag2id
import tensorflow as tf
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import numpy as np
# read data

# all with agu
floder='dataset/all/all_aug/all/'

train=floder+'train'
val=floder+'val'
test=floder+'test'
trainds=Dataset.load_from_disk(train).shuffle()
valds=Dataset.load_from_disk(val)
testds=Dataset.load_from_disk(test)
tf_train_dataset = trainds.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=2,
)
tf_val_dataset = valds.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=1,
)

tf_test_dataset=testds.to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=1)

model.summary()
# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 10
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

keras_callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01,restore_best_weights=True),
        # tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", verbose=2, patience=5, mode="max",restore_best_weights=True)
    ]

model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    callbacks=keras_callbacks,
    epochs=num_epochs,
)

output_dir ='model/sikuroberta/all/aug/'
model.save_pretrained(output_dir)
print("model Saved")

