# image classification
import tensorflow as tf
from huggingface_hub import login
import os
from dotenv import load_dotenv 
load_dotenv()
login(token=os.getenv("HF_TOKEN"),add_to_git_credential=True)

from datasets import load_dataset
food=load_dataset("food101",split="train[:5000]")

food=food.train_test_split(test_size=0.2)

print(food["train"][0])

labels=food["train"].features["label"].names
label2id,id2label=dict(),dict()
for i,label in enumerate(labels):
    label2id[label]=str(i)
    id2label[str(i)]=label

# preprocess

from transformers import AutoImageProcessor

checkpoint="google/vit-base-patch16-224-in21k"
image_processor=AutoImageProcessor.from_pretrained(checkpoint,use_fast=True)

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
normalize=Normalize(mean=image_processor.image_mean,std=image_processor.image_std)
size=(
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"],image_processor.size["width"])
)
_transforms=Compose([RandomResizedCrop(size),ToTensor(),normalize])

def transforms(examples):
    examples["pixel_values"]=[_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

food=food.with_transform(transforms)

from transformers import DefaultDataCollator

data_collator=DefaultDataCollator()

from tensorflow import keras


size = (image_processor.size["height"], image_processor.size["width"])

train_data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = keras.Sequential(
    [
        tf.keras.layers.CenterCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)

import numpy as np
import tensorflow as tf
from PIL import Image


def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

import evaluate
accuracy=evaluate.load("accuracy")

import numpy as np
def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    predictions=np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions,references=labels)
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

# model = AutoModelForImageClassification.from_pretrained(
#     checkpoint,
#     num_labels=len(labels),
#     id2label=id2label,
#     label2id=label2id,
# )

# training_args = TrainingArguments(
#     output_dir="my_awesome_food_model",
#     remove_unused_columns=False,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     push_to_hub=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=food["train"],
#     eval_dataset=food["test"],
#     processing_class=image_processor,
#     compute_metrics=compute_metrics,
# )

# trainer.train()
from transformers import create_optimizer

batch_size = 16
num_epochs = 5
num_train_steps = len(food["train"]) * num_epochs
learning_rate = 3e-5
weight_decay_rate = 0.01

optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)
from transformers import TFAutoModelForImageClassification

model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)
from transformers import create_optimizer
batch_size=16
num_epochs=5
num_train_steps=len(food["train"])*num_epochs
learning_rate=3e-5
weight_decay_rate=0.01
optimizer,lr_schedule=create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0
)
from transformers import TFAutoModelForImageClassification
model=TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)
tf_train_dataset=food["train"].to_tf_dataset(
    columns="pixel_values",label_cols="label",shuffle=True,batch_size=batch_size,collate_fn=data_collator
)
tf_eval_dataset=food["test"].to_tf_dataset(
    columns="pixel_values",label_cols="label",shuffle=True,batch_size=batch_size,collate_fn=data_collator
)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer,loss=loss)
from transformers.keras_callbacks import KerasMetricCallback,PushToHubCallback
metric_callback=KerasMetricCallback(metric_fn=compute_metrics,eval_dataset=tf_eval_dataset)
push_to_hub_callback=PushToHubCallback(
    output_dir="food_classifier",
    tokenizer=image_processor,
    save_strategy="no",
)
callbacks=[metric_callback,push_to_hub_callback]