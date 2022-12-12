# In this notebook we learn about how we can use neural network to detect text.
# FOR BETTER UNDERSTANDING VISIT MY COLLAB link: https://colab.research.google.com/drive/18aDqfBvWpKKnU23dO-isafXqU3uQOYmT?usp=sharing
# source tensorflow text classification link:https://www.tensorflow.org/tutorials/keras/text_classification

# Importing libraries
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import numpy
import tensorflow as tf

# Downloading and exploring Dataset

# Url from where we download the data
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# Using tensorflow keras utility get_file to get the file and unzip in
dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, 
                                  cache_dir='.', cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset),'aclImdb')

# Listing all the directories
os.listdir(dataset_dir)
# Listing content of Train directroy 
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# Reading data of train_dir/pos/1181_9
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

# Remove unuseful folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Now we will use tf.keras.utils.text_dataset_from_directory() to get data 
# from dir into two classes = class a and class b
batch_size =32
seed = 42
# Making Train dataset
raw_train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', 
                                                          batch_size=batch_size,
                                                          validation_split=0.2,
                                                          subset= 'training',
                                                          seed=seed)

# Lets see what happend with data by iterating some 
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print('Review', text_batch.numpy()[i])
    print('Label', label_batch.numpy()[i])

# Now lets check which label corresponds to which type of review
print('Label 0', raw_train_ds.class_names[0])
print('Label 1', raw_train_ds.class_names[1])

# Making Validation dataset
raw_val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', 
                                                        batch_size=batch_size,
                                                        validation_split=0.2,
                                                        subset='validation',
                                                        seed=seed)

# Making Test dataset
raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train',
                                                         batch_size=batch_size)

from tensorflow._api.v2 import strings
# Now we will prepare Dataset for training

# Making function to remove html tags from the file
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html=tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


# Now we create a layer for TextVectorization 

# defining some constants for the model like explicit max, truncate sequences
max_features = 10000
sequence_length = 250


vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
)

# Now we make text_only dataset (without labels) then call adapt

train_text = raw_train_ds.map(lambda x, y:x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text) , label

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print('Review', first_review)
print('Lable', raw_train_ds.class_names[first_label])
print('Vectorized review', vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Now we vectorize every dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# configuring dataset

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)

# creating model 

embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1 , embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.summary()

# compiling model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics = tf.metrics.BinaryAccuracy(threshold=0.0))

# Training model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds,epochs=epochs)

# Evaluting model
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

# assigning variables
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# Ploting data for loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Ploting data for Accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Exporting model

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  tf.keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)


examples = [
  "The movie was great and I liked it alot!",
  "The movie was okay.But can be better",
  "The movie was terrible never watch this move It's a massive fail"
]

export_model.predict(examples)
