#!/usr/bin/env python
# coding: utf-8

# # Korean vishing Detection using KoBERT model from HuggingFace on TensorFlow

# # ***데이터 전처리까지 이상없이 동작 - 이전에는 전처리한 파일의 인코딩 방식 오류로 제대로 읽혀지지가 않았음***
# 
# 
# 

# ## Install dependencies

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Install the tensorFlow to work on GPU for WSL2 Ubuntu 20.04 LTS (Focal Fossa) on Windows 11 Pro as follows described in the [TensorFlow documentation](https://www.tensorflow.org/install/pip#windows-wsl2).
# 1. Install WSL2
# 2. Install the NVIDIA drive
# 3. NVIDIA’s setup docs for CUDA in WSL2
# 4. Install TensorFlow
# 5. Verify the installation
# 6. Install the transformers library
# 7. Install the pandas library
# 8. Install the numpy library
# 9. Install the scikit-learn library
# 10. Install the matplotlib library
# 11. Install the seaborn library
# (optional)
# 12. Install the tqdm library
# 13. Install the sentencepiece library
# 14. Install the pydot library
# 15. Install the graphviz library

# In[2]:


# Verify the installation:
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow is using GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# In[3]:


# !pip install transformers pandas numpy


# In[4]:


# !pip install --upgrade pip


# In[5]:


# !pip install tensorflow
# !pip3 install tensorflow[and-cuda]


# In[6]:


# load the libraries
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification ,TFBertModel, AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AdamWeightDecay
import pandas as pd
import numpy as np
import re, os, time, gc, psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical
from transformers import AdamW
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers import AdamW,
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical


# ## Load the dataset and cleaning

# In[7]:


# Import the dataset KorCCVi
df = pd.read_csv('/content/drive/MyDrive/voice/Korean_Voice_Phishing_Detection/KoBERT/KorCCVi_v2.csv')
# df = pd.read_csv('KorCCViD_v1.3_fullcleansed.csv')
df.head()


# In[8]:


df.info()


# In[9]:


# visualize the dataset with pie chart
df['label'].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['green', 'red'])
plt.title('Percentage of non-vishing and vishing text')
plt.show()


# In[10]:


# function to plot the class distribution
def plot_class_distribution(data, title):
    sns.set(style="whitegrid")
    # sns.set(style="ticks")
    ax = sns.countplot(x='label', data=data)
    ax.set_title(title)

    # Annotate the bars with the number of samples
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.show()


# In[11]:


# drop the colum we don't need
df.drop(['confidence'], axis=1, inplace=True)
plot_class_distribution(df, 'Full Dataset Class Distribution')


# In[12]:


# function to perform the cleaning parts
def apply_replacement(src_df, replace_func):
    ret_df = src_df
    ret_df['transcript'] = ret_df['transcript'].apply(lambda x: replace_func(x))
    return ret_df


# In[13]:


# remove the stopword from the dataset
def stopword_replace(x):
    example_word_replace_list = {'을': '',
                                 '를': '',
                                '이': '',
                                '가': ' ',
                                'ㅡ': '',
                                '은': '',
                                '는': '',
                                'XXX': '',
                                'xxx': '',
                                '어요': '',
                                '아니': '',
                                '입니다': '',
                                '에서': '',
                                '니까': '',
                                '으로': '',
                                '근데': '',
                                '습니다': '',
                                '습니까': '',
                                '저희': '',
                                '합니다': '',
                                '하고': '',
                                '싶어요': '',
                                '있는': '',
                                '있습니다': '',
                                '싶습니다': '',
                                '그냥': '',
                                '고요': '',
                                '에요': '',
                                '예요': '',
                                '으시': '',
                                '그래서': '',}


    for i in example_word_replace_list:
        x = x.replace(i, example_word_replace_list[i])
    return x


# In[14]:


# remove the unwanted word and characters from the dataset
def word_replace(x):
    example_word_replace_list = {'o/': '',
                                 'b/': '',
                                'n/': '',
                                '\n': ' ',
                                'name': '',
                                'laughing': '',
                                'clearing': '',
                                'singing': '',
                                'applauding': ''}

    for i in example_word_replace_list:
        x = x.replace(i, example_word_replace_list[i])
    return x


# In[15]:


# remove the special character from the transcripts
def remove_special_characters(sentence):
    sentence = re.sub(r"[^a-zA-Z0-9ㄱ-ㅣ가-힣]", ' ', sentence)
    sentence = re.sub(r"[-~=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]", '', sentence)
    return sentence


# In[16]:


# remove x and O from the transcripts
def remove_x_o(sentence):
    sentence = re.sub(r"[xX]", '', sentence)
    sentence = re.sub(r"[oO]", '', sentence)
    sentence = re.sub(r"(o|O|\ㅇ|0|x){2,}", '', sentence)
    return sentence


# In[17]:


# remove all the digits and numbers from the transcripts
def remove_digits(sentence):
    sentence = re.sub(r"[0-9]", '', sentence)
    return sentence


# In[18]:


# remove all extra spaces from the transcripts
def remove_extra_spaces(sentence):
    sentence = re.sub(r"\s+", ' ', sentence)
    return sentence


# In[19]:


# Apply all the cleaning functions to the dataset
df = apply_replacement(df, word_replace)
df = apply_replacement(df, stopword_replace)
df = apply_replacement(df, remove_special_characters)
df = apply_replacement(df, remove_x_o)
df = apply_replacement(df, remove_digits)
df = apply_replacement(df, remove_extra_spaces)


# In[20]:


# print the row with any English character in the transcript
print(df[df['transcript'].str.contains('[a-zA-Z]')].head(5))


# In[21]:


# save the cleaned dataset
df.to_csv('KorCCVi_v2_cleaned.csv', index=False)


# ## Training the KoBERT model using the Tokenizer provided by KoBERT

# In[ ]:


# Load the cleaned dataset
df = pd.read_csv('KorCCVi_v2_cleaned.csv')


# In[ ]:


sentences = df['transcript'].values
labels = df['label'].values


# ### Tokenization & Input Formatting Method 1

# In[ ]:


# Load the KoBERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
# tokenizer = BertTokenizer.from_pretrained('monologg/kobert')


# In[ ]:


# # Tokenize the input texts
# def convert_example_to_feature(review):
#     return tokenizer.encode_plus(review,
#                                  add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                                  max_length = None,           # 64, 512, None, max_length, Pad & truncate all sentences
#                                  # pad_to_max_length = True,
#                                  padding='max_length',        # True, longest, max_length, Pad & truncate all sentences
#                                  return_attention_mask = True,
#                                  return_token_type_ids = False,
#                                  truncation=True)


# In[ ]:


# # Map to the expected input to TFBertForSequenceClassification, see https://huggingface.co/transformers/model_doc/bert.html#tfbertforsequenceclassification
# input_ids = []
# attention_masks = []


# In[ ]:


# for sent in sentences:
#     encoded_dict = convert_example_to_feature(sent)
#     input_ids.append(encoded_dict['input_ids'])
#     attention_masks.append(encoded_dict['attention_mask'])


# In[ ]:


# # # Print the shapes of the data.
# print('Original: ', sentences[0])
# print('Token IDs:', input_ids[0])


# In[ ]:


# # Convert the lists into tensors.
# # input_ids = tf.convert_to_tensor(input_ids)
# # attention_masks = tf.convert_to_tensor(attention_masks)
# # labels = tf.convert_to_tensor(labels)
#
# input_ids = tf.constant(input_ids)
# attention_masks = tf.constant(attention_masks)
# labels = tf.constant(labels)


# In[ ]:


# input_ids


# In[ ]:


# labels


# In[ ]:


# attention_masks


# In[ ]:


# # split the dataset into train, validation and test set
# # train_ratio = 0.7
# # validation_ratio = 0.15
# # test_ratio = 0.15
# #
# # train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.2)
# # train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
# # train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.2)
# # Split data into train and validation sets
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
# train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)


# ### Split the dataset

# In[ ]:


# Split data into training and testing sets
# train_inputs, validation_inputs, train_inputs, validation_labels = train_test_split(sentences, labels, random_state=2018, test_size=0.2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)


# In[ ]:


X_train


# In[ ]:


# plot the class distribution of train and validation set
plot_class_distribution(pd.DataFrame(y_train, columns=['label']), 'Train Dataset Class Distribution')
plot_class_distribution(pd.DataFrame(y_test, columns=['label']), 'Validation Dataset Class Distribution')


# ### Tokenization & Input Formatting Method 2
# 

# In[ ]:


# Define hyperparameters
model_name = "monologg/kobert"
batch_size = 16 # 32, 64, 128
epochs = 1
# Lower learning rates are often better for fine-tuning transformers
learning_rate = 3e-5    #2e-5, 5e-5, 4e-5
weight_decay_rate=0.01


# In[ ]:


# Load KoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# #### Tokenize dataset NEW method (BARD)

# In[ ]:


# Preprocess training data
X_train_list = X_train.tolist()
train_inputs = tokenizer(X_train_list, padding="max_length", truncation=True, return_tensors="tf")


# In[ ]:


# Convert training labels to one-hot encoding
train_labels = to_categorical(y_train)


# In[ ]:


# Create TF training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs["input_ids"], train_labels)).batch(batch_size)


# In[ ]:


# Preprocess test data
X_test_list = X_train.tolist()
test_inputs = tokenizer(X_test_list, padding="max_length", truncation=True, return_tensors="tf")


# In[ ]:


# Convert test labels to one-hot encoding
test_labels = to_categorical(y_test)


# #### Training the KoBERT model

# In[ ]:


# Define optimizer, loss function, and metrics
optimizer = AdamWeightDecay(learning_rate=learning_rate)      # No loss argument!
# optimizer1 = AdamW(learning_rate=learning_rate)
optimizer2 = Adam(learning_rate)  # No loss argument!

loss_fn = SparseCategoricalCrossentropy(from_logits=True)

metrics = tf.keras.metrics.SparseCategoricalAccuracy()


# In[ ]:


# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define callbacks
# early_stopping = EarlyStopping(monitor="val_loss", patience=3)
# model_checkpoint = ModelCheckpoint("best_model.hdf5", monitor="val_accuracy", save_best_only=True)


# In[ ]:


# Load pretrained KoBERT model
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2, from_pt=True)


# In[ ]:


# Fine-tune the model
model.compile(optimizer=optimizer)


# In[ ]:


# get the model details
# get_model_details(model)


# In[ ]:


# get the model settings and parameters
# get_model_settings(model)


# In[ ]:


# Train the model
history = model.fit(train_dataset, epochs=epochs, validation_data=(test_inputs["input_ids"], test_labels))

# history = model.fit(train_dataset, epochs=epochs, validation_data=(test_inputs["input_ids"], test_labels))
# history = model.fit(train_dataset)


# #### Tokenize dataset OLD method

# In[ ]:


# # Tokenize all of the sentences and map the tokens to thier word IDs.
# # Map to the expected input to TFBertForSequenceClassification, see https://huggingface.co/transformers/model_doc/bert.html#tfbertforsequenceclassification
# train_input_ids = []
# train_attention_masks = []
#
# validation_input_ids = []
# validation_attention_masks = []
#
# # For every sentence...
# for train_sent, validation_sent in zip(train_inputs, validation_inputs):
#     # `encode_plus` will:
#     #   (1) Tokenize the sentence.
#     #   (2) Prepend the `[CLS]` token to the start.
#     #   (3) Append the `[SEP]` token to the end.
#     #   (4) Map tokens to their IDs.
#     #   (5) Pad or truncate the sentence to `max_length`
#     #   (6) Create attention masks for [PAD] tokens.
#     train_encoded_dict = tokenizer1.encode_plus(
#                         train_sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = 64,           # Pad & truncate all sentences.
#                         padding='max_length',
#                         return_attention_mask = True,   # Construct attn. masks.
#                         truncation=True,
#                         return_tensors = 'tf',     # Return tensorflow tensor.
#                    )
#     validation_encoded_dict = tokenizer1.encode_plus(
#                         validation_sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = 64,           # Pad & truncate all sentences.
#                         padding='max_length',
#                         return_attention_mask = True,   # Construct attn. masks.
#                         truncation=True,
#                         return_tensors = 'tf',     # Return tensorflow tensor.
#                    )
#
#     # Add the encoded sentence to the list.
#     train_input_ids.append(train_encoded_dict['input_ids'])
#     train_attention_masks.append(train_encoded_dict['attention_mask'])
#     validation_input_ids.append(validation_encoded_dict['input_ids'])
#     validation_attention_masks.append(validation_encoded_dict['attention_mask'])


# In[ ]:


# # Convert the lists into tensors.
# train_input_ids = tf.convert_to_tensor(train_input_ids)
# train_attention_masks = tf.convert_to_tensor(train_attention_masks)
# train_labels = tf.convert_to_tensor(train_labels)
#
# validation_input_ids = tf.convert_to_tensor(validation_input_ids)
# validation_attention_masks = tf.convert_to_tensor(validation_attention_masks)
# validation_labels = tf.convert_to_tensor(validation_labels)
#
# # Print sentence 0, now as a list of IDs.
# print('Original: ', train_inputs[0])
# print('Token IDs:', train_input_ids[0])


# In[ ]:


# # prepare the dataset for training
# train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_attention_masks, train_labels))
# validation_dataset = tf.data.Dataset.from_tensor_slices((validation_input_ids, validation_attention_masks, validation_labels))


# In[ ]:


# # shuffle the dataset
# train_dataset = train_dataset.shuffle(len(train_input_ids)).batch(32)
# validation_dataset = validation_dataset.shuffle(len(validation_input_ids)).batch(32)


# In[ ]:


# # print the shape of the dataset
# print(train_dataset)
# print(validation_dataset)


# 

# #### Train the korean BERT model

# In[ ]:


# #Function to get all the model settings and parameters
# def get_model_settings(model):
#     print('Model Name: ', model.name_or_path)
#     print('Model Type: ', model.__class__)
#     print('Model Parameters: ', model.config)
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_pretrained_dict())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_pretrained_dict())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_pretrained_dict())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_pretrained_dict())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())
#     # print('Model Parameters: ', model.config.to_json_string())
#     # print('Model Parameters: ', model.config.to_pretrained_dict())
#     # print('Model Parameters: ', model.config.to_yaml())
#     # print('Model Parameters: ', model.config.to_dict())
#     # print('Model Parameters: ', model.config.to_diff_dict())
#     # print('Model Parameters: ', model.config.to_json_file())


# In[ ]:


# # function to get the model detail after compiling
# def get_model_details(model):
#     print('Model Name: ', model.name)
#     print('Model Type: ', model.__class__)
#     print('Model Parameters: ', model.count_params())
#     print('Model Parameters: ', model.summary())
#     # print('Model Parameters: ', model.weights)
#     # print('Model Parameters: ', model.trainable_weights)
#     # print('Model Parameters: ', model.non_trainable_weights)
#     # print('Model Parameters: ', model.layers)
#     # print('Model Parameters: ', model.layers[0].name)
#     # print('Model Parameters: ', model.layers[0].trainable)
#     # print('Model Parameters: ', model.layers[0].count_params())
#     # print('Model Parameters: ', model.layers[0].input_shape)
#     # print('Model Parameters: ', model.layers[0].output_shape)
#     # print('Model Parameters: ', model.layers[0].get_weights())
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0])
#     # print('Model Parameters: ', model.layers[0].get_weights()[1])
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].size)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].size)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].nbytes)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].nbytes)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].itemsize)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].itemsize)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].dtype)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].dtype)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].ndim)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].ndim)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].flatten())
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].flatten().shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].flatten())
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].flatten().shape)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].flatten().size)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].flatten().size)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].flatten().nbytes)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].flatten().nbytes)
#     # print('Model Parameters: ', model.layers[0].get_weights()[0].flatten().itemsize)
#     # print('Model Parameters: ', model.layers[0].get_weights()[1].flatten().itemsize)


# In[ ]:


# # Load the pre-trained KoBERT model
# model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2, from_pt=True)


# In[ ]:


# model1 = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, from_pt=True)


# In[ ]:


# model0 = TFBertModel.from_pretrained(model_name, from_pt=True,num_labels=2)


# In[ ]:


# model2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# In[ ]:


# # Get the model settings and parameters
# get_model_settings(model)
# print('#'*100)


# In[ ]:


# # Get the model settings and parameters
# get_model_settings(model1)
# print('#'*100)


# In[ ]:


# # Get the model settings and parameters
# get_model_settings(model0)
# print('#'*100)


# In[ ]:


# # Get the model settings and parameters
# get_model_settings(model2)
# print('#'*100)


# In[ ]:


# compare all the models parameters
# model.config == model1.config == model0.config == model2.config


# In[ ]:


# # Define model hyperparameters
# batch_size = 32
# # epochs = 5
# # Lower learning rates are often better for fine-tuning transformers
# learning_rate = 3e-5    #2e-5, 5e-5, 4e-5
# weight_decay_rate=0.01
#
# # Define the optimizer
# optimizer = AdamWeightDecay(learning_rate=learning_rate)      # No loss argument!


# In[ ]:


# Compile the model
# model.compile(optimizer=optimizer)


# In[ ]:


# Get the model details
# get_model_details(model)


# In[ ]:


# # train the model
# history = model.fit(x={'input_ids': train_input_ids, 'attention_mask': train_attention_masks},
#                     y=train_labels,
#                     # batch_size=batch_size,
#                     # epochs=epochs,
#                     validation_data=({'input_ids': validation_input_ids, 'attention_mask': validation_attention_masks}, validation_labels))


# In[ ]:


# model.fit(tokenized_data, labels)


# In[ ]:


# # Fucntion to get all the model parameters and their shape and size in MB
# def get_model_memory_usage(batch_size, mymodel):
#     shapes_mem_count = 0
#     for p in mymodel.trainable_weights:
#         shapes_mem_count += np.prod(p.shape)
#     # shapes_mem_count *= batch_size
#     # shapes_mem_count *= 4  # fp32 bits/element
#     # shapes_mem_count /= 1024**2
#     # print('Model size (MB):', shapes_mem_count)
#     return shapes_mem_count


# In[ ]:


# #put the bellow in a function
# def mode_info(model):
#     model.config
#     # Getthe model weights
#     model.get_weights()
#     # Getthe model summary
#     model.summary()
#     # Getthe model layers
#     model.layers
#     # Getthe model input
#     model.input
#     # Getthe model output
#     model.output
#     # Getthe model loss
#     model.loss
#     # Getthe model metrics
#     model.metrics
#     # Getthe model optimizer
#     model.optimizer
#     # Getthe model sample weight mode
#     model.sample_weight_mode

