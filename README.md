# Chronic Depression Detection using Deep Learning

## Overview
Depression Detection is a controversial aspect of psychologist for a really long time ; Here we use Machine Learning Algorithms to find patterns in the texts posted on *Reddit*


## Text Classification Algorithms

**Rule-based System**

In the early days of text classification, rule-based systems were prevalent. These systems relied on handcrafted rules to classify text based on patterns and keywords.

*Limitations :* Limited scalability and adaptability as rules needed to be manually curated and updated.

**Statistical Methods**

Statistical methods, such as Naive Bayes, gained popularity due to their simplicity and effectiveness. These algorithms calculate probabilities of a document belonging to a particular class based on word frequencies.

*Advantages :* Relatively fast and easy to implement, especially for binary classification tasks.

*Limitations :* Assumes independence between features, which might not hold true for real-world text data.

**Machine Learning Models**

Machine learning models like Support Vector Machines (SVM) and Logistic Regression have been widely used for text classification. These models learn to classify text by optimizing a specified objective function.

*Advancements :* Various enhancements and adaptations have been made to traditional machine learning algorithms to improve performance, such as using different kernel functions in SVM or incorporating feature engineering techniques.

*Limitations:* Performance highly dependent on feature engineering and selection.

**Deep Learning**

Deep learning, especially with neural network architectures, has revolutionized text classification. Models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are capable of automatically learning hierarchical and contextual representations of text data.

*Advantages :* Can capture complex patterns and dependencies in text data, reducing the need for manual feature engineering.
State-of-the-art: Transformer-based architectures like BERT, GPT, and their variants have achieved remarkable performance in text classification tasks, leveraging attention mechanisms for contextual understanding.

*Limitations :* Requires large amounts of labeled data and computational resources for training.

*In this project , we will use RNNs and specially LSTM layers*
## LSTM ( Long-Short Term Memory )

LSTM is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem in traditional RNNs, which struggle to learn and retain information over long sequences

**Memory Cells :** 
LSTM networks contain memory cells that allow them to store information over time. These memory cells can maintain information for long durations, making LSTMs suitable for processing sequences with long-term dependencies.

**Gates :**
LSTMs utilize three types of gates: input gate, forget gate, and output gate.

*Input Gate :* Controls the flow of new information into the memory cell.

*Forget Gate :* Modulates the retention of information in the memory cell.

*Output Gate:* Regulates the information flow from the memory cell to the output.

**Learnable Parameters :**
LSTMs have learnable parameters that adaptively control the flow of information through the network. These parameters are optimized during the training process to capture meaningful patterns in the data.

**Advantages :**
Long-Term Dependencies: LSTMs excel at capturing dependencies in sequences that span long distances, making them effective for tasks such as natural language processing (NLP), time series prediction, and speech recognition.

**Gradient Flow :** The architecture of LSTMs enables better flow of gradients during backpropagation, mitigating the vanishing gradient problem encountered in traditional RNNs. This allows LSTMs to learn from sequences of arbitrary length more effectively.

**Versatility :** LSTMs can be adapted and extended for various sequential data tasks, including sequence classification, sequence generation, and sequence-to-sequence learning.

**Applications :**

*Natural Language Processing (NLP) :* LSTMs are widely used in NLP tasks such as sentiment analysis, text classification, machine translation, and named entity recognition.

*Time Series Prediction :* LSTMs are effective for modeling and forecasting time series data in domains such as finance, weather forecasting, and energy consumption prediction.

*Speech Recognition :* LSTMs have been applied to speech recognition systems, where they help in understanding and transcribing spoken language with high accuracy.
## Data

We use the *50000* posts on *Reddit* where posted between 2009 to 2021 . The original dataset is on Kaggle . For better descriptions chack the this [Link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)



Data has two parts , text and class . text is the original post collected from reddit and class is a binary number 0 or 1 ; 1 means detected as chronic depression and 0 means opposite 
## Model Architecture

The model architecture consists of the following layers:

[**Embedding Layer**](https://www.dremio.com/wiki/embedding-layer/#:~:text=What%20is%20Embedding%20Layer%3F,between%20different%20categories%20or%20classes.) : Maps input words to dense vector space.

[**Bidirectional LSTM Layers**](https://colah.github.io/posts/2015-08-Understanding-LSTMs/): Processes input sequences in both forward and backward directions.

[**Dropout Layers**](https://keras.io/api/layers/regularization_layers/dropout/#:~:text=The%20Dropout%20layer%20randomly%20sets,over%20all%20inputs%20is%20unchanged.) : Regularization technique to prevent overfitting.

[**Dense Layers**](https://keras.io/api/layers/core_layers/dense/) : Fully connected layers with various activation functions.
## Related Projects

Here are some related text classification projects in my profile 

* [IMDB Reviews classifier](https://github.com/mohammad0081/IMDB_reviews_classifier)

