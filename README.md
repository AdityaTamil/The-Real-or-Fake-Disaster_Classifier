# Classic Machine Learning Approach

For the classical model, a traditional machine learning classifier was implemented using TF-IDF features.

## Text Cleaning

The same text cleaning pipeline was applied before training the classical model:

Converted text to lowercase

Replaced user mentions with placeholders

Removed URLs and punctuation

Removed stopwords while keeping important negations

This ensured fair comparison with the deep learning approach.

## Feature Extraction

Cleaned tweets were converted into numerical vectors using TF-IDF.

TF-IDF helps capture the importance of words while reducing the impact of very common terms.

The same vectorizer was used for both training and validation data.

## Dataset Preparation

The dataset was split into training and validation sets.

TF-IDF features were generated only from the training data.

Validation data was transformed using the trained vectorizer.

## Model Architecture

Two classical models were trained:

Naive Bayes

Logistic Regression

Logistic Regression was chosen as the final classical baseline due to better performance.

## Training

Models were trained on TF-IDF features.

Default hyperparameters were used to keep the baseline simple.

No automated tuning or AutoML techniques were applied.

## Evaluation

The classical models were evaluated using:

F1-score

Classification report

Confusion matrix

Confusion matrices were used to analyze false positives and false negatives.

## Submission

The trained Logistic Regression model was used to predict labels for the test dataset.

Predictions were saved in the required Kaggle CSV format:

id,target


The file was submitted for evaluation.

## Summary

The classical approach provided a strong and interpretable baseline with relatively low training cost.
Although it performed slightly worse than the LSTM model, it was simpler to train and easier to analyze.


# Deep Learning Approach (LSTM)

For the deep learning model, an LSTM-based classifier was implemented using PyTorch.

## Text Cleaning

The same cleaning pipeline used in the classical approach was reused:

Converted text to lowercase

Replaced user mentions with placeholders

Removed URLs and punctuation

Removed stopwords while keeping important negations

This ensured consistency across models.

## Vocabulary and Encoding

A vocabulary was built from the training data using word frequency.

Two special tokens were added: <PAD> and <UNK>.

Each tweet was converted into a sequence of word indices.

All sequences were padded or truncated to a fixed length of 30 tokens.

## Dataset and DataLoader

A custom PyTorch Dataset class was created to handle:

Encoded tweet sequences

An additional binary feature indicating the presence of disaster-related words

Target labels

DataLoaders were used to batch and shuffle the data during training.

## Model Architecture

The neural network consists of:

An Embedding layer (learned from scratch)

A single LSTM layer

A fully connected layer

The output of the LSTM is concatenated with a simple auxiliary feature (has_disaster_word) before the final prediction layer.

## Training

Loss Function: Binary Cross-Entropy with logits

Optimizer: Adam

Class imbalance was handled using a positive class weight

The model was trained for 5 epochs

Training loss was printed after each epoch to monitor learning.

## Evaluation

Predictions were generated using a sigmoid activation

A probability threshold was manually chosen based on validation performance

Evaluation metrics used:

F1-score

Classification report

Confusion matrix (visualized using a heatmap)

The confusion matrix helped analyze where the model made incorrect predictions.

##Submission

The trained model was used to predict labels for the test dataset

Predictions were saved in the required Kaggle CSV format:

id,target


The file was submitted for evaluation

## Summary

The LSTM model was able to capture sequential patterns in text and performed better than simple rule-based methods.
However, it required more preprocessing, training time, and tuning compared to the classical baseline.
