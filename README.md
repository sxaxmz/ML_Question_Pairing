# ML_Question_Pairing
A repository that contains ML algorithms and processing on a question-paring dataset from Kaggle (Text Processing, Categorical Encoding, Advanced Feature Engineering, Binary Classification).

Used packages:

| Seaborn | Matoplotlib | Pandas | Math    |
|---------|-------------|--------|---------|
| Numpy   | Sklearn     | os     | mlxtend |
| nltk    | Collections | Scipy  | Plotly  |
| worldcloud    | tqdm | Sqlite3  | xgboost  |
| datetime    | sqlalchemy | re  | fuzzywuzzy  |
| bs4    |  |   |   |

Built Classifiers:
* Logistic Regression (LR)
* Support Vector Machine (SVM)

Files:

| File Name       | Description                                                                                                                                                                                                       |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| explore.py | Explore the downloaded files and perform initial preprocessing such as cleaning, preprocessing text, and merging files.                                                                                                |
| text_process.py | Perform advanced preprocessing and feature engineering features such as common-word-count (CWC) and common-stop-count (CSC) with their ratios utilizing wordcloud and fuzzywuzzy packages which help in text matching and similarity.                                                        |
| word2vec.py  | Take the training set of the data and vectorize the text using spacy GLOVE model and TFIDDVector. |
| model_.py  | Take the vectorized store it in a SQLite DB and train the classifiers  while keeping in mind the model's interpretability and its predictive power. |
| functions_.py   | A file that acts as a central file for main functions used during the text preprocessing stage such as preprocessing and tokenization.                                                                  |


---

Dataset link on Kaggle: [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/)

The data that was preprocessed and trained belongs to the following file:
* train.csv.zip (ID, qid1, qid2, question2, question2, is_duplicated)

Contains the below information.

| Field Name | Description                                                              |
|------------|--------------------------------------------------------------------------|
| ID         | The id of a training set question pair |
| qid1, qid2       | Unique ids of each question (only available in train.csv)                         |
| question1, question2  | The full text of each question                              |
| is_duplicate      | The  target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.|                                      |


---

While processing the data and training the model an approach was followed comparing the model's performance using multiple alpha's to help determine the best alpha (learning rate) to train the classifier on. To choose the best alpha hyperparameter for the model's training, various alphas were tested on the model, and the one that presented the lowest error measure (log loss) was selected. The below is the best alpha used for each of the models.

| Model                                             | Best Alpha |
|---------------------------------------------------|------------|
| LR    | 1      |
| SVM  (Linear)               | 0.0001       |


Log Loss comparison across train and test sets.

| Model                                             | Train Set Loss | Test Set Loss | Number of Data Points |
|---------------------------------------------------|----------------|---------------------------|---------------|
| LR    | 0.5138           | 0.5200                     | 30000          |
| SVM (Linear)                   | 0.4780           | 0.4896                      | 30000          |
| XGBoost                      | 0.3455           | 0.3570                      | 30000         |

It can be noted that the XGBoost model has the best performance in terms of the lowest error recorded (log loss) in both the training 34.55 (0.3455 x 100) and testing sets 35.70 (0.3570 x100) in comparison to other models used.
