"""
Natural Language Processing
We will analyze reviews to det whether +ve or -ve review

"""
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #ignore double quotes

#Cleaning the text
import re
import nltk
nltk.download('stopwords')                              #words like this, that etc
from nltk.corpus import stopwords                       #corpus is a collection of text of same type
from nltk.stem.porter import PorterStemmer              #Stemming ex: loved == love etc

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #Remove all expect words and replace with space
    review = review.lower()                             #To lower case
    review = review.split()                             #split string into list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #Set will make algo faster (splitting and stemming)
    review = ' '.join(review)                           #list to string
    corpus.append(review)    
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)               #can clean data using params, however manual method gives more options. max_feat => most freq
X = cv.fit_transform(corpus).toarray()                  #indedepdant variables
y = dataset.iloc[:, 1].values                           #depedant variable

#Apply classification model. Most commonly used are Naive Bayes, DT and Random Forest. We will use Naive Bayes
#Cannot use viz as there are more than 2 dimensions
#Feat scaling not used because most values are 1 or 0
#Keep in mind that we have only 1000 test cases (instead of 1M), so accuracy won't be very high

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
