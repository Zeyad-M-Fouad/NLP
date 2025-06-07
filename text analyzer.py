#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# In[2]:


data ="C:/Users/Zeyad/Desktop/nlp/analyzer/sentiment analyzer/sentiment_analysis.csv"
df = pd.read_csv(data)
print(df)


# In[3]:


df.drop(['Year','Month','Day','Time of Tweet','Platform'], axis=1, inplace=True)
print (df)
print(type(df))


# In[4]:


feature_cols = ['text']
x = df[feature_cols]
y = df.sentiment


# In[5]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
x = cv.fit_transform(df['text'])


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(x , y,test_size=0.10, random_state=35)


# In[7]:


MNB = MultinomialNB()
MNB.fit(X_train, Y_train)
#print(X_train, Y_train)


# In[8]:


predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracy Score :", accuracy_score)


# In[9]:


print (predicted)


# In[10]:


sentence = ["i hate you"]
sentence_transformed = cv.transform(sentence)
#float() = sentence 
#print (sentence)
prediction = MNB.predict(sentence_transformed)
print('prediction : ' ,prediction)


# In[11]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
print("Accuracy score : ",model.score(X_test, Y_test))


# In[12]:


sentence = ["the sun is rising"]
sentence_transformed = cv.transform(sentence)
prediction = model.predict(sentence_transformed)
print('prediction : ' ,prediction)


# In[13]:


class SentimentAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Sentiment Analyzer")
        self.text_input = scrolledtext.ScrolledText(master, width=60, height=10)
        self.text_input.pack()

        self.analyze_button = tk.Button(master, text="Analyze", command=self.analyze_text)
        self.analyze_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

 # Load data
        data_path = "C:/Users/Zeyad/Desktop/nlp/analyzer/sentiment analyzer/sentiment_analysis.csv"
        self.df = pd.read_csv(data_path)
        self.df.drop(['Year', 'Month', 'Day', 'Time of Tweet', 'Platform'], axis=1, inplace=True)
        self.feature_cols = ['text']
        self.x = self.df[self.feature_cols]
        self.y = self.df.sentiment
        

        # Tokenizer
        self.token = RegexpTokenizer(r'[a-zA-Z0-9]+')

        # CountVectorizer
        self.cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=self.token.tokenize)
        self.x = self.cv.fit_transform(self.df['text'])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.x, self.y, test_size=0.10, random_state=40)

        # Multinomial Naive Bayes model
        self.MNB = MultinomialNB()
        self.MNB.fit(self.X_train, self.Y_train)

        # Support Vector Machine model
        self.model = SVC()
        self.model.fit(self.X_train, self.Y_train)

    def analyze_text(self):
        input_text = self.text_input.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showerror("Error", "Please enter some text.")
            return
        sentence = [input_text]
        sentence_transformed = self.cv.transform(sentence)

        # Multinomial Naive Bayes prediction
        mnb_prediction = self.MNB.predict(sentence_transformed)

        # Support Vector Machine prediction
        svm_prediction = self.model.predict(sentence_transformed)

        self.result_label.config(text=f"MNB Prediction: {mnb_prediction[0]}\nSVM Prediction: {svm_prediction[0]}")


# In[14]:


def main():
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




