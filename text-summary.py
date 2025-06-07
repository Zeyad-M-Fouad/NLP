#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext, messagebox


# In[2]:


data ="C:/Users/Zeyad/Desktop/nlp/nlp summer/samsum-test.csv"
df = pd.read_csv(data)
print(df)


# In[3]:


def split_data(df):
    X = df['dialogue']  # Features
    y = df['summary']   # Target labels
    return X, y

# Split the data
X, y = split_data(df)

print("Features (X):")
print(X.head())
print("\nTarget labels (y):")
print(y.head())


# In[4]:


# Tokenize text into sentences
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def tokenize_text(text):
    return sent_tokenizer.tokenize(text)


# In[5]:


# Compute sentence importance using TF-IDF
def compute_sentence_importance(sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = similarity_matrix.sum(axis=1)
    return scores

# Summarization function
def generate_summary(documents):
    # Convert documents into sentences
    sentences = [sentence for doc in documents for sentence in tokenize_text(doc)]

    # Compute importance scores for each sentence
    sentence_scores = compute_sentence_importance(sentences)

    # Sort sentences by importance score in descending order
    sorted_sentences = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)

    # Select top sentences to form the summary
    num_sentences = min(len(sentences), 3)  # Adjust this value for desired summary length
    summary_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
    
    # Join summary sentences into a single string
    summary = ' '.join(summary_sentences)
    
    return summary

# Compute similarity between two texts based on TF-IDF
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]


# In[6]:


class SummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarizer")

        self.text_input = scrolledtext.ScrolledText(master, width=60, height=10)
        self.text_input.pack()

        self.summarize_button = tk.Button(master, text="Summarize", command=self.summarize)
        self.summarize_button.pack()

        self.summary_output = scrolledtext.ScrolledText(master, width=60, height=10)
        self.summary_output.pack()

    def summarize(self):
        input_text = self.text_input.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showerror("Error", "Please enter some text.")
            return

        # Generate summary
        summary = generate_summary([input_text])

        # Compute similarity
        similarity_score = compute_similarity(summary, input_text)

        # Display summary and similarity score
        self.summary_output.delete("1.0", tk.END)
        self.summary_output.insert(tk.END, "Generated Summary:\n")
        self.summary_output.insert(tk.END, summary + "\n\n")
        self.summary_output.insert(tk.END, f"Similarity score: {similarity_score:.2f}")


# In[7]:


def main():
    root = tk.Tk()
    app = SummarizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:


import tkinter as tk
from tkinter import scrolledtext, messagebox
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tokenize text into sentences
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def tokenize_text(text):
    return sent_tokenizer.tokenize(text)

# Compute sentence importance using TF-IDF
def compute_sentence_importance(sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = similarity_matrix.sum(axis=1)
    return scores

# Summarization function
def generate_summary(documents):
    # Convert documents into sentences
    sentences = [sentence for doc in documents for sentence in tokenize_text(doc)]

    # Compute importance scores for each sentence
    sentence_scores = compute_sentence_importance(sentences)

    # Sort sentences by importance score in descending order
    sorted_sentences = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)

    # Select top sentences to form the summary
    num_sentences = min(len(sentences), 3)  # Adjust this value for desired summary length
    summary_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
    
    # Join summary sentences into a single string
    summary = ' '.join(summary_sentences)
    
    return summary

# Compute similarity between two texts based on TF-IDF
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

class SummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarizer")

        self.text_input = scrolledtext.ScrolledText(master, width=60, height=10)
        self.text_input.pack()

        self.summarize_button = tk.Button(master, text="Summarize", command=self.summarize)
        self.summarize_button.pack()

        self.summary_output = scrolledtext.ScrolledText(master, width=60, height=10)
        self.summary_output.pack()

    def summarize(self):
        input_text = self.text_input.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showerror("Error", "Please enter some text.")
            return

        # Generate summary
        summary = generate_summary([input_text])

        # Compute similarity
        similarity_score = compute_similarity(summary, input_text)

        # Display summary and similarity score
        self.summary_output.delete("1.0", tk.END)
        self.summary_output.insert(tk.END, "Generated Summary:\n")
        self.summary_output.insert(tk.END, summary + "\n\n")
        self.summary_output.insert(tk.END, f"Similarity score: {similarity_score:.2f}")

def main():
    root = tk.Tk()
    app = SummarizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:


import tkinter as tk
from tkinter import scrolledtext, messagebox
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

class TextSummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarizer")

        # Create input field for text
        self.create_text_input()

        # Create button to trigger summarization
        self.create_summarize_button()

        # Create output field for summary
        self.create_summary_output()

    def create_text_input(self):
        self.text_input = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.text_input.pack()

    def create_summarize_button(self):
        self.summarize_button = tk.Button(self.master, text="Summarize", command=self.summarize)
        self.summarize_button.pack()

    def create_summary_output(self):
        self.summary_output = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.summary_output.pack()

    def tokenize_text(self, text):
        """Tokenize text into sentences."""
        return nltk.sent_tokenize(text)

    def compute_sentence_importance(self, sentences):
        """Compute sentence importance using TF-IDF."""
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        scores = similarity_matrix.sum(axis=1)
        return scores

    def generate_summary(self, documents):
        """Generate summary from input text."""
        sentences = [sentence for doc in documents for sentence in self.tokenize_text(doc)]
        sentence_scores = self.compute_sentence_importance(sentences)
        sorted_sentences = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)
        num_sentences = min(len(sentences), 3)
        summary_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
        summary = ' '.join(summary_sentences)
        return summary

    def compute_similarity(self, text1, text2):
        """Compute similarity between two texts based on TF-IDF."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return similarity[0][0]

    def summarize(self):
        """Summarize the input text and display results."""
        input_text = self.text_input.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showerror("Error", "Please enter some text.")
            return

        # Generate summary
        summary = self.generate_summary([input_text])

        # Compute similarity
        similarity_score = self.compute_similarity(summary, input_text)

        # Display summary and similarity score
        self.summary_output.delete("1.0", tk.END)
        self.summary_output.insert(tk.END, "Generated Summary:\n")
        self.summary_output.insert(tk.END, summary + "\n\n")
        self.summary_output.insert(tk.END, f"Similarity score: {similarity_score:.2f}")

def main():
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:




