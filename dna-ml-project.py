print("DNA Machine Learning Project")
print("Python is working!")
dna_sequence = "ATGCGTACGTTAGC"

print(dna_sequence)
print(len(dna_sequence))
print(dna_sequence[0])
print(dna_sequence[1])
print(dna_sequence[2])
print(dna_sequence[-1])
import pandas as pd
import numpy as np
print("libraries working")
import urllib.request

files = {
    "human_data.txt": "https://raw.githubusercontent.com/nageshsinghc4/DNA-Sequence-Machine-learning/master/human_data.txt",
    "chimp_data.txt": "https://raw.githubusercontent.com/nageshsinghc4/DNA-Sequence-Machine-learning/master/chimp_data.txt",
    "dog_data.txt": "https://raw.githubusercontent.com/nageshsinghc4/DNA-Sequence-Machine-learning/master/dog_data.txt"
}

for filename, url in files.items():
    urllib.request.urlretrieve(url, filename)
    print(filename, "downloaded")

print("All datasets downloaded!")
human_data = pd.read_csv("human_data.txt", sep="\t")
chimp_data = pd.read_csv("chimp_data.txt", sep="\t")
dog_data = pd.read_csv("dog_data.txt", sep="\t")

print("Human:", human_data.shape)
print("Chimp:", chimp_data.shape)
print("Dog:", dog_data.shape)

print(human_data.head())
print("Human columns:", human_data.columns)
print("Chimp columns:", chimp_data.columns)
print("Dog columns:", dog_data.columns)
X = human_data["sequence"]
y = human_data["class"]

print("X:", X.shape)
print("y:", y.shape)

print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
def get_kmers(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

X_train_kmers = X_train.apply(get_kmers)
X_test_kmers = X_test.apply(get_kmers)

print(X_train_kmers.head())
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train_kmers)
X_test_vectorized = vectorizer.transform(X_test_kmers)

print("Training matrix:", X_train_vectorized.shape)
print("Testing matrix:", X_test_vectorized.shape)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

print("Predictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")
chimp_kmers = chimp_data["sequence"].apply(get_kmers)
dog_kmers = dog_data["sequence"].apply(get_kmers)

chimp_vectorized = vectorizer.transform(chimp_kmers)
dog_vectorized = vectorizer.transform(dog_kmers)

chimp_pred = model.predict(chimp_vectorized)
dog_pred = model.predict(dog_vectorized)

chimp_accuracy = accuracy_score(chimp_data["class"], chimp_pred)
dog_accuracy = accuracy_score(dog_data["class"], dog_pred)

print("Chimp Accuracy:", chimp_accuracy * 100, "%")
print("Dog Accuracy:", dog_accuracy * 100, "%")
from sklearn.metrics import classification_report

print("\nHuman Classification Report:")
print(classification_report(y_test, y_pred))

print("\nChimp Classification Report:")
print(classification_report(chimp_data["class"], chimp_pred))

print("\nDog Classification Report:")
print(classification_report(dog_data["class"], dog_pred))
def predict_gene_class(sequence):
    kmers = get_kmers(sequence)
    vectorized = vectorizer.transform([kmers])
    prediction = model.predict(vectorized)
    return prediction[0]

new_sequence = "ATGCGTACGTTAGC"

result = predict_gene_class(new_sequence)

print("\nNew DNA Sequence:", new_sequence)
print("Predicted Class:", result)
print("\n========== FINAL RESULTS ==========")

print("Human Accuracy :", round(accuracy * 100, 2), "%")
print("Chimp Accuracy :", round(chimp_accuracy * 100, 2), "%")
print("Dog Accuracy   :", round(dog_accuracy * 100, 2), "%")

print("==================================")
print("DNA Classification Project Completed!")