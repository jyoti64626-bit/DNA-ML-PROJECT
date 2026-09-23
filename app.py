import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="DNA Sequence Classifier",
    page_icon="🧬"
)

st.title("🧬 DNA Sequence Classifier")
st.write("Enter a DNA sequence and the machine-learning model will predict its gene class.")


@st.cache_resource
def train_model():
    human_data = pd.read_csv("human_data.txt", sep="\t")

    X = human_data["sequence"]
    y = human_data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    def get_kmers(sequence, k=6):
        sequence = str(sequence).upper().strip()
        return " ".join(
            sequence[i:i+k]
            for i in range(len(sequence) - k + 1)
        )

    X_train_kmers = X_train.apply(get_kmers)

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train_kmers)

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer, get_kmers


model, vectorizer, get_kmers = train_model()


st.subheader("Enter DNA Sequence")

sequence = st.text_area(
    "DNA sequence:",
    placeholder="Example: ATGCGTACGTTAGC"
)

if st.button("🔬 Predict"):
    if not sequence.strip():
        st.warning("Please enter a DNA sequence.")
    else:
        sequence = sequence.upper().strip()

        invalid = set(sequence) - set("ATGC")

        if invalid:
            st.error("Please enter a valid DNA sequence using only A, T, G and C.")
        else:
            kmers = get_kmers(sequence)
            sequence_vectorized = vectorizer.transform([kmers])

            prediction = model.predict(sequence_vectorized)[0]

            st.success(f"Predicted Gene Class: {prediction}")