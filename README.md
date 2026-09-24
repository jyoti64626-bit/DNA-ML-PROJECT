# DNA Sequence Machine Learning

## Project Overview

This project uses Machine Learning to classify DNA sequences into different gene classes.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes

## Datasets

The project uses DNA sequence datasets from:

- Human
- Chimpanzee
- Dog

## Machine Learning Process

1. Load DNA sequence data
2. Split the data into training and testing sets
3. Convert DNA sequences into k-mers
4. Convert k-mers into numerical features
5. Train a Multinomial Naive Bayes model
6. Evaluate the model
7. Predict the class of a new DNA sequence

## Files

- app.py — Streamlit web application
- dna_ml_project.py — Main Python machine learning program
- human_data.txt — Human DNA dataset
- chimp_data.txt — Chimpanzee DNA dataset
- dog_data.txt — Dog DNA dataset
- requirements.txt — Required Python libraries
## Deployment

The application is deployed using Streamlit Community Cloud.

Users can enter a DNA sequence and get a predicted gene class through the web application.

## Disclaimer

This is an educational machine learning project and is not intended for medical diagnosis or clinical use.
