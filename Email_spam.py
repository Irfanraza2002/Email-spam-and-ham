import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
#inno=Image.open("Inno.png")
#st.image(inno, use_column_width=True)
email_logo=Image.open(r"C:\Users\moham\Downloads\email.jpeg")
st.image(email_logo, caption='Email', use_column_width=False)
# Load the pre-trained model and vectorizer
model_path = r"C:\Users\moham\machine learning\Email_spam.pkl"
vectorizer_path = r"C:\Users\moham\machine learning\tf.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

st.title("Email Spam Classification")


# Input email from user
email = st.text_input("Enter the email: ")

# Transform the input email using the loaded vectorizer
if st.button("Submit"):
    data = tfidf.transform([email]).toarray()
    pred = model.predict(data)[0]
    
    if pred == 'spam':
        st.write("The Email is: Spam")
        spam_image = Image.open(r"C:\Users\moham\Downloads\spam.jpeg")
        st.image(spam_image, caption='Spam Email', use_column_width=False)
    else:
        st.write("The Email is: Ham")
        ham_image = Image.open(r"C:\Users\moham\Downloads\ham.png")
        st.image(ham_image, caption='Ham Email', use_column_width=False)