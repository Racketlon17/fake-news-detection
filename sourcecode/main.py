import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df_Fake = pd.read_csv("data/Fake.csv")
df_True = pd.read_csv("data/True.csv")

df_True["label"] = 1
df_Fake["label"] = 0
df = pd.concat([df_True, df_Fake], ignore_index = True)

df = df.drop(['title','date','subject'], axis=1)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    words = text.split()
    words = [word for word in words if words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)
vectorizer = TfidfVectorizer(max_df=0.7)
x = vectorizer.fit_transform(df["text"])
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)

joblib.dump(model,"model/fake_news_detection_model.plk")
joblib.dump(vectorizer, "vectorizer/tfidf_vectorizer.pkl")