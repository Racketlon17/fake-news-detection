Installation

Clone the repository
clone [<repository-url>](https://github.com/Racketlon17/fake-news-detection.git)
cd fake-news-detection

Create virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Download NLTK data
import nltk
nltk.download('stopwords')


Dataset
The project uses two CSV files:

data/Fake.csv - Contains fake news articles
data/True.csv - Contains real news articles

Each dataset should have columns:

title - Article title
text - Article content
subject - Article category
date - Publication date

Usage
Web Application

Start the Streamlit app
streamlit run main.py

Navigate to the application

Open your browser and go to http://localhost:8501


Use the features:

Predict News: Enter text and get real-time classification
Train Model: Train a new model with your data