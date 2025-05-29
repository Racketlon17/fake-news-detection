# Fake News Detection

## Installation

### Clone the repository
```bash
git clone https://github.com/Racketlon17/fake-news-detection.git
cd fake-news-detection
```
Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
Download NLTK data
```python
import nltk
nltk.download('stopwords')
```

### Usage
Start the streamlit app
```bash
streamlit run main.py
```
Open your browser and go to
http://localhost:8501

