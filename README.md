# 💬 Twitter Sentiment Analyzer

A Streamlit web app that analyzes the sentiment of tweets using a fine-tuned BERT model and verifies them using *Gemini Pro AI*.

### 🚀 Features
- Analyze individual tweets or tweets from a username
- Sentiment prediction using BERT
- Gemini-powered sentiment verification
- Timeline & Pie chart visualizations
- Gemini-generated summary of tweet sentiments
- Built with Selenium, Transformers, Streamlit & Gemini AI

### 🔧 How to Run
1. Clone the repository:

git clone https://github.com/Lavanya-Parashar/Twitter_Sentiment_Analyzer.git

2. Install dependencies:

pip install -r requirements.txt

3. Add your API key in .streamlit/secrets.toml (keep this private):
   ```toml
GEMINI_API_KEY = "your_api_key_here"

5. Place your model.pth file in the project folder (not uploaded due to size limits).
6.	Run the app:

streamlit run seleniumnew_app.py

⸻
📌 Note: model.pth is not included in this repo due to GitHub’s file size limit. 
