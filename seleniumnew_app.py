import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="twitter_bird_icon.svg")
st.title("💬 Twitter Sentiment Analyzer")
st.markdown("Analyze tweet sentiments using *BERT + Gemini Pro* ⚡")

# ----------------- Device Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Model -----------------
# ----------------- Load Model (Fixed for Meta Tensor Error) -----------------
from transformers import BertForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load only architecture (model stays empty)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Step 2: Load your fine-tuned weights
state_dict = torch.load("model.pth", map_location=device)

# Step 3: Load weights into model — strict=False helps skip meta tensor mismatch
model.load_state_dict(state_dict, strict=False)

# Step 4: Now send to device safely
model.to(torch.device("cpu") if device.type == "cpu" else "cuda")

model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ----------------- Gemini Setup -----------------
import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def gemini_sentiment_check(text):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = f"What is the sentiment of this tweet? Respond with Positive or Negative only:\n\n'{text}'"
        response = model.generate_content(prompt)
        return response.text.strip().lower()
    except Exception as e:
        return f"error: {e}"

# ----------------- Prediction Function -----------------
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()

        labels = {0: "Negative", 1: "Positive"}
        sentiment = labels[prediction]

        gemini_response = gemini_sentiment_check(text)
        if gemini_response in ['positive', 'negative']:
            if gemini_response == sentiment.lower():
                source = "✅ Verified by Gemini"
            else:
                sentiment = gemini_response.capitalize()
                source = "🔄 Overridden by Gemini"
        else:
            source = "🔌 Gemini Unavailable – Used BERT"

        return sentiment, source
    except Exception as e:
        return "Error", f"⚠ Prediction failed: {e}"

# ----------------- Emoji Mapping -----------------
def get_emoji(sentiment):
    return "😊" if sentiment == "Positive" else "😠"

# ----------------- Fetch Tweets using Selenium -----------------
def get_tweets_from_user(username, count=10):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        driver.get(f"https://twitter.com/{username}")
        time.sleep(5)

        # Scroll once to load more tweets
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)  # wait for content to load

        tweets = []
        elements = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')
        for el in elements[:count]:
            tweets.append({"text": el.text, "time": str(datetime.datetime.now())})

        driver.quit()
        return tweets
    except Exception as e:
        print("Selenium Error:", e)
        return []

# ----------------- Pie Chart -----------------
def plot_pie_chart(results):
    labels = list(results.keys())
    sizes = list(results.values())
    colors = ["#66FF66", "#FF6666"]  # Positive = Green, Negative = Red
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    st.pyplot(plt)

# ----------------- Timeline Graph -----------------
def plot_timeline(tweets):
    times = [datetime.datetime.fromisoformat(t["time"].replace("Z", "+00:00")) for t in tweets]
    sentiments = []
    for t in tweets:
        sentiment, _ = predict_sentiment(t["text"])
        sentiments.append(1 if sentiment == "Positive" else -1)

    df = pd.DataFrame({"Time": times, "Sentiment": sentiments})
    df.sort_values("Time", inplace=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df["Time"], df["Sentiment"], marker='o', linestyle='-', color='purple')
    plt.yticks([-1, 1], ["Negative 😠", "Positive 😊"])
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.title("📈 Sentiment Timeline")
    plt.tight_layout()
    st.pyplot(plt)

# ----------------- Gemini Summary -----------------
def gemini_summary(all_texts):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        short_texts = all_texts[:6]  # Just the first 6 tweets
        prompt = (
            "Summarize the overall sentiment tone of the following tweets "
            "in 2-3 lines. Just mention whether they are mostly positive or negative, and highlight any common patterns:\n\n"
            + "\n".join(f"- {tweet}" for tweet in short_texts)
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini summary could not be generated.\n\nError: {e}"


# ----------------- Streamlit UI -----------------
mode = st.radio("Choose input method:", ["Enter Tweet", "Analyze Twitter Username"])

if mode == "Enter Tweet":
    user_tweet = st.text_area("Enter your tweet here:")
    if st.button("Analyze Sentiment"):
        sentiment, source = predict_sentiment(user_tweet)
        st.success(f"*Sentiment:* {sentiment} {get_emoji(sentiment)}")
        st.markdown(f"{source}")

elif mode == "Analyze Twitter Username":
    username = st.text_input("Enter Twitter username (without @):")
    if st.button("Fetch and Analyze"):
        with st.spinner("Fetching tweets using Selenium..."):
            tweets = get_tweets_from_user(username)

        if not tweets:
            st.error("❌ Couldn't fetch tweets or user not found.")
        else:
            results = {"Positive": 0, "Negative": 0}
            all_texts = []
            for t in tweets:
                sent, source = predict_sentiment(t["text"])
                results[sent] += 1
                all_texts.append(t["text"])
                st.markdown(f"📝 Tweet:** {t['text']}")
                st.markdown(f"*Sentiment:* {sent} {get_emoji(sent)}")
                st.markdown(f"{source}")
                st.markdown("---")

            st.subheader("📊 Sentiment Distribution")
            plot_pie_chart(results)

            st.subheader("📈 Sentiment Timeline")
            plot_timeline(tweets)

            st.subheader("🧠 Gemini Summary")
            st.info(gemini_summary(all_texts))

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("<center>🚀 Made with ❤️ using Streamlit, BERT, Gemini & Selenium</center>", unsafe_allow_html=True)