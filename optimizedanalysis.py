import streamlit as st
from transformers import pipeline
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Disable GPU for model inference
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Hugging Face models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    return summarizer, sentiment_analyzer

# Fetch news data
def fetch_news(stock_ticker, from_date, to_date, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&from={from_date}&to={to_date}&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error("Failed to fetch news. Check your API key and connection.")
        return []

# Summarize articles with dynamic length
# Summarize articles with dynamic length and validation
# Summarize articles with error handling
def summarize_articles(articles, summarizer):
    summaries = []
    for article in articles:
        # Safely retrieve title and description
        title = article.get("title", "")
        description = article.get("description", "")
        
        # Ensure both title and description are strings
        if not isinstance(title, str):
            title = ""
        if not isinstance(description, str):
            description = ""
        
        # Skip articles with no meaningful content
        if not title.strip() and not description.strip():
            continue
        
        # Concatenate title and description
        text = title.strip() + ". " + description.strip()
        
        # Adjust summarization lengths dynamically
        input_length = len(text.split())
        max_length = min(50, input_length)  # Ensure max length doesn't exceed input length
        min_length = max(10, max_length // 2)  # Set a reasonable min length
        
        try:
            # Generate summary
            summary = summarizer(text[:1024], max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append({"title": title, "summary": summary[0]["summary_text"]})
        except Exception as e:
            st.warning(f"Error summarizing article: {title}. Skipping it. ({str(e)})")
    return summaries


# Sentiment analysis
def analyze_sentiment(summaries, sentiment_analyzer):
    sentiment_results = []
    for summary in summaries:
        sentiment = sentiment_analyzer(summary["summary"])[0]
        sentiment_results.append({
            "title": summary["title"],
            "summary": summary["summary"],
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
        })
    return pd.DataFrame(sentiment_results)

# Plot sentiment trends
def plot_sentiment_trends(sentiment_df):
    sentiment_df["sentiment_category"] = sentiment_df["sentiment"].apply(
        lambda x: "Positive" if x == "POSITIVE" else "Negative" if x == "NEGATIVE" else "Neutral"
    )
    sentiment_counts = sentiment_df["sentiment_category"].value_counts()
    sentiment_counts.plot(kind="bar", color=["green", "red", "gray"], figsize=(10, 5))
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt)

# Main Streamlit App
def main():
    st.title("Optimized Generative AI Stock Trend Analyzer")
    st.sidebar.header("Inputs")
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
    num_days = st.sidebar.slider("Analyze news for the last (days):", 1, 30, 7)
    api_key = st.sidebar.text_input("Enter NewsAPI Key:")

    if st.sidebar.button("Analyze"):
        st.write(f"Fetching and analyzing news for {stock_ticker} over the last {num_days} days...")
        
        # Fetch news
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")
        articles = fetch_news(stock_ticker, from_date, to_date, api_key)
        
        if articles:
            st.write(f"### Found {len(articles)} articles. Starting analysis...")

            # Load models
            summarizer, sentiment_analyzer = load_models()

            # Summarize news
            summaries = summarize_articles(articles, summarizer)

            # Analyze sentiment
            sentiment_df = analyze_sentiment(summaries, sentiment_analyzer)

            # Display Results
            st.write("### Sentiment Analysis Results")
            st.dataframe(sentiment_df)

            # Plot sentiment trends
            st.write("### Sentiment Distribution")
            plot_sentiment_trends(sentiment_df)

            # Display summarized insights
            st.write("### Summarized News Insights")
            for idx, row in sentiment_df.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Summary:** {row['summary']}")
                st.write(f"**Sentiment:** {row['sentiment']} (Score: {row['score']:.2f})")
                st.write("---")
        else:
            st.error("No articles found. Please try a different stock ticker or date range.")

if __name__ == "__main__":
    main()
