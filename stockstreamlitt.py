import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import talib
from datetime import datetime, timedelta
import chromadb
from alpha_vantage.techindicators import TechIndicators
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import functools
import streamlit as st
import torch

class OptimizedMarketAnalyzer:
    def __init__(self, alpha_vantage_key: str):
        self.alpha_vantage_key = alpha_vantage_key
        
        # Use lightweight models and cache their loading
        self._initialize_models()
        
        # Initialize ChromaDB with in-memory storage
        self.db = chromadb.Client()
        self.collection = self.db.create_collection(
            name="market_insights",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Cache for storing results
        self.cache = {}
        
    @st.cache_resource
    def _initialize_models(self):
        """Initialize AI models with caching and lightweight options."""
        # Use smaller, faster models
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load minimal version of FinBERT
        self.market_analyzer = pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Use lightweight embedding model
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        """Fetch and cache stock data."""
        return yf.download(ticker, period="1mo", progress=False)

    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def _fetch_company_news(self, ticker: str, max_news: int = 5) -> List[str]:
        """Fetch and cache recent news with limit."""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('h3')[:max_news]
            return [item.text for item in news_items]
        except:
            return ["No recent news available"]

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment with batching for efficiency."""
        return self.sentiment_analyzer(texts, batch_size=8)

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators efficiently."""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values

        # Calculate multiple indicators at once
        return {
            'rsi': talib.RSI(close)[-1],
            'macd': talib.MACD(close)[0][-1],
            'adx': talib.ADX(high, low, close)[-1],
            'volume_sma': talib.SMA(volume, timeperiod=20)[-1],
            'trend': 'UPTREND' if close[-1] > talib.SMA(close, timeperiod=20)[-1] else 'DOWNTREND'
        }

    @functools.lru_cache(maxsize=128)
    def generate_analysis(self, technical_data: str) -> str:
        """Generate analysis using cached results."""
        prompt = f"Market Analysis: {technical_data}"
        analysis = self.market_analyzer(prompt)
        return analysis[0]['label']

    def analyze_stock(self, ticker: str) -> Dict:
        """Perform optimized stock analysis."""
        cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d_%H')}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get stock data
        data = self.get_stock_data(ticker)
        
        # Calculate technical indicators
        technical_data = self.calculate_technical_indicators(data)
        
        # Get news and sentiment (limited to 5 latest news)
        news = self._fetch_company_news(ticker)
        sentiments = self.analyze_sentiment(news)
        
        # Generate final analysis
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'technical_analysis': technical_data,
            'news_sentiment': sentiments,
            'market_signal': self._generate_signal(technical_data, sentiments)
        }
        
        # Cache the results
        self.cache[cache_key] = analysis
        
        return analysis

    def _generate_signal(self, technical_data: Dict, sentiments: List[Dict]) -> Dict:
        """Generate trading signal efficiently."""
        # Quick signal generation based on technical and sentiment data
        technical_score = (
            (1 if technical_data['rsi'] < 30 else -1 if technical_data['rsi'] > 70 else 0) +
            (1 if technical_data['macd'] > 0 else -1)
        )
        
        sentiment_score = sum(1 if s['label'] == 'POSITIVE' else -1 for s in sentiments) / len(sentiments)
        
        combined_score = (technical_score + sentiment_score) / 2
        
        return {
            'signal': 'BUY' if combined_score > 0.5 else 'SELL' if combined_score < -0.5 else 'HOLD',
            'confidence': abs(combined_score) * 100,
            'sentiment': 'POSITIVE' if sentiment_score > 0 else 'NEGATIVE'
        }