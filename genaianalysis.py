from langchain_community.llms import HuggingFaceHub
from transformers import pipeline
import yfinance as yf
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility
import requests
from datetime import datetime, timedelta
import asyncio
import os
from typing import Dict, List, Any

class DataFetcher:
    """Handle all data fetching operations"""
    
    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        
    def fetch_technical_data(self, symbol: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            hist['RSI'] = momentum.rsi(hist['Close'], window=14)
            macd = trend.MACD(hist['Close'])
            hist['MACD'] = macd.macd()
            hist['MACD_Signal'] = macd.macd_signal()
            
            current = hist.iloc[-1]
            prev = hist.iloc[-2]
            
            return {
                'current_price': round(current['Close'], 2),
                'daily_change': round(((current['Close']/prev['Close'] - 1) * 100), 2),
                'sma_20': round(current['SMA_20'], 2),
                'sma_50': round(current['SMA_50'], 2),
                'sma_200': round(current['SMA_200'], 2),
                'rsi': round(current['RSI'], 2),
                'macd': round(current['MACD'], 3),
                'macd_signal': round(current['MACD_Signal'], 3),
                'volume': int(current['Volume']),
                'avg_volume_10d': int(hist['Volume'].tail(10).mean()),
                'volatility_30d': round(hist['Close'].pct_change().std() * np.sqrt(252) * 100, 2),
                'high_52w': round(hist['High'].max(), 2),
                'low_52w': round(hist['Low'].min(), 2)
            }
            
        except Exception as e:
            print(f"Error fetching technical data: {str(e)}")
            return {}

    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'market_cap_b': round(info.get('marketCap', 0) / 1e9, 2),
                'pe_ratio': round(info.get('forwardPE', 0), 2),
                'pb_ratio': round(info.get('priceToBook', 0), 2),
                'ps_ratio': round(info.get('priceToSalesTrailing12Months', 0), 2),
                'profit_margin': round(info.get('profitMargins', 0) * 100, 2),
                'revenue_growth': round(info.get('revenueGrowth', 0) * 100, 2),
                'operating_margin': round(info.get('operatingMargins', 0) * 100, 2),
                'debt_to_equity': round(info.get('debtToEquity', 0), 2),
                'current_ratio': round(info.get('currentRatio', 0), 2),
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0,
                'earnings_growth': round(info.get('earningsQuarterlyGrowth', 0) * 100, 2),
                'industry': info.get('industry', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'company_name': info.get('longName', symbol)
            }
            
        except Exception as e:
            print(f"Error fetching fundamental data: {str(e)}")
            return {}

    def fetch_recent_headlines(self, symbol: str, days: int = 7) -> List[str]:
     headlines = []
    
     try:
        # Fetch from Yahoo Finance
        stock = yf.Ticker(symbol)
        news = stock.news
        if news:
            for item in news[:10]:
                if isinstance(item, dict) and 'title' in item:
                    headlines.append(item['title'])
        
        if not headlines:
            headlines = [f"No recent news available for {symbol}"]
        
        return headlines[:10]
        
     except Exception as e:
        print(f"Error fetching news headlines: {str(e)}")
        return [f"Unable to fetch news for {symbol}"] 

class GenAIStockAnalyzer:
    def __init__(self, huggingface_token):
        """Initialize the GenAI analyzer with both LLM and sentiment analysis"""
        # Initialize main LLM for analysis
        """self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=huggingface_token,
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 500,
                "top_p": 0.95
            }
        )
        """
        self.llm = HuggingFaceHub(
        repo_id="deepseek-ai/deepseek-coder-1.3b-instruct",
        huggingfacehub_api_token=huggingface_token,
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 500,
            "top_p": 0.95,
            "do_sample": True
        }
    )
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            token=huggingface_token
        )
        
        self.setup_prompts()
        self.data_fetcher = DataFetcher()

    def setup_prompts(self):
        self.prompts = {
            'analysis': """
            As a financial analyst, provide a detailed analysis for {symbol} based on the following data:

            Technical Indicators:
            - Current Price: ${price} ({change}% daily change)
            - Moving Averages: 20-day: ${sma_20}, 50-day: ${sma_50}, 200-day: ${sma_200}
            - RSI: {rsi}
            - MACD: {macd} (Signal: {macd_signal})
            - Volume: {volume:,} (10-day avg: {avg_volume:,})
            - Volatility (30d): {volatility}%
            - 52-Week Range: ${low_52w} - ${high_52w}

            Fundamental Data:
            - Market Cap: ${market_cap}B
            - P/E Ratio: {pe_ratio}
            - Revenue Growth: {revenue_growth}%
            - Operating Margin: {operating_margin}%
            - Industry: {industry}

            Recent Headlines Sentiment: {sentiment_summary}

            Provide a comprehensive analysis including:
            1. Technical Outlook (trend analysis, key levels, signals)
            2. Fundamental Assessment (valuation, growth prospects, risks)
            3. Market Sentiment Analysis
            4. Investment Recommendation
            5. Key Risk Factors
            6. Price Targets (support/resistance levels)
            """
        }

    async def analyze_sentiment(self, headlines: List[str]) -> Dict:
        """Analyze sentiment of headlines using FinBERT"""
        if not headlines:
            return {"overall": "neutral", "score": 0.0}
        
        # Get sentiment for each headline
        sentiments = self.sentiment_analyzer(headlines)
        
        # Calculate overall sentiment
        sentiment_scores = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for sentiment in sentiments:
            sentiment_scores[sentiment['label']] += 1
        
        # Determine overall sentiment
        total = len(sentiments)
        sentiment_summary = {
            'positive': sentiment_scores['positive'] / total,
            'negative': sentiment_scores['negative'] / total,
            'neutral': sentiment_scores['neutral'] / total
        }
        
        # Get dominant sentiment
        dominant_sentiment = max(sentiment_summary.items(), key=lambda x: x[1])
        
        return {
            "overall": dominant_sentiment[0],
            "score": dominant_sentiment[1],
            "details": sentiment_summary
        }

    async def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        # Fetch all required data
        technical_data = self.data_fetcher.fetch_technical_data(symbol)
        fundamental_data = self.data_fetcher.fetch_fundamental_data(symbol)
        headlines = self.data_fetcher.fetch_recent_headlines(symbol)
        
        # Analyze sentiment
        sentiment_analysis = await self.analyze_sentiment(headlines)
        
        # Format analysis prompt
        analysis_prompt = self.prompts['analysis'].format(
            symbol=symbol,
            price=technical_data['current_price'],
            change=technical_data['daily_change'],
            sma_20=technical_data['sma_20'],
            sma_50=technical_data['sma_50'],
            sma_200=technical_data['sma_200'],
            rsi=technical_data['rsi'],
            macd=technical_data['macd'],
            macd_signal=technical_data['macd_signal'],
            volume=technical_data['volume'],
            avg_volume=technical_data['avg_volume_10d'],
            volatility=technical_data['volatility_30d'],
            low_52w=technical_data['low_52w'],
            high_52w=technical_data['high_52w'],
            market_cap=fundamental_data['market_cap_b'],
            pe_ratio=fundamental_data['pe_ratio'],
            revenue_growth=fundamental_data['revenue_growth'],
            operating_margin=fundamental_data['operating_margin'],
            industry=fundamental_data['industry'],
            sentiment_summary=f"{sentiment_analysis['overall'].title()} ({sentiment_analysis['score']:.2%} confidence)"
        )
        
        # Get analysis from LLM
        response = await self.llm.agenerate([analysis_prompt])
        
        return {
            'symbol': symbol,
            'company_name': fundamental_data.get('company_name', symbol),
            'technical_data': technical_data,
            'fundamental_data': fundamental_data,
            'sentiment': sentiment_analysis,
            'headlines': headlines,
            'analysis': response.generations[0][0].text
        }

async def main():
    # Get HuggingFace token
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        huggingface_token = input("Enter your HuggingFace token: ")
    
    # Initialize analyzer
    analyzer = GenAIStockAnalyzer(huggingface_token)
    
    # Get stock symbol
    symbol = input("Enter stock symbol to analyze (e.g., AAPL): ").upper()
    
    # Perform analysis
    try:
        print(f"\nAnalyzing {symbol}...")
        result = await analyzer.analyze_stock(symbol)
        
        # Print results
        print("\n" + "="*50)
        print(f"Analysis Report for {result['company_name']} ({symbol})")
        print("="*50)
        print("\nAnalysis:")
        print(result['analysis'])
        
        print("\nRecent Headlines and Sentiment:")
        for headline in result['headlines']:
            print(f"- {headline}")
        
        print(f"\nOverall Sentiment: {result['sentiment']['overall'].title()} "
              f"({result['sentiment']['score']:.2%} confidence)")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())