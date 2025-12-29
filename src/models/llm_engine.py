# src/models/llm_engine.py - FIXED VERSION

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from pathlib import Path

class OptimizedLLMEngine:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_models()
        self._load_financial_context()
    
    def _setup_models(self):
        """Setup AI models with proper fallback handling"""
        self.use_transformers = False
        self.use_openai = False
        
        # Try OpenAI first (most reliable)
        try:
            import openai
            from src.config import config
            
            if config.OPENAI_API_KEY and len(config.OPENAI_API_KEY) > 20:
                openai.api_key = config.OPENAI_API_KEY
                
                # Test API connection
                try:
                    openai.Model.list()
                    self.use_openai = True
                    self.openai = openai
                    self.logger.info("✅ OpenAI GPT-4 initialized successfully")
                    return
                except Exception as e:
                    self.logger.warning(f"OpenAI test failed: {e}")
        except Exception as e:
            self.logger.warning(f"OpenAI initialization failed: {e}")
        
        # Fallback to rule-based
        self.logger.info("✅ Using enhanced rule-based AI (no external API needed)")
        self._setup_rule_based_models()
    
    def _setup_rule_based_models(self):
        """Enhanced rule-based sentiment analysis"""
        self.positive_words = {
            'bullish', 'growth', 'profit', 'gain', 'strong', 'buy', 'upgrade',
            'beat', 'exceed', 'rally', 'surge', 'momentum', 'breakout', 'outperform',
            'robust', 'solid', 'impressive', 'positive', 'optimistic', 'confidence'
        }
        self.negative_words = {
            'bearish', 'decline', 'loss', 'weak', 'sell', 'downgrade',
            'miss', 'crash', 'fall', 'pressure', 'breakdown', 'drop', 'underperform',
            'disappointing', 'concerning', 'risk', 'volatile', 'uncertainty'
        }
    
    def _load_financial_context(self):
        """Load financial market context"""
        self.market_signals = {
            'bull': ['uptrend', 'breakout', 'support', 'momentum'],
            'bear': ['downtrend', 'breakdown', 'resistance', 'selling'],
            'neutral': ['sideways', 'consolidation', 'range-bound']
        }
    
    def generate_portfolio_summary(self, portfolio_data: Dict, weights: Dict) -> str:
        """Generate portfolio summary with robust error handling"""
        try:
            if not portfolio_data or not weights:
                return "Insufficient portfolio data for analysis. Please configure your portfolio."
            
            # Calculate portfolio metrics
            holdings_performance = []
            total_weight = 0
            
            for symbol, weight in weights.items():
                if symbol in portfolio_data:
                    data = portfolio_data[symbol]
                    if not data.empty and 'Daily_Return' in data.columns:
                        latest_return = data['Daily_Return'].iloc[-1]
                        if pd.notna(latest_return):
                            holdings_performance.append({
                                'symbol': symbol,
                                'return': latest_return,
                                'weight': weight
                            })
                            total_weight += weight
            
            if not holdings_performance:
                return self._generate_fallback_summary(weights)
            
            # Calculate weighted portfolio return
            portfolio_return = sum(h['return'] * h['weight'] for h in holdings_performance) / total_weight if total_weight > 0 else 0
            
            # Use OpenAI if available
            if self.use_openai:
                return self._generate_openai_summary(holdings_performance, portfolio_return)
            
            # Use enhanced rule-based analysis
            return self._generate_enhanced_summary(holdings_performance, portfolio_return)
            
        except Exception as e:
            self.logger.error(f"Portfolio summary generation failed: {e}")
            return self._generate_fallback_summary(weights)
    
    def _generate_openai_summary(self, holdings: List[Dict], portfolio_return: float) -> str:
        """Generate summary using OpenAI GPT-4"""
        try:
            prompt = f"""Analyze this institutional portfolio performance:

Portfolio Return: {portfolio_return*100:.2f}%
Holdings: {holdings}

Provide a concise, professional 3-sentence summary covering:
1. Overall performance assessment
2. Key drivers (best/worst performers)
3. Forward-looking insight

Keep it under 100 words, institutional tone."""

            response = self.openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            summary = response['choices'][0]['message']['content'].strip()
            self.logger.info("✅ OpenAI summary generated successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"OpenAI summary failed: {e}")
            return self._generate_enhanced_summary(holdings, portfolio_return)
    
    def _generate_enhanced_summary(self, holdings: List[Dict], portfolio_return: float) -> str:
        """Enhanced rule-based summary generation"""
        
        # Performance classification
        if portfolio_return > 0.01:
            performance = "strong positive momentum"
            sentiment = "bullish"
        elif portfolio_return > 0:
            performance = "modest gains"
            sentiment = "cautiously optimistic"
        elif portfolio_return > -0.01:
            performance = "relatively flat"
            sentiment = "neutral"
        else:
            performance = "facing headwinds"
            sentiment = "defensive positioning recommended"
        
        # Find top and bottom performers
        sorted_holdings = sorted(holdings, key=lambda x: x['return'], reverse=True)
        best = sorted_holdings[0] if sorted_holdings else None
        worst = sorted_holdings[-1] if sorted_holdings else None
        
        # Generate summary
        summary_parts = [
            f"Portfolio showing {performance} with {portfolio_return*100:+.2f}% daily return."
        ]
        
        if best and worst and best != worst:
            summary_parts.append(
                f"{best['symbol']} leads the portfolio (+{best['return']*100:.1f}%) "
                f"while {worst['symbol']} lags ({worst['return']*100:+.1f}%)."
            )
        
        summary_parts.append(f"Overall market sentiment: {sentiment}.")
        
        return " ".join(summary_parts)
    
    def _generate_fallback_summary(self, weights: Dict) -> str:
        """Fallback summary when data is insufficient"""
        asset_count = len(weights)
        largest_position = max(weights.items(), key=lambda x: x[1]) if weights else (None, 0)
        
        return (f"Portfolio configured with {asset_count} assets. "
                f"Largest position: {largest_position[0]} ({largest_position[1]*100:.1f}%). "
                f"Awaiting market data for comprehensive analysis.")
    
    def generate_market_insight(self, stock_data: pd.DataFrame, symbol: str) -> str:
        """Generate market insight for individual stock"""
        try:
            if stock_data.empty or len(stock_data) < 2:
                return f"Insufficient data for {symbol} analysis."
            
            analysis = self._analyze_stock_metrics(stock_data, symbol)
            
            if 'error' in analysis:
                return f"Unable to analyze {symbol} - {analysis['error']}"
            
            if self.use_openai:
                return self._generate_openai_stock_insight(analysis, symbol)
            
            return self._create_insight_text(analysis, symbol)
            
        except Exception as e:
            self.logger.error(f"Market insight generation failed for {symbol}: {e}")
            return f"Market analysis for {symbol} is temporarily unavailable."
    
    def _analyze_stock_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze stock technical metrics"""
        try:
            if data.empty or len(data) < 2:
                return {'error': 'Insufficient data'}
            
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            price_change = (latest['Close'] - previous['Close']) / previous['Close'] * 100
            
            # Trend analysis
            if len(data) >= 20:
                sma_20 = data['Close'].tail(20).mean()
                trend = 'bullish' if latest['Close'] > sma_20 else 'bearish'
            else:
                trend = 'neutral'
            
            # Volume analysis
            avg_volume = data['Volume'].tail(10).mean()
            volume_signal = 'high' if latest['Volume'] > avg_volume * 1.5 else 'normal'
            
            return {
                'symbol': symbol,
                'price_change': price_change,
                'current_price': latest['Close'],
                'trend': trend,
                'volume_signal': volume_signal,
                'rsi': latest.get('RSI', 50)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_openai_stock_insight(self, analysis: Dict, symbol: str) -> str:
        """Generate stock insight using OpenAI"""
        try:
            prompt = f"""Analyze {symbol}:
- Price change: {analysis['price_change']:.1f}%
- Trend: {analysis['trend']}
- Volume: {analysis['volume_signal']}
- RSI: {analysis['rsi']:.0f}

Provide 2-sentence technical analysis for institutional investors."""

            response = self.openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI stock insight failed: {e}")
            return self._create_insight_text(analysis, symbol)
    
    def _create_insight_text(self, analysis: Dict, symbol: str) -> str:
        """Create insight text from analysis"""
        if 'error' in analysis:
            return f"Unable to analyze {symbol} - insufficient data."
        
        price_change = analysis['price_change']
        trend = analysis['trend']
        
        direction = "gained" if price_change > 0 else "declined"
        insight = f"{symbol} has {direction} {abs(price_change):.1f}% recently. "
        
        if trend == 'bullish':
            insight += "Technical indicators suggest bullish momentum. "
        elif trend == 'bearish':
            insight += "Technical indicators show bearish pressure. "
        
        if analysis['volume_signal'] == 'high':
            insight += "Above-average trading volume confirms the move."
        
        rsi = analysis.get('rsi', 50)
        if rsi > 70:
            insight += " RSI indicates overbought conditions."
        elif rsi < 30:
            insight += " RSI suggests oversold levels."
        
        return insight.strip()

# Global instance
llm_engine = OptimizedLLMEngine()