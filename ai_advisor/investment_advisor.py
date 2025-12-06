"""
Agentic AI Investment Advisor (Powered by Gemini)
==================================================
An AI-powered investment advisor that uses Google Gemini with function calling to:
- Query your trading database for real market data
- Analyze historical performance of instruments
- Generate personalized investment recommendations
- Simulate portfolio returns based on actual data

This demonstrates RAG (Retrieval Augmented Generation) with agentic capabilities.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google import genai
from google.genai import types
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# Gemini API Configuration
# Set your API key as environment variable: export GEMINI_API_KEY="your-key-here"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL = "gemini-2.0-flash"

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


# ============================================================================
# DATABASE TOOL FUNCTIONS
# ============================================================================

def get_available_instruments(limit: int = 20) -> dict:
    """Get list of available instruments in the database"""
    conn = get_db_connection()
    query = """
    SELECT DISTINCT
        i.instrument_id,
        i.symbol,
        i.asset_class,
        COUNT(p.price_id) as price_records,
        MIN(p.ts) as first_date,
        MAX(p.ts) as last_date
    FROM instruments i
    LEFT JOIN prices p ON i.instrument_id = p.instrument_id
    GROUP BY i.instrument_id, i.symbol, i.asset_class
    ORDER BY price_records DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()

    return {
        "total_instruments": len(df),
        "instruments": df.to_dict('records')
    }


def get_instrument_stats(symbol: str) -> dict:
    """Get detailed statistics for a specific instrument"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get instrument ID
    cursor.execute("SELECT instrument_id FROM instruments WHERE UPPER(symbol) = UPPER(?)", (symbol,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"error": f"Instrument '{symbol}' not found in database"}

    instrument_id = row[0]

    # Get price history
    query = """
    SELECT ts, px_open, px_high, px_low, px_close, px_vol
    FROM prices WHERE instrument_id = ? ORDER BY ts
    """
    df = pd.read_sql_query(query, conn, params=(instrument_id,))
    conn.close()

    if len(df) == 0:
        return {"error": f"No price data found for {symbol}"}

    # Calculate statistics
    df['ts'] = pd.to_datetime(df['ts'], format='mixed')
    df['daily_return'] = df['px_close'].pct_change()

    avg_daily_return = df['daily_return'].mean()
    daily_volatility = df['daily_return'].std()

    annualized_return = avg_daily_return * 252 * 100
    annualized_volatility = daily_volatility * np.sqrt(252) * 100
    sharpe_ratio = (annualized_return - 2) / annualized_volatility if annualized_volatility > 0 else 0

    first_price = df['px_close'].iloc[0]
    last_price = df['px_close'].iloc[-1]
    total_return = ((last_price - first_price) / first_price) * 100

    # Max drawdown
    cumulative = (1 + df['daily_return']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return {
        "symbol": symbol.upper(),
        "data_period": {
            "start": df['ts'].min().strftime('%Y-%m-%d'),
            "end": df['ts'].max().strftime('%Y-%m-%d'),
            "trading_days": len(df)
        },
        "price_info": {
            "current_price": round(last_price, 2),
            "year_high": round(df['px_high'].tail(252).max(), 2) if len(df) >= 252 else round(df['px_high'].max(), 2),
            "year_low": round(df['px_low'].tail(252).min(), 2) if len(df) >= 252 else round(df['px_low'].min(), 2)
        },
        "performance": {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized_return, 2),
            "annualized_volatility_pct": round(annualized_volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown_pct": round(max_drawdown, 2)
        }
    }


def get_ml_predictions(symbol: str) -> dict:
    """Get ML model predictions for an instrument"""
    conn = get_db_connection()

    query = """
    SELECT
        mp.symbol,
        COUNT(*) as total_predictions,
        AVG(mp.hybrid_probability) as avg_confidence,
        SUM(CASE WHEN mp.recommendation = 'TAKE' THEN 1 ELSE 0 END) as take_count,
        SUM(CASE WHEN mp.recommendation = 'SKIP' THEN 1 ELSE 0 END) as skip_count,
        SUM(CASE WHEN mp.hybrid_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy,
        AVG(CASE WHEN mp.actual_outcome = 1 THEN 1.0 ELSE 0.0 END) as actual_win_rate
    FROM ml_predictions mp
    WHERE UPPER(mp.symbol) = UPPER(?)
    GROUP BY mp.symbol
    """

    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if len(df) == 0:
        return {"error": f"No ML predictions found for {symbol}"}

    row = df.iloc[0]
    take_ratio = row['take_count'] / row['total_predictions'] if row['total_predictions'] > 0 else 0

    if take_ratio > 0.6:
        recommendation = "BUY"
        reasoning = "ML model shows strong positive signals"
    elif take_ratio > 0.4:
        recommendation = "HOLD"
        reasoning = "ML model shows mixed signals"
    else:
        recommendation = "AVOID"
        reasoning = "ML model shows negative signals"

    return {
        "symbol": symbol.upper(),
        "ml_analysis": {
            "total_predictions": int(row['total_predictions']),
            "model_accuracy": round(float(row['accuracy']) * 100, 2),
            "avg_confidence": round(float(row['avg_confidence']) * 100, 2),
            "historical_win_rate": round(float(row['actual_win_rate']) * 100, 2)
        },
        "recommendation": {
            "action": recommendation,
            "take_signal_ratio": round(take_ratio * 100, 2),
            "reasoning": reasoning
        }
    }


def simulate_investment(symbol: str, amount: float, years: int) -> dict:
    """Simulate investment returns using Monte Carlo simulation"""
    stats = get_instrument_stats(symbol)

    if "error" in stats:
        return stats

    perf = stats["performance"]
    annual_return = perf["annualized_return_pct"] / 100
    volatility = perf["annualized_volatility_pct"] / 100

    # Monte Carlo simulation (1000 scenarios)
    n_simulations = 1000
    results = []

    for _ in range(n_simulations):
        portfolio_value = amount
        for year in range(years):
            year_return = np.random.normal(annual_return, volatility)
            portfolio_value *= (1 + year_return)
        results.append(portfolio_value)

    results = np.array(results)

    return {
        "symbol": symbol.upper(),
        "investment": {
            "initial_amount": amount,
            "time_horizon_years": years
        },
        "projections": {
            "expected_value": round(float(np.mean(results)), 2),
            "expected_return_pct": round(float((np.mean(results) - amount) / amount * 100), 2),
            "median_value": round(float(np.median(results)), 2),
            "best_case_90th_pct": round(float(np.percentile(results, 90)), 2),
            "worst_case_10th_pct": round(float(np.percentile(results, 10)), 2)
        },
        "risk_assessment": {
            "probability_of_loss": round(float((results < amount).sum() / len(results) * 100), 2),
            "probability_of_doubling": round(float((results >= amount * 2).sum() / len(results) * 100), 2),
            "annual_volatility_pct": round(volatility * 100, 2)
        },
        "methodology": "Monte Carlo simulation with 1000 scenarios based on historical returns"
    }


def compare_instruments(symbols: list) -> dict:
    """Compare multiple instruments by their performance"""
    comparisons = []

    for symbol in symbols[:5]:  # Limit to 5
        stats = get_instrument_stats(symbol)
        if "error" not in stats:
            comparisons.append({
                "symbol": symbol.upper(),
                "annualized_return_pct": stats["performance"]["annualized_return_pct"],
                "volatility_pct": stats["performance"]["annualized_volatility_pct"],
                "sharpe_ratio": stats["performance"]["sharpe_ratio"],
                "max_drawdown_pct": stats["performance"]["max_drawdown_pct"]
            })

    comparisons.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    return {
        "comparison": comparisons,
        "best_risk_adjusted": comparisons[0]["symbol"] if comparisons else None,
        "ranking_method": "Ranked by Sharpe Ratio (risk-adjusted returns)"
    }


def get_top_performers(metric: str = "returns", limit: int = 5) -> dict:
    """Get top performing instruments"""
    conn = get_db_connection()

    query = """
    SELECT DISTINCT i.symbol
    FROM instruments i
    JOIN prices p ON i.instrument_id = p.instrument_id
    GROUP BY i.instrument_id
    HAVING COUNT(*) > 100
    LIMIT 50
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    performers = []
    for symbol in df['symbol']:
        stats = get_instrument_stats(symbol)
        if "error" not in stats:
            performers.append({
                "symbol": symbol,
                "annualized_return_pct": stats["performance"]["annualized_return_pct"],
                "sharpe_ratio": stats["performance"]["sharpe_ratio"],
                "volatility_pct": stats["performance"]["annualized_volatility_pct"]
            })

    if metric == "sharpe_ratio":
        performers.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    else:
        performers.sort(key=lambda x: x["annualized_return_pct"], reverse=True)

    return {
        "metric": metric,
        "top_performers": performers[:limit]
    }


def get_portfolio_summary() -> dict:
    """Get overall portfolio and database summary"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT account_id) FROM account_metrics")
    num_accounts = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(total_trades), SUM(net_pnl), AVG(win_rate) FROM account_metrics")
    row = cursor.fetchone()
    total_trades = row[0] or 0
    total_pnl = row[1] or 0
    avg_win_rate = (row[2] or 0) * 100

    cursor.execute("SELECT COUNT(DISTINCT instrument_id) FROM prices")
    num_instruments = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ml_predictions")
    num_predictions = cursor.fetchone()[0]

    conn.close()

    return {
        "database_summary": {
            "total_accounts": num_accounts,
            "total_trades": total_trades,
            "total_net_pnl": round(total_pnl, 2),
            "average_win_rate_pct": round(avg_win_rate, 2),
            "instruments_tracked": num_instruments,
            "ml_predictions": num_predictions
        }
    }


# ============================================================================
# GEMINI FUNCTION DECLARATIONS
# ============================================================================

# Define tools for Gemini
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_available_instruments",
                description="Get a list of all available stocks/instruments in the database. Use when user asks what stocks are available.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max instruments to return (default 20)")
                    }
                )
            ),
            types.FunctionDeclaration(
                name="get_instrument_stats",
                description="Get historical statistics for a specific stock including returns, volatility, and price history.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol (e.g., AAPL, SPY, MSFT)")
                    },
                    required=["symbol"]
                )
            ),
            types.FunctionDeclaration(
                name="get_ml_predictions",
                description="Get ML model predictions and recommendations for a stock. Use when asked about AI predictions.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol")
                    },
                    required=["symbol"]
                )
            ),
            types.FunctionDeclaration(
                name="simulate_investment",
                description="Simulate investment projection. Use when user wants to know potential returns for investing money in a stock.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol"),
                        "amount": types.Schema(type=types.Type.NUMBER, description="Dollar amount to invest"),
                        "years": types.Schema(type=types.Type.INTEGER, description="Time horizon (1-5 years)")
                    },
                    required=["symbol", "amount", "years"]
                )
            ),
            types.FunctionDeclaration(
                name="compare_instruments",
                description="Compare multiple stocks by their performance metrics.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbols": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="List of stock symbols to compare"
                        )
                    },
                    required=["symbols"]
                )
            ),
            types.FunctionDeclaration(
                name="get_top_performers",
                description="Get the best performing stocks based on returns or risk-adjusted metrics.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "metric": types.Schema(type=types.Type.STRING, description="Metric: 'returns' or 'sharpe_ratio'"),
                        "limit": types.Schema(type=types.Type.INTEGER, description="Number to return")
                    }
                )
            ),
            types.FunctionDeclaration(
                name="get_portfolio_summary",
                description="Get overall database and portfolio summary statistics.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={}
                )
            )
        ]
    )
]


def execute_function(function_name: str, args: dict) -> str:
    """Execute a function and return JSON result"""

    if function_name == "get_available_instruments":
        result = get_available_instruments(args.get("limit", 20))
    elif function_name == "get_instrument_stats":
        result = get_instrument_stats(args["symbol"])
    elif function_name == "get_ml_predictions":
        result = get_ml_predictions(args["symbol"])
    elif function_name == "simulate_investment":
        result = simulate_investment(args["symbol"], args["amount"], args["years"])
    elif function_name == "compare_instruments":
        result = compare_instruments(args["symbols"])
    elif function_name == "get_top_performers":
        result = get_top_performers(args.get("metric", "returns"), args.get("limit", 5))
    elif function_name == "get_portfolio_summary":
        result = get_portfolio_summary()
    else:
        result = {"error": f"Unknown function: {function_name}"}

    return json.dumps(result)


# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_advisor(user_message: str, conversation_history: list = None) -> dict:
    """
    Chat with the AI Investment Advisor using Gemini with function calling.
    """

    if conversation_history is None:
        conversation_history = []

    system_instruction = """You are an AI Investment Advisor with access to a real trading database containing:
- Historical price data for 297+ instruments (stocks, ETFs)
- ML predictions from hybrid CNN-XGBoost models with 54%+ accuracy
- Over 1 million trade records with actual outcomes
- Performance metrics across 50 trading accounts

Your role:
1. Help users make informed investment decisions based on REAL data from our database
2. ALWAYS use the available functions to fetch actual data before making recommendations
3. Provide specific numbers and percentages from the database
4. Simulate investment scenarios when users ask about investing money
5. Warn about risks - past performance doesn't guarantee future results
6. Be conversational but professional
7. Format currency nicely (e.g., $10,000)

When a user asks about investing money, ALWAYS run simulate_investment to show projections.
When asked for recommendations, use get_top_performers or get_ml_predictions.
When comparing options, use compare_instruments."""

    # Build conversation for Gemini
    contents = []

    for msg in conversation_history:
        contents.append(types.Content(
            role=msg["role"],
            parts=[types.Part.from_text(text=msg["content"])]
        ))

    # Add current user message
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)]
    ))

    # Initial request to Gemini
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.7
        )
    )

    # Process function calls (agentic loop)
    while response.candidates[0].content.parts:
        # Check if there are function calls
        function_calls = [
            part for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not function_calls:
            break

        # Add assistant response to contents
        contents.append(response.candidates[0].content)

        # Execute each function call
        function_responses = []
        for part in function_calls:
            fc = part.function_call
            print(f"  [Calling: {fc.name}({dict(fc.args)})]")

            result = execute_function(fc.name, dict(fc.args))

            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result}
                )
            )

        # Add function results
        contents.append(types.Content(
            role="user",
            parts=function_responses
        ))

        # Get next response
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                temperature=0.7
            )
        )

    # Extract final text
    final_text = ""
    for part in response.candidates[0].content.parts:
        if part.text:
            final_text += part.text

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "model", "content": final_text})

    return {
        "response": final_text,
        "conversation_history": conversation_history
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   AI INVESTMENT ADVISOR - Powered by Gemini")
    print("   (Connected to your Trading Journal Database)")
    print("=" * 70)
    print("\nExample questions:")
    print("  - What stocks do you have data for?")
    print("  - If I invest $10,000 in SPY for 3 years, what returns can I expect?")
    print("  - Compare AAPL, MSFT, and GOOGL")
    print("  - What are the top performing stocks?")
    print("  - What does the ML model predict for TSLA?")
    print("\nType 'quit' to exit\n")

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy investing!")
                break

            if not user_input:
                continue

            print("\nAdvisor is analyzing...")

            result = chat_with_advisor(user_input, history)
            history = result["conversation_history"]

            print(f"\nAdvisor: {result['response']}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
