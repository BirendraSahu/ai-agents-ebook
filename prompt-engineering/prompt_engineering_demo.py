import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import openai
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Prompt Engineering Demo - Finance Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 Prompt Engineering Demo: Finance Data Analysis")
st.markdown("**Learn how different prompts produce dramatically different results with the same data**")

# Sidebar for API key
with st.sidebar:
    st.header("🔑 Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use GPT models",
        value=st.session_state.get("openai_api_key", "")
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        try:
            openai.api_key = openai_api_key
            client = openai.OpenAI(api_key=openai_api_key)
            st.session_state.openai_client = client
            st.success("✅ API Key configured")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter your OpenAI API key")
        st.session_state.openai_client = None
    
    st.divider()
    st.markdown("### 📚 About This Demo")
    st.markdown("""
    This application demonstrates:
    - How **bad prompts** produce vague, incorrect, or unhelpful results
    - How **good prompts** produce accurate, actionable insights
    - The importance of:
      - Clear objectives
      - Context provision
      - Specific instructions
      - Output formatting
    """)

# Generate sample finance data
@st.cache_data
def generate_finance_data():
    """Generate realistic sample finance data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Generate stock price data with trends
    base_price = 100
    prices = []
    for i in range(len(dates)):
        trend = 0.1 * np.sin(i / 50)  # Long-term trend
        noise = np.random.normal(0, 2)
        price = base_price + trend * i + noise + np.random.normal(0, 5)
        prices.append(max(price, 10))  # Ensure positive prices
    
    # Generate volume data
    volumes = np.random.lognormal(10, 0.5, len(dates)).astype(int)
    
    # Generate financial metrics
    revenue = np.random.lognormal(12, 0.3, len(dates))
    expenses = revenue * (0.6 + np.random.normal(0, 0.1))
    profit = revenue - expenses
    
    df = pd.DataFrame({
        'Date': dates,
        'Stock_Price': prices,
        'Volume': volumes,
        'Revenue': revenue,
        'Expenses': expenses,
        'Profit': profit,
        'Market_Cap': prices * volumes * 0.1,
        'PE_Ratio': np.random.normal(25, 5, len(dates)),
        'Dividend_Yield': np.random.normal(2.5, 0.5, len(dates)),
    })
    
    # Add some anomalies
    df.loc[df.index[100:105], 'Stock_Price'] *= 1.2  # Price spike
    df.loc[df.index[200:205], 'Revenue'] *= 0.7  # Revenue drop
    df.loc[df.index[300:305], 'Profit'] *= -1  # Loss period
    
    return df

# Initialize finance data
finance_df = generate_finance_data()

# Main content
if not st.session_state.get("openai_client"):
    st.info("👆 Please enter your OpenAI API key in the sidebar to continue")
    st.stop()

# Display sample data
st.header("📈 Sample Finance Data")
st.markdown("**This is the data we'll analyze with different prompts:**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", f"{len(finance_df):,}")
with col2:
    st.metric("Date Range", f"{finance_df['Date'].min().date()} to {finance_df['Date'].max().date()}")
with col3:
    st.metric("Avg Stock Price", f"${finance_df['Stock_Price'].mean():.2f}")

# Show data preview
with st.expander("📊 View Sample Data (First 100 rows)"):
    st.dataframe(finance_df.head(100), use_container_width=True)

# Statistics summary
with st.expander("📈 Data Statistics"):
    st.dataframe(finance_df.describe(), use_container_width=True)

st.divider()

# Prompt Engineering Examples
st.header("🎯 Prompt Engineering Examples")

# Example scenarios with multiple bad prompts showing incremental improvements
scenarios = {
    "Financial Analysis": {
        "bad_prompts": [
            {
                "level": "Worst",
                "prompt": "Tell me about this data",
                "why_bad": "Completely vague, no context, no objective, no format"
            },
            {
                "level": "Very Bad",
                "prompt": "Analyze the finance data",
                "why_bad": "Slightly better (mentions 'analyze'), but still no specifics, no format, no context"
            },
            {
                "level": "Bad",
                "prompt": "Can you analyze the stock prices and revenue in this finance data?",
                "why_bad": "Mentions specific columns, but no structure, no output format, no detailed instructions"
            },
            {
                "level": "Okay",
                "prompt": "Please analyze the finance data and tell me about trends, anomalies, and financial health. Use numbers from the data.",
                "why_bad": "Has objectives and mentions using numbers, but lacks structure, no clear sections, no specific metrics to calculate"
            }
        ],
        "good_prompt": """Analyze the provided finance data and provide:
1. **Key Trends**: Identify significant trends in stock price, revenue, and profit over time
2. **Anomalies**: Highlight any unusual patterns or outliers (price spikes, revenue drops, loss periods)
3. **Financial Health**: Assess overall financial health based on profit margins, PE ratio, and dividend yield
4. **Recommendations**: Provide 3 actionable recommendations for investors

Format your response with clear sections and use specific numbers from the data.""",
        "context": "You are a financial analyst with expertise in stock market analysis and corporate finance."
    },
    "Risk Assessment": {
        "bad_prompts": [
            {
                "level": "Worst",
                "prompt": "Is this risky?",
                "why_bad": "Extremely vague, yes/no question, no context, no analysis requested"
            },
            {
                "level": "Very Bad",
                "prompt": "What are the risks in this data?",
                "why_bad": "Asks about risks but no structure, no specific risk types, no format"
            },
            {
                "level": "Bad",
                "prompt": "Analyze the risks in this finance data. Look at volatility and negative periods.",
                "why_bad": "Mentions specific aspects (volatility, negative periods) but no detailed analysis, no risk classification, no mitigation"
            },
            {
                "level": "Okay",
                "prompt": "Perform a risk assessment. Calculate volatility, identify risk factors, classify risk level, and suggest mitigation strategies.",
                "why_bad": "Has structure (4 tasks) but lacks specifics: no statistical measures mentioned, no format for risk classification, no detail on mitigation"
            }
        ],
        "good_prompt": """Perform a comprehensive risk assessment of this finance data:

1. **Volatility Analysis**: Calculate and explain the volatility of stock price and profit
2. **Risk Indicators**: Identify specific risk factors (e.g., negative profit periods, high PE ratios, declining trends)
3. **Risk Level**: Classify overall risk as Low/Medium/High with justification
4. **Risk Mitigation**: Suggest 3 specific risk mitigation strategies

Use statistical measures (standard deviation, coefficient of variation) and reference specific dates/values from the data.""",
        "context": "You are a risk management expert specializing in financial risk analysis."
    },
    "Forecasting": {
        "bad_prompts": [
            {
                "level": "Worst",
                "prompt": "What will happen next?",
                "why_bad": "Completely vague, no time frame, no metrics, no method, no context"
            },
            {
                "level": "Very Bad",
                "prompt": "Predict the future stock price",
                "why_bad": "Mentions what to predict but no time frame, no method, no confidence, no other metrics"
            },
            {
                "level": "Bad",
                "prompt": "Forecast the stock price, revenue, and profit for the next month based on historical data.",
                "why_bad": "Has time frame and metrics, but no method, no confidence intervals, no assumptions, no trend analysis"
            },
            {
                "level": "Okay",
                "prompt": "Create a forecast for the next 30 days. Analyze trends first, then predict stock price, revenue, and profit. State your assumptions.",
                "why_bad": "Better structure but missing: confidence intervals, confidence level rating, specific time series methods, pattern references"
            }
        ],
        "good_prompt": """Based on the historical finance data provided, create a forecast:

1. **Trend Analysis**: Identify the underlying trends in stock price, revenue, and profit
2. **Forecast Next 30 Days**: Provide point estimates and confidence intervals for:
   - Stock Price
   - Revenue
   - Profit
3. **Assumptions**: Clearly state the assumptions behind your forecast
4. **Confidence Level**: Rate your confidence (High/Medium/Low) and explain why

Use time series analysis principles and reference specific patterns from the historical data.""",
        "context": "You are a quantitative analyst specializing in financial forecasting and time series analysis."
    },
    "Performance Comparison": {
        "bad_prompts": [
            {
                "level": "Worst",
                "prompt": "Compare the data",
                "why_bad": "Extremely vague, no time periods, no metrics, no structure"
            },
            {
                "level": "Very Bad",
                "prompt": "Compare 2023 and 2024",
                "why_bad": "Has time periods but no metrics to compare, no structure, no format"
            },
            {
                "level": "Bad",
                "prompt": "Compare the financial performance between 2023 and 2024. Look at stock price, revenue, and profit.",
                "why_bad": "Has periods and metrics but no structure, no specific calculations, no best/worst periods, no insights"
            },
            {
                "level": "Okay",
                "prompt": "Compare Q1, Q2, Q3, Q4 of 2023 and 2024. Calculate average stock price, total revenue, profit margin, and volatility for each period.",
                "why_bad": "Better structure with periods and calculations, but missing: best/worst identification, percentage changes, table format, insights on factors"
            }
        ],
        "good_prompt": """Compare the financial performance across different time periods:

1. **Period Segmentation**: Divide the data into quarters and compare:
   - Q1 2023 vs Q2 2023 vs Q3 2023 vs Q4 2023
   - 2023 vs 2024 (year-over-year)
2. **Key Metrics Comparison**: For each period, compare:
   - Average stock price
   - Total revenue
   - Average profit margin
   - Volatility (standard deviation)
3. **Best/Worst Periods**: Identify the best and worst performing periods with specific metrics
4. **Insights**: Explain what factors might have contributed to performance differences

Present results in a clear table format with percentage changes.""",
        "context": "You are a financial performance analyst expert in comparative analysis."
    }
}

# Select scenario
selected_scenario = st.selectbox(
    "Select Analysis Scenario",
    options=list(scenarios.keys()),
    help="Choose a scenario to see how different prompts affect results"
)

scenario = scenarios[selected_scenario]

# Function to call OpenAI WITHOUT data context (to demonstrate the problem)
def call_openai_no_data(prompt: str, context: str = "", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API WITHOUT providing data context - demonstrates hallucination"""
    try:
        client = st.session_state.openai_client
        
        # NO data summary - this is the problem!
        system_prompt = f"{context}\n\nYou are asked to analyze finance data, but the actual data is not provided in this context."
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to call OpenAI WITH data context
def call_openai(prompt: str, context: str = "", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API with the prompt"""
    try:
        client = st.session_state.openai_client
        
        # Prepare data summary for context
        data_summary = f"""
        Finance Data Summary:
        - Date Range: {finance_df['Date'].min().date()} to {finance_df['Date'].max().date()}
        - Total Records: {len(finance_df):,}
        - Average Stock Price: ${finance_df['Stock_Price'].mean():.2f}
        - Average Revenue: ${finance_df['Revenue'].mean():.2f}
        - Average Profit: ${finance_df['Profit'].mean():.2f}
        - Stock Price Range: ${finance_df['Stock_Price'].min():.2f} - ${finance_df['Stock_Price'].max():.2f}
        - Profit Range: ${finance_df['Profit'].min():.2f} - ${finance_df['Profit'].max():.2f}
        - Key Statistics:
          * Stock Price Std Dev: ${finance_df['Stock_Price'].std():.2f}
          * Revenue Std Dev: ${finance_df['Revenue'].std():.2f}
          * Profit Margin: {(finance_df['Profit'] / finance_df['Revenue'] * 100).mean():.2f}%
          * Average PE Ratio: {finance_df['PE_Ratio'].mean():.2f}
          * Average Dividend Yield: {finance_df['Dividend_Yield'].mean():.2f}%
        
        Sample Data (first 5 rows):
        {finance_df.head(5).to_string()}
        
        Notable Patterns:
        - Price spike around {finance_df.loc[finance_df.index[100:105], 'Date'].iloc[0].date()}
        - Revenue drop around {finance_df.loc[finance_df.index[200:205], 'Date'].iloc[0].date()}
        - Loss period around {finance_df.loc[finance_df.index[300:305], 'Date'].iloc[0].date()}
        """
        
        system_prompt = f"{context}\n\nYou have access to the following finance data:\n{data_summary}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Special section: No Data vs With Data (after scenario is selected)
st.divider()
st.header("⚠️ Critical Lesson: Providing Data Context (RAG Scenario)")
st.markdown("**What happens when you ask LLM to analyze data WITHOUT actually providing the data?**")

with st.expander("🔍 Expand to see the 'No Data' problem", expanded=True):
    st.markdown("""
    ### The Problem: Missing Data Context
    
    In many real-world scenarios (like RAG applications), you might ask the LLM to analyze data, 
    but if the data isn't properly included in the context, the LLM will:
    
    - ❌ **Hallucinate** - Make up numbers and trends
    - ❌ **Use training data** - Reference general knowledge instead of your specific data
    - ❌ **Provide generic answers** - Give vague, non-specific responses
    - ❌ **No actual analysis** - Can't analyze what it can't see
    
    ### The Solution: Always Provide Data Context
    
    - ✅ Include data summary in system/user prompts
    - ✅ Use RAG to retrieve relevant data chunks
    - ✅ Provide specific data points when asking questions
    - ✅ Reference actual values from your dataset
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ Without Data Context (RAG Problem)")
        no_data_prompt = "Can you analyze the stock prices and revenue in this finance data?"
        st.code(no_data_prompt, language="text")
        st.markdown("""
        **What happens:**
        - LLM has no access to actual data
        - Will make assumptions or hallucinate
        - Can't reference specific values
        - Provides generic analysis
        - Uses training data knowledge instead
        """)
        
        if st.button("🔴 Run Without Data", key="no_data", type="secondary"):
            with st.spinner("🔄 Running prompt WITHOUT data context..."):
                # Call without data summary
                no_data_result = call_openai_no_data(no_data_prompt)
                st.session_state.no_data_result = no_data_result
    
    with col2:
        st.markdown("### ✅ With Data Context (RAG Solution)")
        with_data_prompt = "Can you analyze the stock prices and revenue in this finance data?"
        st.code(with_data_prompt, language="text")
        st.markdown("""
        **What happens:**
        - LLM has full data context
        - Can reference actual values
        - Provides specific analysis
        - Uses real numbers from dataset
        - No hallucination
        """)
        
        if st.button("🟢 Run With Data", key="with_data", type="primary"):
            with st.spinner("🔄 Running prompt WITH data context..."):
                # Call with data summary
                with_data_result = call_openai(with_data_prompt, scenario.get("context", ""))
                st.session_state.with_data_result = with_data_result
    
    # Show comparison if results exist
    if "no_data_result" in st.session_state or "with_data_result" in st.session_state:
        st.divider()
        st.markdown("### 📊 Results Comparison: No Data vs With Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ❌ Without Data Context")
            if "no_data_result" in st.session_state:
                st.markdown(st.session_state.no_data_result)
                st.warning("⚠️ **Notice:** The LLM is making assumptions or using general knowledge, not your actual data!")
                st.markdown("""
                **Common issues you'll see:**
                - Generic statements about finance
                - Made-up numbers or trends
                - References to general market knowledge
                - No specific dates or values from YOUR data
                """)
            else:
                st.info("Click 'Run Without Data' to see result")
        
        with col2:
            st.markdown("#### ✅ With Data Context")
            if "with_data_result" in st.session_state:
                st.markdown(st.session_state.with_data_result)
                st.success("✅ **Notice:** The LLM references actual values from your dataset!")
                st.markdown("""
                **What you'll see:**
                - Specific numbers from your data
                - Actual dates and periods
                - Real trends and patterns
                - Accurate calculations
                """)
            else:
                st.info("Click 'Run With Data' to see result")
        
        if "no_data_result" in st.session_state and "with_data_result" in st.session_state:
            st.info("""
            💡 **Key Takeaway:** 
            - **Without data**: LLM hallucinates or uses generic knowledge
            - **With data**: LLM provides specific, accurate analysis based on your actual data
            - This is why **RAG (Retrieval Augmented Generation)** is crucial - it ensures the LLM has access to relevant data context
            - Always verify that your RAG system is actually retrieving and including data in the prompt!
            """)

st.divider()

# Display prompts with incremental bad prompts
st.subheader("📊 Prompt Comparison: From Worst to Best")

# Create tabs for different bad prompt levels
bad_prompt_tabs = st.tabs([f"❌ {bp['level']}" for bp in scenario["bad_prompts"]] + ["✅ Good"])

# Display bad prompts in tabs
for idx, bad_prompt_info in enumerate(scenario["bad_prompts"]):
    with bad_prompt_tabs[idx]:
        st.markdown(f"### {bad_prompt_info['level']} Prompt")
        st.code(bad_prompt_info["prompt"], language="text")
        st.markdown(f"**Why it's {bad_prompt_info['level'].lower()}:**")
        st.markdown(f"- {bad_prompt_info['why_bad']}")
        
        # Show what's missing compared to good prompt
        st.markdown("**What's missing:**")
        missing_items = []
        if "vague" in bad_prompt_info['why_bad'].lower() or "no context" in bad_prompt_info['why_bad'].lower():
            missing_items.append("❌ Clear context and role")
        if "no structure" in bad_prompt_info['why_bad'].lower() or "no format" in bad_prompt_info['why_bad'].lower():
            missing_items.append("❌ Structured output format")
        if "no specific" in bad_prompt_info['why_bad'].lower() or "no detailed" in bad_prompt_info['why_bad'].lower():
            missing_items.append("❌ Specific instructions and metrics")
        if "no numbers" in bad_prompt_info['why_bad'].lower() or "no dates" in bad_prompt_info['why_bad'].lower():
            missing_items.append("❌ Data references and specific values")
        
        if missing_items:
            for item in missing_items:
                st.markdown(item)
        else:
            st.markdown("❌ Various improvements needed (see 'Why it's bad' above)")

# Display good prompt in last tab
with bad_prompt_tabs[-1]:
    st.markdown("### ✅ Good Prompt")
    st.code(scenario["good_prompt"], language="text")
    st.markdown("**Why it's good:**")
    st.markdown("""
    - ✅ Clear, specific objective with numbered tasks
    - ✅ Structured output format with sections
    - ✅ Specific instructions and metrics to calculate
    - ✅ References to data elements (columns, dates, values)
    - ✅ Defined role/context in system prompt
    - ✅ Asks for justification and explanations
    """)

# Show progression
st.divider()
st.markdown("### 📈 Prompt Quality Progression")
progression_text = " → ".join([bp['level'] for bp in scenario["bad_prompts"]] + ["Good"])
st.markdown(f"**Quality Level:** `{progression_text}`")
st.markdown("""
**Key Improvements at Each Level:**
- **Worst → Very Bad**: Adds basic topic mention
- **Very Bad → Bad**: Adds specific columns/metrics
- **Bad → Okay**: Adds structure and some instructions
- **Okay → Good**: Adds complete structure, format, role, and detailed requirements
""")

st.divider()

# Run analysis
st.subheader("🚀 Run Analysis")

# Create buttons for each prompt level
st.markdown("**Select which prompt to run:**")

button_cols = st.columns(len(scenario["bad_prompts"]) + 1)

# Buttons for bad prompts
for idx, bad_prompt_info in enumerate(scenario["bad_prompts"]):
    with button_cols[idx]:
        button_label = f"🔴 {bad_prompt_info['level']}"
        if st.button(button_label, key=f"bad_{idx}", use_container_width=True):
            st.session_state[f"run_bad_{idx}"] = True
            st.session_state["selected_bad_prompt"] = bad_prompt_info

# Button for good prompt
with button_cols[-1]:
    if st.button("🟢 Good Prompt", key="good", type="primary", use_container_width=True):
        st.session_state["run_good"] = True

# Display results
for idx in range(len(scenario["bad_prompts"])):
    if st.session_state.get(f"run_bad_{idx}", False):
        bad_prompt_info = scenario["bad_prompts"][idx]
        with st.spinner(f"🔄 Running {bad_prompt_info['level']} prompt..."):
            bad_result = call_openai(bad_prompt_info["prompt"])
            st.session_state[f"bad_result_{idx}"] = bad_result
            st.session_state[f"bad_prompt_info_{idx}"] = bad_prompt_info
        st.session_state[f"run_bad_{idx}"] = False

if st.session_state.get("run_good", False):
    with st.spinner("🔄 Running good prompt..."):
        good_result = call_openai(scenario["good_prompt"], scenario["context"])
        st.session_state.good_result = good_result
    st.session_state["run_good"] = False

# Show results
results_exist = any(f"bad_result_{idx}" in st.session_state for idx in range(len(scenario["bad_prompts"]))) or "good_result" in st.session_state

if results_exist:
    st.divider()
    st.subheader("📊 Results Comparison")
    
    # Create columns for results
    num_results = sum(1 for idx in range(len(scenario["bad_prompts"])) if f"bad_result_{idx}" in st.session_state)
    if "good_result" in st.session_state:
        num_results += 1
    
    if num_results > 0:
        # Show bad prompt results
        for idx in range(len(scenario["bad_prompts"])):
            if f"bad_result_{idx}" in st.session_state:
                bad_prompt_info = st.session_state[f"bad_prompt_info_{idx}"]
                with st.expander(f"❌ {bad_prompt_info['level']} Prompt Result", expanded=(idx == 0)):
                    st.markdown(f"**Prompt:** `{bad_prompt_info['prompt']}`")
                    st.markdown("**Result:**")
                    st.markdown(st.session_state[f"bad_result_{idx}"])
                    st.markdown(f"**Why it's {bad_prompt_info['level'].lower()}:** {bad_prompt_info['why_bad']}")
        
        # Show good prompt result
        if "good_result" in st.session_state:
            st.divider()
            st.markdown("### ✅ Good Prompt Result")
            st.markdown("**Prompt:**")
            st.code(scenario["good_prompt"], language="text")
            st.markdown("**Result:**")
            st.markdown(st.session_state.good_result)
            st.markdown("**Why it's good:**")
            st.markdown("""
            - ✅ Clear structure with numbered sections
            - ✅ Specific metrics and calculations
            - ✅ References to actual data values
            - ✅ Actionable recommendations
            - ✅ Professional formatting
            """)
        
        # Side-by-side comparison if both bad and good exist
        if any(f"bad_result_{idx}" in st.session_state for idx in range(len(scenario["bad_prompts"]))) and "good_result" in st.session_state:
            st.divider()
            st.subheader("⚖️ Direct Comparison")
            
            # Find the worst bad result for comparison
            worst_idx = 0
            if f"bad_result_{worst_idx}" in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### ❌ {st.session_state[f'bad_prompt_info_{worst_idx}']['level']} Prompt")
                    st.markdown(st.session_state[f"bad_result_{worst_idx}"][:500] + "..." if len(st.session_state[f"bad_result_{worst_idx}"]) > 500 else st.session_state[f"bad_result_{worst_idx}"])
                
                with col2:
                    st.markdown("### ✅ Good Prompt")
                    st.markdown(st.session_state.good_result[:500] + "..." if len(st.session_state.good_result) > 500 else st.session_state.good_result)
                
                st.info("💡 **Notice the difference:** The good prompt produces structured, specific, actionable insights with data references, while the bad prompt produces vague, generic responses.")
    else:
        st.info("Click any prompt button above to see results")

# Key Learnings
st.divider()
st.header("📚 Key Learnings")

with st.expander("🎓 Prompt Engineering Best Practices"):
    st.markdown("""
    ### 1. **ALWAYS Provide Data Context** ⚠️ CRITICAL
    - ❌ Bad: "Analyze this finance data" (without providing the data)
    - ✅ Good: "Analyze this finance data: [include data summary/context]"
    - **Why:** Without data, LLM will hallucinate or use generic knowledge
    
    ### 2. **Be Specific and Clear**
    - ❌ Bad: "Analyze this"
    - ✅ Good: "Analyze the stock price trends and identify the top 3 anomalies with specific dates and values"
    
    ### 3. **Provide Context**
    - ❌ Bad: No context about the data
    - ✅ Good: "You are a financial analyst. Here's the data: [summary]. Analyze..."
    
    ### 4. **Define Output Format**
    - ❌ Bad: "Tell me about risks"
    - ✅ Good: "List 5 risk factors in bullet points, each with: (1) Risk name, (2) Severity, (3) Impact"
    
    ### 5. **Use Structured Instructions**
    - ❌ Bad: Single vague question
    - ✅ Good: Numbered steps or sections (1. Analyze X, 2. Compare Y, 3. Recommend Z)
    
    ### 6. **Reference Specific Data Elements**
    - ❌ Bad: "Look at the trends"
    - ✅ Good: "Analyze the stock price column, identify periods where price > $120, and explain the correlation with revenue"
    
    ### 7. **Set the Role**
    - ❌ Bad: Generic assistant
    - ✅ Good: "You are a risk management expert specializing in financial markets"
    
    ### 8. **Ask for Justification**
    - ❌ Bad: "Is it risky?"
    - ✅ Good: "Classify risk level (Low/Medium/High) and provide 3 specific reasons with data references"
    """)

with st.expander("💡 Common Prompt Engineering Mistakes"):
    st.markdown("""
    ### Mistakes to Avoid:
    
    1. **⚠️ NO DATA CONTEXT** (MOST CRITICAL): Asking to analyze data without providing it
       - Example: "Analyze this finance data" (but data not in prompt)
       - Result: LLM hallucinates or uses generic knowledge
       - Fix: Always include data summary/context in RAG applications
    
    2. **Too Vague**: "What do you think about this data?"
    3. **No Context**: Not providing background information
    4. **Unclear Output**: Not specifying what format you want
    5. **Too Broad**: Asking for everything at once
    6. **No Examples**: Not showing what good output looks like
    7. **Ignoring Constraints**: Not mentioning data limitations
    8. **Single Shot**: Not breaking complex tasks into steps
    """)

with st.expander("🔧 Advanced Techniques"):
    st.markdown("""
    ### Advanced Prompt Engineering:
    
    1. **RAG (Retrieval Augmented Generation)**: 
       - Retrieve relevant data chunks from your database
       - Include retrieved data in the prompt context
       - Ensures LLM has access to actual data, not just training data
       - Critical for domain-specific analysis
    
    2. **Chain of Thought**: Ask the model to "think step by step"
    3. **Few-Shot Learning**: Provide examples of desired output
    4. **Role Playing**: Assign specific expert roles
    5. **Iterative Refinement**: Start broad, then narrow down
    6. **Constraint Setting**: Define boundaries and limitations
    7. **Output Formatting**: Specify JSON, tables, markdown, etc.
    8. **Temperature Control**: Lower (0.2) for factual, higher (0.8) for creative
    9. **Data Summarization**: Provide statistical summaries when full data is too large
    10. **Context Window Management**: Prioritize most relevant data in limited context windows
    """)

# Footer
st.divider()
st.markdown("---")
st.markdown("**Prompt Engineering Demo** - Demonstrating the power of well-crafted prompts")
