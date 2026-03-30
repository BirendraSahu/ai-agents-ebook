# 📊 Prompt Engineering Demo - Finance Data Analysis

A Streamlit application that demonstrates the importance of prompt engineering by comparing bad vs good prompts using finance data analysis.

## 🎯 Objective

This application teaches prompt engineering by:
- Showing how **bad prompts** produce vague, incorrect, or unhelpful results
- Demonstrating how **good prompts** produce accurate, actionable insights
- Providing real-time comparisons of results from different prompts
- Explaining best practices and common mistakes

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_prompt_demo.txt
```

### 2. Run the Application

```bash
streamlit run prompt_engineering_demo.py
```

### 3. Configure OpenAI API Key

1. Enter your OpenAI API key in the sidebar
2. The application will validate the key and enable all features

## 📋 Features

### 1. **Sample Finance Data**
- Automatically generated realistic finance data
- Includes: Stock prices, revenue, expenses, profit, market cap, PE ratio, dividend yield
- 2 years of daily data with realistic trends and anomalies

### 2. **Multiple Analysis Scenarios**
- **Financial Analysis**: Trend identification, anomaly detection, health assessment
- **Risk Assessment**: Volatility analysis, risk indicators, mitigation strategies
- **Forecasting**: Time series forecasting with confidence intervals
- **Performance Comparison**: Quarter-over-quarter and year-over-year analysis

### 3. **Side-by-Side Comparison**
- Compare bad prompts vs good prompts
- See real-time results from OpenAI GPT models
- Understand why certain prompts work better

### 4. **Educational Content**
- Best practices for prompt engineering
- Common mistakes to avoid
- Advanced techniques and strategies

## 📊 Example Scenarios

### Scenario 1: Financial Analysis

**Bad Prompt:**
```
Tell me about this data
```

**Good Prompt:**
```
Analyze the provided finance data and provide:
1. Key Trends: Identify significant trends in stock price, revenue, and profit over time
2. Anomalies: Highlight any unusual patterns or outliers
3. Financial Health: Assess overall financial health
4. Recommendations: Provide 3 actionable recommendations

Format your response with clear sections and use specific numbers from the data.
```

### Scenario 2: Risk Assessment

**Bad Prompt:**
```
Is this risky?
```

**Good Prompt:**
```
Perform a comprehensive risk assessment:
1. Volatility Analysis: Calculate and explain volatility
2. Risk Indicators: Identify specific risk factors
3. Risk Level: Classify as Low/Medium/High with justification
4. Risk Mitigation: Suggest 3 specific strategies

Use statistical measures and reference specific dates/values.
```

## 🎓 Key Learnings

### Best Practices

1. **Be Specific**: Clear, detailed instructions
2. **Provide Context**: Background information and data summary
3. **Define Output Format**: Specify structure and format
4. **Use Structured Instructions**: Numbered steps or sections
5. **Reference Data Elements**: Point to specific columns/values
6. **Set the Role**: Define expert persona
7. **Ask for Justification**: Request reasoning and evidence

### Common Mistakes

- ❌ Too vague: "What do you think?"
- ❌ No context: Missing background information
- ❌ Unclear output: No format specification
- ❌ Too broad: Asking for everything at once
- ❌ No examples: Not showing desired output
- ❌ Ignoring constraints: Not mentioning limitations
- ❌ Single shot: Not breaking into steps

## 🔧 Advanced Techniques

1. **Chain of Thought**: "Think step by step"
2. **Few-Shot Learning**: Provide examples
3. **Role Playing**: Assign expert roles
4. **Iterative Refinement**: Start broad, then narrow
5. **Constraint Setting**: Define boundaries
6. **Output Formatting**: JSON, tables, markdown
7. **Temperature Control**: Adjust for creativity vs accuracy

## 📈 Data Structure

The application generates finance data with:
- **Date**: Daily timestamps (2023-01-01 to 2024-12-31)
- **Stock_Price**: Simulated stock prices with trends
- **Volume**: Trading volume
- **Revenue**: Company revenue
- **Expenses**: Operating expenses
- **Profit**: Revenue - Expenses
- **Market_Cap**: Market capitalization
- **PE_Ratio**: Price-to-earnings ratio
- **Dividend_Yield**: Dividend yield percentage

## 🛠️ Technical Details

- **Framework**: Streamlit
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **Data Generation**: NumPy and Pandas
- **Caching**: Streamlit's `@st.cache_data` for data generation

## 💡 Usage Tips

1. **Start with Bad Prompts**: See how vague prompts produce poor results
2. **Compare with Good Prompts**: Understand the difference
3. **Experiment**: Try modifying prompts to see how results change
4. **Learn from Examples**: Study the provided good prompts
5. **Apply Best Practices**: Use the learnings in your own prompts

## 🔒 Security

- API keys are stored in session state only
- No keys are logged or persisted
- Keys are cleared when the session ends

## 📝 License

This is a demonstration application for educational purposes.

## 🤝 Contributing

Feel free to extend this application with:
- More analysis scenarios
- Additional data types
- Different LLM providers
- Advanced visualization
- Export functionality

---

**Happy Prompt Engineering!** 🚀
