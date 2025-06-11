# core/llm_query.py
"""
Natural language querying for Powerball data using LLMs.
Allows users to ask questions in plain English about lottery patterns and trends.
"""

import os
import streamlit as st
from typing import Dict, Optional, List, Tuple, Any
import pandas as pd
import json
import re
import time
import hashlib
import plotly.express as px
from .storage import get_store

# Cache for API requests to avoid redundant calls
if 'llm_cache' not in st.session_state:
    st.session_state.llm_cache = {}
    
# LLM usage metrics for the session
if 'llm_usage_stats' not in st.session_state:
    st.session_state.llm_usage_stats = {
        'total_calls': 0,
        'cached_calls': 0,
        'total_tokens': 0,
        'total_cost': 0.0,  # Approximate cost tracking
        'providers': {}  # Per-provider stats
    }

# Track available LLM providers and API keys
AVAILABLE_PROVIDERS = {}

# Check for OpenAI
try:
    import openai
    from openai import OpenAI
    AVAILABLE_PROVIDERS["OpenAI"] = "OPENAI_API_KEY"
except ImportError:
    pass

# Check for Anthropic
try:
    import anthropic
    AVAILABLE_PROVIDERS["Anthropic"] = "ANTHROPIC_API_KEY"
except ImportError:
    pass
    
# Approximate costs per 1000 tokens (for informational purposes)
TOKEN_COSTS = {
    "OpenAI": {
        "gpt-4o": {
            "input": 0.01,  # $0.01 per 1K input tokens
            "output": 0.03   # $0.03 per 1K output tokens
        }
    },
    "Anthropic": {
        "claude-3-5-sonnet-20241022": {
            "input": 0.003,  # $0.003 per 1K input tokens
            "output": 0.015   # $0.015 per 1K output tokens
        }
    }
}

def create_cache_key(prompt: str, provider: str, model: str) -> str:
    """Create a unique cache key for a prompt and provider combination."""
    # Get only a specific part of the prompt for the cache key to ensure similar 
    # questions with minor differences (like whitespace) still hit the cache
    prompt_parts = prompt.split("User question:", 1)
    if len(prompt_parts) > 1:
        # Only use the actual question for caching, not the data description
        prompt_for_hash = prompt_parts[1].strip().lower()
    else:
        # Fallback to using the full prompt if it doesn't have the expected format
        prompt_for_hash = prompt.strip().lower()
    
    # Create a hash of the prompt + provider + model
    hash_input = f"{prompt_for_hash}|{provider}|{model}"
    return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

def get_cached_response(prompt: str, provider: str, model: str) -> Optional[str]:
    """Check if a response is cached and still valid (less than 24 hours old)."""
    cache_key = create_cache_key(prompt, provider, model)
    
    if cache_key in st.session_state.llm_cache:
        cached_item = st.session_state.llm_cache[cache_key]
        timestamp = cached_item.get('timestamp', 0)
        response = cached_item.get('response')
        
        # Check if cache is still valid (less than 24 hours old)
        cache_age_hours = (time.time() - timestamp) / 3600
        if cache_age_hours < 24 and response:
            # Update cached call stats
            st.session_state.llm_usage_stats['cached_calls'] += 1
            return response
    
    return None

def cache_response(prompt: str, provider: str, model: str, response: str) -> None:
    """Cache a response for future use."""
    cache_key = create_cache_key(prompt, provider, model)
    
    st.session_state.llm_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }

def update_usage_statistics(provider: str, model: str, prompt_tokens: int, 
                           completion_tokens: int) -> None:
    """Update the token usage statistics for tracking."""
    total_tokens = prompt_tokens + completion_tokens
    
    # Update global stats
    st.session_state.llm_usage_stats['total_calls'] += 1
    st.session_state.llm_usage_stats['total_tokens'] += total_tokens
    
    # Ensure provider exists in stats
    if provider not in st.session_state.llm_usage_stats['providers']:
        st.session_state.llm_usage_stats['providers'][provider] = {
            'total_tokens': 0,
            'calls': 0,
            'cost': 0.0
        }
    
    # Update provider-specific stats
    provider_stats = st.session_state.llm_usage_stats['providers'][provider]
    provider_stats['total_tokens'] += total_tokens
    provider_stats['calls'] += 1
    
    # Calculate approximate cost
    cost = 0.0
    if provider in TOKEN_COSTS and model in TOKEN_COSTS[provider]:
        cost_rates = TOKEN_COSTS[provider][model]
        input_cost = (prompt_tokens / 1000) * cost_rates['input']
        output_cost = (completion_tokens / 1000) * cost_rates['output']
        cost = input_cost + output_cost
    
    provider_stats['cost'] += cost
    st.session_state.llm_usage_stats['total_cost'] += cost

def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment or user input, return None if not available."""
    env_var = AVAILABLE_PROVIDERS.get(provider)
    if not env_var:
        return None
    
    api_key = os.environ.get(env_var)
    if api_key:
        # Mask the key partially for display
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        st.success(f"Using {provider} API key from environment: {masked_key}")
        return api_key
    
    # If not in environment, ask the user
    user_key = st.text_input(
        f"{provider} API Key", 
        type="password",
        help=f"Enter your {provider} API key to use this LLM provider"
    )
    
    if user_key:
        # Save temporarily for this session
        os.environ[env_var] = user_key
        return user_key
    
    return None

def format_data_for_prompt(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Convert dataframe to a string representation for prompts.
    
    Args:
        df: The DataFrame to format
        max_rows: Maximum number of rows to include (affects token usage)
        
    Returns:
        String representation of DataFrame for LLM prompt
    """
    if df.empty:
        return "No data available."
    
    # Get column info and types - but be selective to save tokens
    columns_info = []
    key_columns = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'powerball']
    
    # For larger dataframes with many columns, only describe key columns in detail
    columns_to_describe = df.columns if len(df.columns) < 10 else [c for c in df.columns if c in key_columns]
    
    for col in columns_to_describe:
        dtype = str(df[col].dtype)
        sample = str(df[col].iloc[0]) if not df[col].iloc[0] is None else "None"
        columns_info.append(f"- {col} ({dtype}, example: {sample})")
    
    # Add description of columns
    if set(['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']).issubset(df.columns):
        info = [
            "The data represents Powerball lottery draws with the following structure:",
            "- n1, n2, n3, n4, n5: The five white balls drawn (numbers 1-69)",
            "- powerball: The red Powerball number (numbers 1-26)"
        ]
        columns_info.extend(info)
    
    # Add date range if available (useful context but low token cost)
    if 'draw_date' in df.columns:
        try:
            df_dates = pd.to_datetime(df['draw_date'])
            min_date = df_dates.min().strftime('%Y-%m-%d')
            max_date = df_dates.max().strftime('%Y-%m-%d')
            columns_info.append(f"- Date range: {min_date} to {max_date}")
        except:
            pass  # Skip if dates can't be converted
    
    # Sample data - use smaller sample to save tokens
    sample_data = df.head(max_rows).to_string(index=False)
    
    return (
        "DataFrame Information:\n"
        f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        f"- Columns:\n" + "\n".join(columns_info) + "\n\n"
        f"Sample Data (first {min(max_rows, len(df))} rows):\n{sample_data}\n"
    )

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Look for Python code blocks with markdown or triple backticks
    code_pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Fallback - try to find code without explicit formatting
    # This is more error-prone but helps with some LLM outputs
    lines = response.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            in_code_block = True
            code_lines.append(line)
        elif in_code_block and (
            line.strip().startswith("df.") or 
            "plt." in line or 
            "px." in line or
            "=" in line
        ):
            code_lines.append(line)
    
    if code_lines:
        return "\n".join(code_lines)
    
    return None

def query_openai(prompt: str, api_key: str, max_tokens: int = 1000,
              use_cache: bool = True, temperature: float = 0.2) -> str:
    """
    Query OpenAI models with caching, token tracking, and proper error handling.
    
    Args:
        prompt: The query text
        api_key: OpenAI API key
        max_tokens: Maximum token limit for the response
        use_cache: Whether to use and update the response cache
        temperature: Controls randomness (lower = more deterministic)
        
    Returns:
        Response text or error message
    """
    # Define model name
    model = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    
    # Check cache first if enabled
    if use_cache:
        cached_response = get_cached_response(prompt, "OpenAI", model)
        if cached_response:
            st.success("⚡ Retrieved response from cache (saving API costs)")
            return cached_response
    
    try:
        # Initialize the client
        client = OpenAI(api_key=api_key)
        
        # Track time for performance monitoring
        start_time = time.time()
        
        # Query the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a data analysis assistant specializing in lottery statistics. "
                    "When answering questions about Powerball data, focus on statistical analysis "
                    "and data visualization. Always include code where appropriate. "
                    "For visualizations, use Plotly Express (px) which is already imported. "
                    "Your code will be executed directly, so make sure it's correct and complete. "
                    "You have access to a pandas DataFrame called 'df' containing lottery data. "
                    "Be concise and efficient with your explanations to minimize token usage."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Log token usage statistics and timing
        response_time = time.time() - start_time
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Calculate token usage
        if hasattr(response, 'usage'):
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Update usage statistics
            update_usage_statistics("OpenAI", model, prompt_tokens, completion_tokens)
            
            # Estimate cost
            input_cost = (prompt_tokens / 1000) * TOKEN_COSTS["OpenAI"][model]["input"]
            output_cost = (completion_tokens / 1000) * TOKEN_COSTS["OpenAI"][model]["output"]
            total_cost = input_cost + output_cost
            
            # Display usage information
            st.caption(f"⏱️ Response time: {response_time:.2f}s | "
                      f"Tokens: {prompt_tokens} (input) + {completion_tokens} (output) = {total_tokens} total | "
                      f"Est. cost: ${total_cost:.4f}")
        
        # Cache successful responses if enabled
        if use_cache:
            cache_response(prompt, "OpenAI", model, response_text)
        
        return response_text
    except Exception as e:
        error_msg = f"Error querying OpenAI: {str(e)}"
        st.error(error_msg)
        # Don't cache errors
        return f"Sorry, I encountered an error: {str(e)}"

def query_anthropic(prompt: str, api_key: str, max_tokens: int = 1000,
                 use_cache: bool = True, temperature: float = 0.2) -> str:
    """
    Query Anthropic Claude models with caching, token tracking, and proper error handling.
    
    Args:
        prompt: The query text
        api_key: Anthropic API key
        max_tokens: Maximum token limit for the response
        use_cache: Whether to use and update the response cache
        temperature: Controls randomness (lower = more deterministic)
        
    Returns:
        Response text or error message
    """
    # Define model name
    model = "claude-3-5-sonnet-20241022"  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024.
    
    # Check cache first if enabled
    if use_cache:
        cached_response = get_cached_response(prompt, "Anthropic", model)
        if cached_response:
            st.success("⚡ Retrieved response from cache (saving API costs)")
            return cached_response
    
    try:
        # Initialize the client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Track time for performance monitoring
        start_time = time.time()
        
        # Query the API
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a data analysis assistant specializing in lottery statistics. "
                   "When answering questions about Powerball data, focus on statistical analysis "
                   "and data visualization. Always include code where appropriate. "
                   "For visualizations, use Plotly Express (px) which is already imported. "
                   "Your code will be executed directly, so make sure it's correct and complete. "
                   "You have access to a pandas DataFrame called 'df' containing lottery data. "
                   "Be concise and efficient with your explanations to minimize token usage.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Log timing information
        response_time = time.time() - start_time
        
        # Extract response text
        response_text = response.content[0].text
        
        # Calculate token usage
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            # Update usage statistics
            update_usage_statistics("Anthropic", model, input_tokens, output_tokens)
            
            # Estimate cost
            input_cost = (input_tokens / 1000) * TOKEN_COSTS["Anthropic"][model]["input"]
            output_cost = (output_tokens / 1000) * TOKEN_COSTS["Anthropic"][model]["output"]
            total_cost = input_cost + output_cost
            
            # Display usage information
            st.caption(f"⏱️ Response time: {response_time:.2f}s | "
                      f"Tokens: {input_tokens} (input) + {output_tokens} (output) = {total_tokens} total | "
                      f"Est. cost: ${total_cost:.4f}")
        
        # Cache successful responses if enabled
        if use_cache:
            cache_response(prompt, "Anthropic", model, response_text)
        
        return response_text
    except Exception as e:
        error_msg = f"Error querying Anthropic: {str(e)}"
        st.error(error_msg)
        # Don't cache errors
        return f"Sorry, I encountered an error: {str(e)}"

def execute_code_safely(code: str, df: pd.DataFrame) -> Dict:
    """
    Execute generated code in a safe environment with the dataframe.
    
    Args:
        code: Python code generated by LLM
        df: DataFrame to use in the execution
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Import necessary libraries
        import io
        import sys
        import gc
        from contextlib import redirect_stdout
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import Counter
        import resource
        import tracemalloc
        
        # Request high memory for the execution (516MB)
        memory_mb = 516
        memory_bytes = memory_mb * 1024 * 1024
        
        # Try to increase memory limits if platform supports it
        try:
            # Try to raise the memory limit (soft, hard)
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, AttributeError, resource.error):
            # If setting memory limit fails, continue anyway
            pass
        
        # Run garbage collection before execution
        gc.collect()
        
        # Prepare a more complete execution environment with limited data to prevent memory issues
        local_vars = {
            # Use a sample of data if it's large to prevent memory issues
            'df': df.iloc[:min(1000, len(df))].copy() if len(df) > 1000 else df.copy(),
            'pd': pd, 
            'px': px,
            'np': np,
            'plt': plt,
            'Counter': Counter,
            'stats': __import__('scipy.stats'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
            'math': __import__('math')
        }
        
        result = {'success': False, 'error': None, 'fig': None, 'data': None, 'text': None}
        
        # Start memory tracking
        tracemalloc.start()
        
        # Define a memory checker function
        def check_memory():
            current, peak = tracemalloc.get_traced_memory()
            # If we're using more than 90% of our limit, raise error
            if peak > 0.9 * memory_bytes:
                raise MemoryError(f"Memory limit approaching ({peak / 1024 / 1024:.1f}MB of {memory_mb}MB). Query is too complex.")
            return peak
        
        # Execute in smaller chunks to monitor memory usage
        lines = code.strip().split('\n')
        chunk_size = 10
        
        # Execute setup code first (import statements, etc.)
        setup_code = '\n'.join([line for line in lines if line.strip().startswith(('import ', 'from '))])
        if setup_code:
            exec(setup_code, globals(), local_vars)
            check_memory()
            
        # Now execute the rest in chunks
        remaining_lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
        
        # Capture stdout
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            # Execute the code with the expanded environment
            for i in range(0, len(remaining_lines), chunk_size):
                chunk = '\n'.join(remaining_lines[i:i+chunk_size])
                if chunk.strip():  # Skip empty chunks
                    exec(chunk, globals(), local_vars)
                    check_memory()  # Check after each chunk
        
        # Capture any printed output
        captured_output = output_buffer.getvalue()
        if captured_output.strip():
            result['text'] = captured_output
        
        # Check for generated outputs
        result['success'] = True
        if 'fig' in local_vars:
            result['fig'] = local_vars['fig']
        
        # For text or data outputs
        if 'result' in local_vars:
            result_val = local_vars['result']
            if isinstance(result_val, pd.DataFrame):
                # Limit large dataframes in output
                if len(result_val) > 500:
                    result['data'] = result_val.head(500).copy()
                    if result['text'] is None:
                        result['text'] = f"Note: Output limited to first 500 rows of {len(result_val)} total."
                    else:
                        result['text'] += f"\n\nNote: Output limited to first 500 rows of {len(result_val)} total."
                else:
                    result['data'] = result_val
            elif result['text'] is None:  # Don't override captured output
                result['text'] = str(result_val)
        
        # Run garbage collection after execution
        tracemalloc.stop()
        gc.collect()
        
        return result
    except MemoryError:
        return {
            'success': False,
            'error': "Memory limit exceeded (516MB). Try simplifying your query or reducing the data sample size.",
            'fig': None,
            'data': None,
            'text': None
        }
    except Exception as e:
        import traceback
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'error': error_message,
            'fig': None,
            'data': None,
            'text': None
        }

def direct_analysis(df: pd.DataFrame, query: str) -> dict:
    """Perform direct analysis without using external LLMs, based on common lottery questions."""
    result = {
        'explanation': '',
        'code': '',
        'success': True,
        'fig': None,
        'data': None,
        'text': None
    }
    
    query = query.lower()
    
    # Check for app summary request
    if 'summary' in query or 'overview' in query or 'about this app' in query or 'what is this' in query or 'what does this app do' in query:
        # App summary
        result['text'] = """
🎯 **Powerball Insights: Data Analysis Platform**

This application provides comprehensive analysis and visualization of Powerball lottery data, with the following features:

• **Frequency Analysis**: See which numbers are drawn most and least often
• **Day of Week Analysis**: Compare drawing patterns across different days of the week
• **Time Trends**: Track how numbers and patterns change over months and years
• **Combination Analysis**: Find frequently occurring pairs and groups of numbers
• **Sum Analysis**: Analyze the mathematical properties of winning number combinations
• **Machine Learning**: Experimental predictions and pattern recognition
• **Natural Language Queries**: Ask questions about the data in plain English
        """
        
        # Total draws count and date range if data is available
        total_draws = len(df)
        if 'draw_date' in df.columns:
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            min_date = df['draw_date'].min().strftime('%B %d, %Y')
            max_date = df['draw_date'].max().strftime('%B %d, %Y')
            result['text'] += f"\n\nThe current dataset contains {total_draws} Powerball drawings from {min_date} to {max_date}."
        else:
            result['text'] += f"\n\nThe current dataset contains {total_draws} Powerball drawings."
            
        result['explanation'] = "This is a summary of the Powerball Insights application's capabilities and the current dataset."
        
        # Simple visual overview
        tabs = ['Frequency', 'Day of Week', 'Time Trends', 'Combinations', 'Sums', 'ML Experimental', 'AI Query']
        tab_counts = [1] * len(tabs)  # Just for visualization
        
        fig = px.bar(
            x=tabs, 
            y=tab_counts,
            title="Powerball Insights: Module Overview",
            labels={'x': 'Analysis Module', 'y': ''},
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        )
        fig.update_layout(showlegend=False)
        result['fig'] = fig
        
        result['code'] = """# App summary visualization
tabs = ['Frequency', 'Day of Week', 'Time Trends', 'Combinations', 'Sums', 'ML Experimental', 'AI Query']
tab_counts = [1] * len(tabs)  # Just for visualization

fig = px.bar(
    x=tabs, 
    y=tab_counts,
    title="Powerball Insights: Module Overview",
    labels={'x': 'Analysis Module', 'y': ''},
    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
)
fig.update_layout(showlegend=False)
"""
        
        return result
        
    # Check for help or assistance requests
    if 'help' in query or 'guide' in query or 'capabilities' in query or 'features' in query or 'how to use' in query:
        # Help guide
        result['text'] = """
## How to Use the AI Query Feature

You can ask questions about the Powerball data in two ways:

**1. Direct Analysis (No API needed):**
- Ask common lottery questions without needing any API keys
- Works with questions about frequency, trends, combinations, sums, and day-of-week patterns
- Instant results for most common lottery questions

**2. LLM API (Advanced Analysis):**
- For more complex or custom questions
- Requires an OpenAI or Anthropic API key
- Provides detailed analysis with advanced visualizations

### Example Questions You Can Ask:
- "What are the most frequently drawn numbers?"
- "Show me trends over time for popular numbers"
- "What combinations of numbers appear together most often?"
- "What is the average sum of winning numbers?"
- "Compare draw patterns by day of week"
- "Which numbers are due to be drawn based on recency?"
        """
        
        result['explanation'] = "Here's a guide to help you use the AI query feature effectively."
        result['code'] = "# No code needed for help guide"
        return result
        
    # Check for common analysis patterns
    if 'average sum' in query or 'mean sum' in query or 'typical sum' in query:
        # Calculate average sum of numbers
        white_sum = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1).mean()
        
        # Create detailed text answer
        result_details = []
        result_details.append(f"🎯 **Average sum of white balls: {white_sum:.2f}**")
        
        # Add more context about the range
        sums_white = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
        min_sum = sums_white.min()
        max_sum = sums_white.max()
        result_details.append(f"Range of sums: {min_sum} to {max_sum}")
        
        # Add percentile information
        median_sum = sums_white.median()
        result_details.append(f"Median sum: {median_sum:.2f}")
        
        # Calculate most common sum
        most_common_sum = sums_white.value_counts().idxmax()
        most_common_count = sums_white.value_counts().max()
        result_details.append(f"Most frequent sum: {most_common_sum} (appeared {most_common_count} times)")
        
        if 'powerball' in query or 'all numbers' in query or 'all balls' in query:
            all_sum = df[['n1', 'n2', 'n3', 'n4', 'n5', 'powerball']].sum(axis=1).mean()
            result_details.insert(1, f"🎯 **Average sum including Powerball: {all_sum:.2f}**")
        
        result['text'] = "\n\n".join(result_details)
            
        # Add visualization
        sums_white = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
        fig = px.histogram(
            sums_white, 
            title="Distribution of Sum of White Balls",
            labels={'value': 'Sum', 'count': 'Frequency'},
            marginal="box"
        )
        fig.add_vline(x=white_sum, line_dash="dash", line_color="red", 
                     annotation_text="Average", annotation_position="top right")
        result['fig'] = fig
        
        result['explanation'] = "I've calculated the average sum of the white balls across all draws and " \
                              "provided a histogram showing the distribution of these sums."
        
        result['code'] = """# Calculate average sum of white balls
white_sum = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1).mean()
print(f"Average sum of white balls: {white_sum:.2f}")

# Visualize distribution of sums
sums_white = df[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
fig = px.histogram(
    sums_white, 
    title="Distribution of Sum of White Balls",
    labels={'value': 'Sum', 'count': 'Frequency'},
    marginal="box"
)
fig.add_vline(x=white_sum, line_dash="dash", line_color="red", 
             annotation_text="Average", annotation_position="top right")
"""
    
    elif 'frequent' in query or 'popular' in query or 'common' in query:
        # Analyze frequency of numbers
        if 'powerball' in query and 'white' not in query:
            # Just Powerball analysis
            pb_counts = df['powerball'].value_counts().reset_index()
            pb_counts.columns = ['Number', 'Frequency']
            pb_counts = pb_counts.sort_values('Number')
            
            fig = px.bar(
                pb_counts, 
                x='Number', 
                y='Frequency',
                title="Powerball Number Frequency",
                labels={'Number': 'Powerball Number', 'Frequency': 'Times Drawn'}
            )
            result['fig'] = fig
            result['data'] = pb_counts.sort_values('Frequency', ascending=False).head(10)
            
            result['explanation'] = "I've analyzed the frequency of Powerball numbers across all draws, " \
                                  "showing which numbers appear most often."
            
            result['code'] = """# Analyze Powerball frequency
pb_counts = df['powerball'].value_counts().reset_index()
pb_counts.columns = ['Number', 'Frequency']
pb_counts = pb_counts.sort_values('Number')

# Create visualization
fig = px.bar(
    pb_counts, 
    x='Number', 
    y='Frequency',
    title="Powerball Number Frequency",
    labels={'Number': 'Powerball Number', 'Frequency': 'Times Drawn'}
)
"""
        else:
            # White ball analysis
            balls = pd.melt(df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'], var_name='position', value_name='number')
            counts = balls['number'].value_counts().reset_index()
            counts.columns = ['Number', 'Frequency']
            counts = counts.sort_values('Number')
            
            fig = px.bar(
                counts, 
                x='Number', 
                y='Frequency',
                title="White Ball Frequency",
                labels={'Number': 'Ball Number', 'Frequency': 'Times Drawn'}
            )
            result['fig'] = fig
            result['data'] = counts.sort_values('Frequency', ascending=False).head(10)
            
            result['explanation'] = "I've analyzed the frequency of white ball numbers across all draws, " \
                                  "showing which numbers appear most often."
            
            result['code'] = """# Analyze white ball frequency
balls = pd.melt(df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'], var_name='position', value_name='number')
counts = balls['number'].value_counts().reset_index()
counts.columns = ['Number', 'Frequency']
counts = counts.sort_values('Number')

# Create visualization
fig = px.bar(
    counts, 
    x='Number', 
    y='Frequency',
    title="White Ball Frequency",
    labels={'Number': 'Ball Number', 'Frequency': 'Times Drawn'}
)
"""
    
    elif 'day' in query and ('week' in query or 'dow' in query):
        # Day of week analysis
        df_with_day = df.copy()
        df_with_day['draw_date'] = pd.to_datetime(df_with_day['draw_date'])
        df_with_day['day_of_week'] = df_with_day['draw_date'].dt.day_name()
        
        # Order days of week properly
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df_with_day['day_of_week'].value_counts().reindex(days_order).reset_index()
        dow_counts.columns = ['Day', 'Count']
        dow_counts = dow_counts.fillna(0)
        
        fig = px.bar(
            dow_counts, 
            x='Day', 
            y='Count',
            title="Draws by Day of Week",
            labels={'Day': 'Day of Week', 'Count': 'Number of Draws'},
            category_orders={"Day": days_order}
        )
        result['fig'] = fig
        result['data'] = dow_counts
        
        result['explanation'] = "I've analyzed the distribution of draws across different days of the week."
        
        if 'sum' in query:
            # Also analyze average sum by day of week
            df_with_day['white_sum'] = df_with_day[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
            day_avg = df_with_day.groupby('day_of_week')['white_sum'].mean().reindex(days_order).reset_index()
            day_avg.columns = ['Day', 'Avg Sum']
            
            fig2 = px.bar(
                day_avg, 
                x='Day', 
                y='Avg Sum',
                title="Average Sum by Day of Week",
                labels={'Day': 'Day of Week', 'Avg Sum': 'Average Sum of White Balls'},
                category_orders={"Day": days_order}
            )
            result['fig'] = fig2  # Replace previous figure
            result['data'] = day_avg
            
            result['explanation'] = "I've analyzed the average sum of white balls for different days of the week."
        
        result['code'] = """# Day of week analysis
df_with_day = df.copy()
df_with_day['draw_date'] = pd.to_datetime(df_with_day['draw_date'])
df_with_day['day_of_week'] = df_with_day['draw_date'].dt.day_name()

# Order days of week properly
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts = df_with_day['day_of_week'].value_counts().reindex(days_order).reset_index()
dow_counts.columns = ['Day', 'Count']
dow_counts = dow_counts.fillna(0)

# Create visualization
fig = px.bar(
    dow_counts, 
    x='Day', 
    y='Count',
    title="Draws by Day of Week",
    labels={'Day': 'Day of Week', 'Count': 'Number of Draws'},
    category_orders={"Day": days_order}
)
"""
    
    elif 'pair' in query or 'combination' in query:
        # Pair analysis - find frequently occurring pairs
        pairs = []
        for _, row in df.iterrows():
            white_balls = [row.n1, row.n2, row.n3, row.n4, row.n5]
            for i in range(len(white_balls)):
                for j in range(i+1, len(white_balls)):
                    pairs.append(tuple(sorted([white_balls[i], white_balls[j]])))
        
        pair_counts = pd.Series(pairs).value_counts().head(20).reset_index()
        pair_counts.columns = ['Pair', 'Frequency']
        pair_counts['Pair'] = pair_counts['Pair'].apply(lambda x: f"{x[0]} & {x[1]}")
        
        fig = px.bar(
            pair_counts.head(10), 
            x='Pair', 
            y='Frequency',
            title="Top 10 Most Frequent White Ball Pairs",
            labels={'Pair': 'Number Pair', 'Frequency': 'Times Drawn Together'}
        )
        result['fig'] = fig
        result['data'] = pair_counts
        
        result['explanation'] = "I've analyzed the most frequent pairs of white ball numbers that appear together in draws."
        
        result['code'] = """# Find frequently occurring pairs
pairs = []
for _, row in df.iterrows():
    white_balls = [row.n1, row.n2, row.n3, row.n4, row.n5]
    for i in range(len(white_balls)):
        for j in range(i+1, len(white_balls)):
            pairs.append(tuple(sorted([white_balls[i], white_balls[j]])))

pair_counts = pd.Series(pairs).value_counts().head(20).reset_index()
pair_counts.columns = ['Pair', 'Frequency']
pair_counts['Pair'] = pair_counts['Pair'].apply(lambda x: f"{x[0]} & {x[1]}")

# Create visualization
fig = px.bar(
    pair_counts.head(10), 
    x='Pair', 
    y='Frequency',
    title="Top 10 Most Frequent White Ball Pairs",
    labels={'Pair': 'Number Pair', 'Frequency': 'Times Drawn Together'}
)
"""
    
    elif 'due' in query or 'overdue' in query or 'recency' in query or 'last seen' in query or 'days since' in query:
        # Calculate days since each number was last drawn
        try:
            # Ensure draw_date is datetime
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            
            # Get the most recent date in the dataset
            most_recent_date = df['draw_date'].max()
            
            # Function to calculate days since last appearance for each number
            def days_since_last(number_col):
                days_since = {}
                all_numbers = pd.concat([df[number_col]])
                
                # For each possible number, find when it last appeared
                for num in range(1, 70 if number_col != 'powerball' else 27):
                    # Find the most recent date this number appeared
                    dates_with_num = df[df[number_col] == num]['draw_date']
                    if len(dates_with_num) > 0:
                        last_date = dates_with_num.max()
                        days_since[num] = (most_recent_date - last_date).days
                    else:
                        # If never drawn, set to a very high number
                        days_since[num] = 9999
                
                return days_since
            
            # Calculate days since last appearance for each white ball and Powerball
            white_days = {}
            for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
                days = days_since_last(col)
                # Merge with existing counts, taking the minimum days value
                for num, day_count in days.items():
                    if num in white_days:
                        white_days[num] = min(white_days[num], day_count)
                    else:
                        white_days[num] = day_count
            
            pb_days = days_since_last('powerball')
            
            # Convert to DataFrames for better display
            white_df = pd.DataFrame({
                'number': list(white_days.keys()),
                'days_since_last': list(white_days.values())
            })
            white_df = white_df[white_df['number'] <= 69].sort_values('days_since_last', ascending=False)
            
            pb_df = pd.DataFrame({
                'number': list(pb_days.keys()),
                'days_since_last': list(pb_days.values())
            })
            pb_df = pb_df[pb_df['number'] <= 26].sort_values('days_since_last', ascending=False)
            
            # Create visualizations
            fig1 = px.bar(white_df.head(15), x='number', y='days_since_last', 
                         title='Top 15 White Balls with Longest Days Since Last Drawn',
                         labels={'number': 'Ball Number', 'days_since_last': 'Days Since Last Drawn'},
                         color='days_since_last',
                         height=400, color_continuous_scale='Viridis')
            
            fig2 = px.bar(pb_df.head(10), x='number', y='days_since_last',
                         title='Top 10 Powerballs with Longest Days Since Last Drawn',
                         labels={'number': 'Ball Number', 'days_since_last': 'Days Since Last Drawn'},
                         color='days_since_last',
                         height=400, color_continuous_scale='Reds')
            
            # Combine figures into one
            from plotly.subplots import make_subplots
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=('White Balls Overdue', 'Powerballs Overdue'),
                               vertical_spacing=0.2)
            
            # Add traces from the individual figures
            for trace in fig1.data:
                fig.add_trace(trace, row=1, col=1)
            
            for trace in fig2.data:
                fig.add_trace(trace, row=2, col=1)
            
            # Update layout
            fig.update_layout(height=800, title='Numbers Overdue for Drawing (Days Since Last Appearance)')
            
            # Format text explanation
            white_text = "\n".join([f"• Ball {row['number']}: {row['days_since_last']} days" 
                                   for _, row in white_df.head(5).iterrows()])
            
            pb_text = "\n".join([f"• Powerball {row['number']}: {row['days_since_last']} days" 
                               for _, row in pb_df.head(3).iterrows()])
            
            result['text'] = f"""
## Most Overdue Numbers

### Top 5 White Balls:
{white_text}

### Top 3 Powerballs:
{pb_text}

These numbers have gone the longest since their last appearance.
            """
            
            result['explanation'] = "This analysis shows which numbers have been 'overdue' the longest based on days since their last appearance."
            result['fig'] = fig
            result['data'] = pd.concat([
                white_df.head(15).assign(type="White Ball"),
                pb_df.head(10).assign(type="Powerball")
            ])
            
            # Provide code
            result['code'] = """
# Calculate days since each number was last drawn
df['draw_date'] = pd.to_datetime(df['draw_date'])
most_recent_date = df['draw_date'].max()

# Function to calculate days since last appearance
def days_since_last(number_col):
    days_since = {}
    for num in range(1, 70 if number_col != 'powerball' else 27):
        dates_with_num = df[df[number_col] == num]['draw_date']
        if len(dates_with_num) > 0:
            last_date = dates_with_num.max()
            days_since[num] = (most_recent_date - last_date).days
        else:
            days_since[num] = 9999
    return days_since

# Calculate for white balls and powerball
white_days = {}
for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
    days = days_since_last(col)
    for num, day_count in days.items():
        if num in white_days:
            white_days[num] = min(white_days[num], day_count)
        else:
            white_days[num] = day_count

pb_days = days_since_last('powerball')

# Create DataFrames
white_df = pd.DataFrame({
    'number': list(white_days.keys()),
    'days_since_last': list(white_days.values())
})
white_df = white_df[white_df['number'] <= 69].sort_values('days_since_last', ascending=False)

pb_df = pd.DataFrame({
    'number': list(pb_days.keys()),
    'days_since_last': list(pb_days.values())
})
pb_df = pb_df[pb_df['number'] <= 26].sort_values('days_since_last', ascending=False)

# Create visualizations using plotly subplots
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, 
                   subplot_titles=('White Balls Overdue', 'Powerballs Overdue'),
                   vertical_spacing=0.2)

# Add white ball bars
fig.add_trace(
    px.bar(white_df.head(15), x='number', y='days_since_last', color='days_since_last').data[0],
    row=1, col=1
)

# Add powerball bars
fig.add_trace(
    px.bar(pb_df.head(10), x='number', y='days_since_last', color='days_since_last').data[0],
    row=2, col=1
)

# Update layout
fig.update_layout(height=800, title='Numbers Overdue for Drawing (Days Since Last Appearance)')
            """
            
        except Exception as e:
            # Provide a fallback if there's an error
            result['explanation'] = f"Error analyzing recency data: {str(e)}"
            result['code'] = "# Error in recency analysis"
            result['text'] = "Unable to calculate days since last drawn. Make sure your dataset includes draw dates."
        
        return result
    
    elif 'trend' in query or 'over time' in query or 'year' in query or 'month' in query:
        # Time trend analysis
        df_time = df.copy()
        df_time['draw_date'] = pd.to_datetime(df_time['draw_date'])
        df_time['year'] = df_time['draw_date'].dt.year
        
        if 'sum' in query:
            # Sum over time
            df_time['white_sum'] = df_time[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
            time_avg = df_time.groupby('year')['white_sum'].mean().reset_index()
            
            fig = px.line(
                time_avg, 
                x='year', 
                y='white_sum',
                title="Average Sum of White Balls by Year",
                labels={'year': 'Year', 'white_sum': 'Average Sum'},
                markers=True
            )
            result['fig'] = fig
            result['data'] = time_avg
            
            result['explanation'] = "I've analyzed how the average sum of white balls has changed over the years."
            
            result['code'] = """# Analyze sum trend over time
df_time = df.copy()
df_time['draw_date'] = pd.to_datetime(df_time['draw_date'])
df_time['year'] = df_time['draw_date'].dt.year
df_time['white_sum'] = df_time[['n1', 'n2', 'n3', 'n4', 'n5']].sum(axis=1)
time_avg = df_time.groupby('year')['white_sum'].mean().reset_index()

# Create visualization
fig = px.line(
    time_avg, 
    x='year', 
    y='white_sum',
    title="Average Sum of White Balls by Year",
    labels={'year': 'Year', 'white_sum': 'Average Sum'},
    markers=True
)
"""
        else:
            # General number trends - focus on one number for simplicity
            number_trends = pd.DataFrame()
            for year in sorted(df_time['year'].unique()):
                year_df = df_time[df_time['year'] == year]
                counts = pd.melt(year_df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'])['value'].value_counts()
                # Get top 5 numbers for this year
                top_nums = counts.head(5).index.tolist()
                for num in top_nums:
                    number_trends = pd.concat([number_trends, pd.DataFrame({
                        'year': [year],
                        'number': [num],
                        'frequency': [counts[num]]
                    })])
            
            # Focus on numbers that appear multiple times in top 5s
            common_numbers = number_trends['number'].value_counts()
            recurring = common_numbers[common_numbers > 1].index.tolist()
            if recurring:
                filtered = number_trends[number_trends['number'].isin(recurring[:5])]
                
                fig = px.line(
                    filtered, 
                    x='year', 
                    y='frequency',
                    color='number',
                    title="Frequency Trends of Popular Numbers",
                    labels={'year': 'Year', 'frequency': 'Frequency', 'number': 'Ball Number'},
                    markers=True
                )
                result['fig'] = fig
                result['data'] = filtered
                
                result['explanation'] = "I've analyzed how the frequency of popular white ball numbers has changed over the years."
                
            else:
                # Fallback if no recurring numbers
                all_trends = pd.DataFrame()
                for year in sorted(df_time['year'].unique()):
                    year_df = df_time[df_time['year'] == year]
                    balls = pd.melt(year_df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'])
                    all_trends = pd.concat([all_trends, pd.DataFrame({
                        'year': [year],
                        'avg_value': [balls['value'].mean()]
                    })])
                
                fig = px.line(
                    all_trends, 
                    x='year', 
                    y='avg_value',
                    title="Average Ball Value Trend Over Time",
                    labels={'year': 'Year', 'avg_value': 'Average Ball Value'},
                    markers=True
                )
                result['fig'] = fig
                result['data'] = all_trends
                
                result['explanation'] = "I've analyzed how the average white ball value has changed over the years."
            
            result['code'] = """# Analyze number trends over time
df_time = df.copy()
df_time['draw_date'] = pd.to_datetime(df_time['draw_date'])
df_time['year'] = df_time['draw_date'].dt.year

number_trends = pd.DataFrame()
for year in sorted(df_time['year'].unique()):
    year_df = df_time[df_time['year'] == year]
    counts = pd.melt(year_df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'])['value'].value_counts()
    # Get top 5 numbers for this year
    top_nums = counts.head(5).index.tolist()
    for num in top_nums:
        number_trends = pd.concat([number_trends, pd.DataFrame({
            'year': [year],
            'number': [num],
            'frequency': [counts[num]]
        })])

# Focus on numbers that appear multiple times in top 5s
common_numbers = number_trends['number'].value_counts()
recurring = common_numbers[common_numbers > 1].index.tolist()
filtered = number_trends[number_trends['number'].isin(recurring[:5])]

# Create visualization
fig = px.line(
    filtered, 
    x='year', 
    y='frequency',
    color='number',
    title="Frequency Trends of Popular Numbers",
    labels={'year': 'Year', 'frequency': 'Frequency', 'number': 'Ball Number'},
    markers=True
)
"""
    
    else:
        # Fallback for unrecognized queries - provide basic frequency analysis
        balls = pd.melt(df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'], var_name='position', value_name='number')
        counts = balls['number'].value_counts().reset_index()
        counts.columns = ['Number', 'Frequency']
        
        fig = px.bar(
            counts.sort_values('Number'), 
            x='Number', 
            y='Frequency',
            title="White Ball Frequency Distribution",
            labels={'Number': 'Ball Number', 'Frequency': 'Times Drawn'}
        )
        result['fig'] = fig
        result['data'] = counts.sort_values('Frequency', ascending=False).head(10)
        
        pb_counts = df['powerball'].value_counts().reset_index()
        pb_counts.columns = ['Number', 'Frequency']
        
        fig2 = px.bar(
            pb_counts.sort_values('Number'), 
            x='Number', 
            y='Frequency',
            title="Powerball Frequency Distribution",
            labels={'Number': 'Powerball Number', 'Frequency': 'Times Drawn'},
            color_discrete_sequence=['#ff5c5c']
        )
        
        result['explanation'] = "I've provided a basic frequency analysis of white balls and Powerball numbers."
        result['fig'] = fig  # Show just one figure to keep it simple
        
        result['code'] = """# Basic frequency analysis
# White ball frequency
balls = pd.melt(df, value_vars=['n1', 'n2', 'n3', 'n4', 'n5'], var_name='position', value_name='number')
counts = balls['number'].value_counts().reset_index()
counts.columns = ['Number', 'Frequency']

fig = px.bar(
    counts.sort_values('Number'), 
    x='Number', 
    y='Frequency',
    title="White Ball Frequency Distribution",
    labels={'Number': 'Ball Number', 'Frequency': 'Times Drawn'}
)

# Powerball frequency
pb_counts = df['powerball'].value_counts().reset_index()
pb_counts.columns = ['Number', 'Frequency']

fig2 = px.bar(
    pb_counts.sort_values('Number'), 
    x='Number', 
    y='Frequency',
    title="Powerball Frequency Distribution",
    labels={'Number': 'Powerball Number', 'Frequency': 'Times Drawn'},
    color_discrete_sequence=['#ff5c5c']
)
"""
    
    return result


def render_page() -> None:
    """Render the LLM query page."""
    st.header("🤖 Ask the Numbers (AI Analysis)")
    
    # Data retrieval
    df = get_store().latest()
    if df.empty:
        st.warning("No data is available. Please upload or ingest data first.")
        return
    
    # Data overview with expandable details
    with st.expander("Dataset Overview", expanded=False):
        st.markdown(format_data_for_prompt(df, max_rows=3).replace("\n", "  \n"))
        st.dataframe(df.head())
    
    # Add a notice about the analysis methods
    st.info("""
    **Two analysis methods available:**
    - **Direct Analysis**: Get instant answers to common lottery questions without needing API keys
    - **LLM API**: For advanced custom questions using OpenAI or Anthropic (requires API key)
    """)
    
    # Create tabs for LLM API and No-API modes
    tabs = st.tabs(["Direct Analysis (No API needed)", "LLM API (Advanced)"])
    
    # Tab 1: Direct Analysis (No API required)
    with tabs[0]:
        user_query_direct = st.text_area(
            "Ask a question about the Powerball data",
            placeholder="Example: What are the most frequently drawn numbers? Or: Show me a visualization of draws by day of week."
        )
        
        if user_query_direct and st.button("Analyze", type="primary", key="direct_analyze"):
            with st.spinner("Analyzing data..."):
                result = direct_analysis(df, user_query_direct)
                
                # First show text result if available for better visibility
                if result['text']:
                    st.success(result['text'])
                
                # Display explanation
                st.subheader("Analysis")
                st.markdown(result['explanation'])
                
                # Show code
                with st.expander("View Code", expanded=False):
                    st.code(result['code'], language="python")
                
                # Show results
                if result['fig']:
                    st.plotly_chart(result['fig'], use_container_width=True)
                
                if result['data'] is not None:
                    st.subheader("Data Result")
                    st.dataframe(result['data'])
    
    # Tab 2: LLM API (requires API keys)
    with tabs[1]:
        # Check if any providers are available
        if not AVAILABLE_PROVIDERS:
            st.warning(
                "No LLM providers found. Please install at least one of the following packages: "
                "openai, anthropic."
            )
        else:
            # Provider selection
            provider = st.selectbox(
                "Select LLM Provider",
                list(AVAILABLE_PROVIDERS.keys())
            )
            
            # Get API key for selected provider
            api_key = get_api_key(provider)
            if not api_key:
                st.warning(f"Please provide an API key for {provider} to continue.")
            else:
                # Token usage controls
                with st.expander("Advanced Settings", expanded=False):
                    st.write("Control token usage to manage API costs:")
                    max_tokens = st.slider(
                        "Max Response Tokens", 
                        min_value=100, 
                        max_value=4000, 
                        value=1000, 
                        step=100,
                        help="Higher values allow more detailed responses but cost more. 1000 tokens is ~750 words."
                    )
                    
                    data_sample_size = st.slider(
                        "Data Sample Size", 
                        min_value=3, 
                        max_value=20, 
                        value=5, 
                        step=1,
                        help="How many rows of data to include in the prompt. Smaller samples use fewer tokens."
                    )
                    
                    conciseness = st.radio(
                        "Response Style",
                        options=["Detailed", "Balanced", "Concise"],
                        index=1,
                        horizontal=True,
                        help="Controls how concise or detailed the model responses will be"
                    )
                    
                    # Conciseness will be mapped to temperature in the query functions
                
                # Input for user query
                user_query = st.text_area(
                    "Ask a question about the Powerball data",
                    placeholder="Example: What are the most frequently drawn numbers? Or: Show me a visualization of draws by day of week.",
                    key="llm_query"
                )
                
                # Display estimated token usage
                prompt_preview = f"User question about Powerball data: {user_query}\nWith {data_sample_size} rows of data"
                est_prompt_tokens = len(prompt_preview.split()) * 1.3  # Rough estimate
                st.caption(f"Estimated prompt tokens: ~{int(est_prompt_tokens)} (Actual usage may vary)")
                
                # Cache controls
                use_cache = st.toggle(
                    "Use response caching",
                    value=True, 
                    help="Cache responses for 24 hours to save costs on similar queries"
                )
                
                # Show usage stats
                if st.session_state.llm_usage_stats['total_calls'] > 0:
                    with st.expander("Usage Statistics", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total API Calls", st.session_state.llm_usage_stats['total_calls'])
                            st.metric("Cached Calls", st.session_state.llm_usage_stats['cached_calls'])
                        with col2:
                            st.metric("Total Tokens", st.session_state.llm_usage_stats['total_tokens'])
                            st.metric("Estimated Cost", f"${st.session_state.llm_usage_stats['total_cost']:.4f}")
                        
                        # Provider-specific stats if available
                        if provider in st.session_state.llm_usage_stats['providers']:
                            provider_stats = st.session_state.llm_usage_stats['providers'][provider]
                            st.subheader(f"{provider} Stats")
                            st.text(f"Calls: {provider_stats['calls']}")
                            st.text(f"Tokens: {provider_stats['total_tokens']}")
                            st.text(f"Cost: ${provider_stats['cost']:.4f}")
                
                if user_query and st.button("Analyze", type="primary", key="llm_analyze"):
                    with st.spinner(f"Querying {provider} for analysis..."):
                        # Prepare detailed prompt with adjusted sample size
                        prompt = (
                            f"Here is the lottery data information:\n\n"
                            f"{format_data_for_prompt(df, max_rows=data_sample_size)}\n\n"
                            f"User question: {user_query}\n\n"
                            f"Please analyze this data to answer the user's question. "
                            f"Include visualizations where appropriate using Plotly Express (px). "
                            f"Ensure your code runs correctly with the DataFrame structure shown above. "
                            f"Be {'concise and efficient' if conciseness=='Concise' else 'clear and thorough' if conciseness=='Detailed' else 'balanced'} in your explanations. "
                            f"Make any necessary data cleaning as part of your code."
                        )
                        
                        try:
                            # Map conciseness to temperature
                            temp_map = {"Detailed": 0.7, "Balanced": 0.4, "Concise": 0.2}
                            temperature = temp_map[conciseness]
                            
                            # Query the selected LLM with token limits and our caching toggle
                            if provider == "OpenAI":
                                response = query_openai(
                                    prompt=prompt, 
                                    api_key=api_key, 
                                    max_tokens=max_tokens,
                                    use_cache=use_cache,
                                    temperature=temperature
                                )
                            elif provider == "Anthropic":
                                response = query_anthropic(
                                    prompt=prompt, 
                                    api_key=api_key, 
                                    max_tokens=max_tokens,
                                    use_cache=use_cache,
                                    temperature=temperature
                                )
                            else:
                                st.error(f"Provider {provider} is configured but not implemented.")
                                return
                            
                            # Display LLM explanation
                            st.subheader("Analysis")
                            st.markdown(response)
                            
                            # Extract and execute code
                            code = extract_code_from_response(response)
                            if code:
                                st.subheader("Extracted Code")
                                st.code(code, language="python")
                                
                                # Execute the code and show results
                                with st.spinner("Executing analysis code..."):
                                    result = execute_code_safely(code, df)
                                    
                                    if result['success']:
                                        if result['fig']:
                                            st.subheader("Visualization")
                                            st.plotly_chart(result['fig'], use_container_width=True)
                                        
                                        if result['data'] is not None:
                                            st.subheader("Data Result")
                                            st.dataframe(result['data'])
                                        
                                        if result['text']:
                                            st.subheader("Result")
                                            st.markdown(result['text'])
                                            
                                        if not any([result['fig'], result['data'], result['text']]):
                                            st.info("Code executed successfully but didn't produce any explicit outputs.")
                                    else:
                                        st.error(f"Error executing code: {result['error']}")
                            else:
                                st.warning("No executable code was found in the response.")
                        except Exception as e:
                            error_msg = f"Error querying LLM API: {str(e)}"
                            st.error(error_msg)
                            
                            # Check if there's an API key error or rate limit
                            api_error = False
                            if "api key" in str(e).lower() or "auth" in str(e).lower() or "rate" in str(e).lower():
                                api_error = True
                                st.warning("🔑 There appears to be an issue with your API key or rate limits.")
                            
                            # Offer options for handling the error
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Try Direct Analysis Instead", key="try_direct"):
                                    with st.spinner("Running direct analysis..."):
                                        # Attempt to answer the question using built-in analysis
                                        result = direct_analysis(df, user_query)
                                        
                                        # Display results in the same format as the direct tab
                                        if result['text']:
                                            st.success(result['text'])
                                        
                                        st.subheader("Analysis")
                                        st.markdown(result['explanation'])
                                        
                                        # Show code
                                        with st.expander("View Code", expanded=False):
                                            st.code(result['code'], language="python")
                                        
                                        # Show results
                                        if result['fig']:
                                            st.plotly_chart(result['fig'], use_container_width=True)
                                        
                                        if result['data'] is not None:
                                            st.subheader("Data Result")
                                            st.dataframe(result['data'])
                            
                            with col2:
                                if api_error and st.button("Update API Key", key="update_key"):
                                    # Clear the stored key to prompt for a new one
                                    if provider == "OpenAI":
                                        if "OPENAI_API_KEY" in os.environ:
                                            del os.environ["OPENAI_API_KEY"]
                                        st.success("Please enter a new API key above.")
                                        st.rerun()
                                    elif provider == "Anthropic":
                                        if "ANTHROPIC_API_KEY" in os.environ:
                                            del os.environ["ANTHROPIC_API_KEY"]
                                        st.success("Please enter a new API key above.")
                                        st.rerun()
                            
                            st.info("If you're experiencing persistent API issues, try using the 'Direct Analysis' tab instead, which doesn't require an API key.")
    
    # Memory and usage tips
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Tips for better questions", expanded=False):
            st.markdown("""
        For best results, try asking:
        - "What numbers are most frequently drawn?"
        - "Show the distribution of Powerball numbers over time"
        - "Calculate the average sum of winning numbers"
        """)
            
    with col2:
        with st.expander("Handling memory limits", expanded=False):
            st.markdown("""
        ### Memory Management Tips
        
        If you see a "Memory limit exceeded" error:
        
        1. **Simplify your query** - Ask for a more specific analysis
        2. **Limit data** - Add "using the most recent 100 draws" to your query
        3. **Break it down** - Split complex requests into smaller questions
        4. **Use Direct Analysis** - The built-in analysis handles large datasets better
        
        The system has a memory limit of 516MB for code execution.
        """)
            
    # Original tips continued below
    with st.expander("Advanced usage tips", expanded=False):
        st.markdown("""
        - "Find pairs of numbers that appear together frequently"
        - "Visualize the trend of numbers by month or year"
        - "Which day of the week has the highest average sum of numbers?"
        """)