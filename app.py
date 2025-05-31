import pandas as pd
import io
import os
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
import docx
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from tabulate import tabulate
from flask import Flask, render_template, request, jsonify, send_from_directory, session , send_file

import uuid # For unique filenames
import shutil # For removing directories
import time # For performance timing

# --- Configuration ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found. Please set it in a .env file or as an environment variable.")

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" # Using the previous model as requested

# If Tesseract is not in your PATH, specify its path here.
# pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
# pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' # Example for macOS

app = Flask(__name__)
app.secret_key = os.urandom(24) # Required for session management - CRITICAL for sessions to work
app.config['UPLOAD_FOLDER'] = 'uploads' # Folder to save initial uploaded files
app.config['STATIC_PLOT_FOLDER'] = 'static/img' # Folder to save plots
app.config['TEMP_DATA_FOLDER'] = 'temp_data' # Folder to store temporary DataFrames

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_PLOT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_DATA_FOLDER'], exist_ok=True)


# --- Global/Session Data Storage ---
def get_session_data():
    """Retrieves session-specific data for the current user."""
    return session.get('ingested_data', {"content": None, "type": None})

def set_session_data(data):
    """Sets session-specific data."""
    session['ingested_data'] = data
    print(f"DEBUG: Session data updated. Type: {data['type']}") # Debugging line

def get_session_history():
    """Retrieves session-specific conversation history."""
    return session.get('conversation_history', [])

def set_session_history(history):
    """Sets session-specific conversation history."""
    session['conversation_history'] = history

# --- Helper to clean up old temp data files (optional, but good practice) ---
def clean_old_temp_data(max_age_seconds=3600): # 1 hour
    now = time.time()
    for filename in os.listdir(app.config['TEMP_DATA_FOLDER']):
        filepath = os.path.join(app.config['TEMP_DATA_FOLDER'], filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"DEBUG: Cleaned up old temp data file: {filepath}")
                except Exception as e:
                    print(f"ERROR: Could not remove old temp data file {filepath}: {e}")

# --- File Ingestion Functions ---

def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            set_session_data({"content": f.read(), "type": "text"})
        print(f"DEBUG: Successfully loaded TXT as text from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading TXT file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or df.dropna(how='all').empty:
            print(f"WARNING: CSV file '{file_path}' loaded but appears empty or contains no meaningful data. Setting session to None.")
            set_session_data({"content": None, "type": None})
            return False

        df_temp_filename = f"{uuid.uuid4().hex}.parquet"
        df_temp_filepath = os.path.join(app.config['TEMP_DATA_FOLDER'], df_temp_filename)
        df.to_parquet(df_temp_filepath) # Save DataFrame as a Parquet file
        
        set_session_data({"content": df_temp_filepath, "type": "dataframe"}) 

        print(f"DEBUG: Successfully loaded CSV as DataFrame from {file_path}. DataFrame shape: {df.shape}")
        print(f"DEBUG: Session 'ingested_data' after CSV load: Type={session['ingested_data']['type']}, Content_path={session['ingested_data']['content']}")
        return True
    except Exception as e:
        print(f"Error loading CSV file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def load_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        if df.empty or df.dropna(how='all').empty:
            print(f"WARNING: XLSX file '{file_path}' loaded but appears empty or contains no meaningful data. Setting session to None.")
            set_session_data({"content": None, "type": None})
            return False

        df_temp_filename = f"{uuid.uuid4().hex}.parquet"
        df_temp_filepath = os.path.join(app.config['TEMP_DATA_FOLDER'], df_temp_filename)
        df.to_parquet(df_temp_filepath) # Save DataFrame as a Parquet file

        set_session_data({"content": df_temp_filepath, "type": "dataframe"})

        print(f"DEBUG: Successfully loaded XLSX as DataFrame from {file_path}. DataFrame shape: {df.shape}")
        print(f"DEBUG: Session 'ingested_data' after XLSX load: Type={session['ingested_data']['type']}, Content_path={session['ingested_data']['content']}")
        return True
    except Exception as e:
        print(f"Error loading XLSX file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def load_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        set_session_data({"content": '\n'.join(full_text), "type": "text"})
        print(f"DEBUG: Successfully loaded DOCX as text from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading DOCX file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        set_session_data({"content": '\n'.join(full_text), "type": "text"})
        print(f"DEBUG: Successfully loaded PDF as text from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading PDF file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def load_image_ocr(file_path):
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if text.strip():
            set_session_data({"content": text, "type": "text"}) # Explicitly set as 'text'
            print(f"DEBUG: Successfully performed OCR on image from {file_path}. Text content found.")
            return True
        else:
            print(f"WARNING: No text found after OCR on image file: {file_path}. Setting session to None.")
            set_session_data({"content": None, "type": None}) # No text found
            return False
    except Exception as e:
        print(f"Error performing OCR on image file '{file_path}': {e}")
        set_session_data({"content": None, "type": None})
        return False

def upload_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    success = False
    
    # Start timing for data processing
    data_processing_start_time = time.time()

    if file_extension == '.txt':
        success = load_txt(file_path)
    elif file_extension == '.csv':
        success = load_csv(file_path)
    elif file_extension == '.xlsx':
        success = load_xlsx(file_path)
    elif file_extension in ['.doc', '.docx']:
        success = load_docx(file_path)
    elif file_extension == '.pdf':
        success = load_pdf(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
        success = load_image_ocr(file_path)
    else:
        print(f"Unsupported file type uploaded: {file_extension}. Setting session to None.")
        set_session_data({"content": None, "type": None})
        return False

    data_processing_end_time = time.time()
    print(f"DEBUG: Data processing (read, convert to Parquet) finished in {data_processing_end_time - data_processing_start_time:.2f} seconds.")


    if not success:
        print(f"File processing failed for {file_path}. Current session type after attempt: {get_session_data()['type']}")
        set_session_data({"content": None, "type": None})
    return success

# --- Together.ai LLM Integration ---

def get_llm_response(prompt_messages, retries=3, initial_delay=1):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": prompt_messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }

    for i in range(retries):
        try:
            response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
            return response_json['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                delay = initial_delay * (2 ** i) # Exponential backoff
                print(f"WARNING: Rate limit hit (429). Retrying in {delay:.2f} seconds... (Attempt {i+1}/{retries})")
                time.sleep(delay)
            else:
                print(f"ERROR: HTTP error during Together.ai API call: {e} (Status: {response.status_code})")
                return f"An HTTP error occurred with the AI: {e}"
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network or request error communicating with Together.ai API: {e}")
            return f"A network error occurred with the AI: {e}"
        except KeyError as e:
            print(f"ERROR: Unexpected response structure from Together.ai API: {e} - Response: {response.text}")
            return f"An error occurred due to an unexpected AI response: {e}"
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in get_llm_response: {e}")
            return f"An unexpected error occurred: {e}"

    return "Failed to get a response from the AI after multiple retries due to rate limiting."


# --- Data Analysis Utility Functions ---

def get_dataframe_from_session():
    """Helper to safely get the loaded DataFrame from session, reading from temp file."""
    session_data = get_session_data()
    print(f"DEBUG: get_dataframe_from_session called. Current session type: {session_data['type']}")
    if session_data["type"] == "dataframe" and session_data["content"] is not None:
        df_temp_filepath = session_data["content"]
        if os.path.exists(df_temp_filepath):
            try:
                df = pd.read_parquet(df_temp_filepath) # Read from Parquet file
                print(f"DEBUG: DataFrame successfully re-created from temp file: {df_temp_filepath}. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"ERROR: Failed to read DataFrame from temp file '{df_temp_filepath}': {e}. Session data might be corrupted or file missing.")
                set_session_data({"content": None, "type": None}) # Clear session if file cannot be read
                return None
        else:
            print(f"WARNING: DataFrame temp file not found at {df_temp_filepath}. Session is invalid for DataFrame.")
            set_session_data({"content": None, "type": None}) # Clear session if file is gone
            return None
    print("DEBUG: No structured data in session (type is not 'dataframe' or content is None).")
    return None

def save_dataframe_to_session(df):
    """Helper to save a modified DataFrame back to the session's temp file."""
    if df is None:
        set_session_data({"content": None, "type": None})
        return False
    
    session_data = get_session_data()
    df_temp_filepath = session_data.get("content")
    
    if not df_temp_filepath or not os.path.exists(df_temp_filepath):
        # If no existing file or file is gone, create a new one
        df_temp_filename = f"{uuid.uuid4().hex}.parquet"
        df_temp_filepath = os.path.join(app.config['TEMP_DATA_FOLDER'], df_temp_filename)
        print(f"DEBUG: Creating new temp parquet file: {df_temp_filepath}")
        
    try:
        df.to_parquet(df_temp_filepath)
        set_session_data({"content": df_temp_filepath, "type": "dataframe"})
        print(f"DEBUG: DataFrame successfully saved/updated to temp file: {df_temp_filepath}. Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save DataFrame to temp file '{df_temp_filepath}': {e}")
        return False

def get_dataframe_columns():
    """Returns a list of column names if a DataFrame is loaded, else empty list."""
    df = get_dataframe_from_session()
    if df is not None:
        return df.columns.tolist()
    return []

def get_dataframe_column_dtypes():
    """Returns a list of dictionaries with column name and dtype."""
    df = get_dataframe_from_session()
    if df is not None:
        return [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
    return []

def describe_dataframe():
    df = get_dataframe_from_session()
    if df is None:
        return "No structured data loaded to describe."
    return df.describe(include='all').to_markdown(index=True, numalign="left", stralign="left")

def get_column_info(column_name):
    df = get_dataframe_from_session()
    if df is None:
        return "No structured data loaded."
    if column_name not in df.columns:
        return f"Column '{column_name}' not found."

    col_info = io.StringIO()
    col_data = df[column_name]
    col_info.write(f"Column: {column_name}\n")
    col_info.write(f"Data Type: {col_data.dtype}\n")
    col_info.write(f"Non-null Count: {col_data.count()}\n")
    col_info.write(f"Missing Values: {col_data.isnull().sum()}\n")

    if pd.api.types.is_numeric_dtype(col_data):
        col_info.write(f"Mean: {col_data.mean():.2f}\n")
        col_info.write(f"Median: {col_data.median():.2f}\n")
        col_info.write(f"Std Dev: {col_data.std():.2f}\n")
        col_info.write(f"Min: {col_data.min()}\n")
        col_info.write(f"Max: {col_data.max()}\n")
    elif pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() < 20:
        col_info.write(f"Unique Values: {col_data.nunique()}\n")
        col_info.write(f"Value Counts:\n{col_data.value_counts().to_markdown(numalign='left', stralign='left')}\n")
    else:
        col_info.write(f"First 5 unique values: {col_data.unique()[:5].tolist()}\n")

    return col_info.getvalue()

def aggregate_data(group_cols, agg_col, agg_func):
    df = get_dataframe_from_session()
    if df is None:
        return "No structured data loaded for aggregation."
    if not all(col in df.columns for col in group_cols):
        return f"One or more grouping columns not found: {group_cols}"
    if agg_col not in df.columns:
        return f"Aggregation column '{agg_col}' not found."

    try:
        if agg_func == 'mean':
            result = df.groupby(group_cols)[agg_col].mean()
        elif agg_func == 'sum':
            result = df.groupby(group_cols)[agg_col].sum()
        elif agg_func == 'count':
            result = df.groupby(group_cols)[agg_col].count()
        elif agg_func == 'median':
            result = df.groupby(group_cols)[agg_col].median()
        elif agg_func == 'min':
            result = df.groupby(group_cols)[agg_col].min()
        elif agg_func == 'max':
            result = df.groupby(group_cols)[agg_col].max()
        else:
            return f"Unsupported aggregation function: {agg_func}"

        return f"Aggregation Result:\n{tabulate(result.reset_index(), headers='keys', tablefmt='grid')}"
    except Exception as e:
        return f"Error during aggregation: {e}"

def filter_data(column, operator, value):
    df = get_dataframe_from_session()
    if df is None:
        return "No structured data loaded for filtering."
    if column not in df.columns:
        return f"Column '{column}' not found."

    try:
        # Attempt to cast value to the column's type if possible
        if pd.api.types.is_numeric_dtype(df[column]):
            value = float(value) # Try float for broader compatibility
        # For date columns, try to parse
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            value = pd.to_datetime(value)

        if operator == '>':
            filtered_df = df[df[column] > value]
        elif operator == '<':
            filtered_df = df[df[column] < value]
        elif operator == '==':
            filtered_df = df[df[column] == value]
        elif operator == '>=':
            filtered_df = df[df[column] >= value]
        elif operator == '<=':
            filtered_df = df[df[column] <= value]
        elif operator == '!=':
            filtered_df = df[df[column] != value]
        elif operator == 'contains':
            filtered_df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            return f"Unsupported operator: {operator}"

        if not filtered_df.empty:
            return f"Filtered Data ({len(filtered_df)} rows):\n{tabulate(filtered_df.head(), headers='keys', tablefmt='grid')}"
        else:
            return "No data found matching the filter criteria."
    except Exception as e:
        return f"Error during filtering: {e}"

# --- NEW: Data Cleaning Functions ---

def handle_missing_values(column, method, value=None):
    df = get_dataframe_from_session()
    if df is None:
        return {"success": False, "message": "No structured data loaded."}
    if column not in df.columns and column != "all_columns":
        return {"success": False, "message": f"Column '{column}' not found."}

    original_rows = len(df)
    original_cols = len(df.columns)
    missing_before = df.isnull().sum().sum()

    try:
        new_df = df.copy() # Work on a copy
        
        if method == "drop_row":
            if column == "all_columns":
                new_df.dropna(inplace=True)
            else:
                new_df.dropna(subset=[column], inplace=True)
            rows_affected = original_rows - len(new_df)
            message = f"Dropped {rows_affected} rows with missing values."
        elif method == "drop_column":
            if column == "all_columns": # This option is less common for drop_column
                # Drop columns that have any missing values
                cols_to_drop = new_df.columns[new_df.isnull().any()].tolist()
                new_df.dropna(axis=1, inplace=True)
                message = f"Dropped {len(cols_to_drop)} columns with missing values: {', '.join(cols_to_drop)}"
            else:
                if new_df[column].isnull().any():
                    new_df.drop(columns=[column], inplace=True)
                    message = f"Dropped column '{column}' due to missing values."
                else:
                    message = f"Column '{column}' has no missing values to drop."
        elif method in ["mean", "median", "mode"]:
            if not pd.api.types.is_numeric_dtype(new_df[column]) and method in ["mean", "median"]:
                return {"success": False, "message": f"Cannot fill non-numeric column '{column}' with {method}. Use 'mode' or 'fill_value'."}
            
            fill_val = None
            if method == "mean":
                fill_val = new_df[column].mean()
            elif method == "median":
                fill_val = new_df[column].median()
            elif method == "mode":
                fill_val = new_df[column].mode()[0] # .mode() can return multiple if tied

            new_df[column].fillna(fill_val, inplace=True)
            message = f"Filled missing values in '{column}' with its {method} ({fill_val})."
        elif method == "fill_value":
            if value is None:
                return {"success": False, "message": "A 'value' must be provided for 'fill_value' method."}
            new_df[column].fillna(value, inplace=True)
            message = f"Filled missing values in '{column}' with '{value}'."
        else:
            return {"success": False, "message": f"Unsupported missing value handling method: {method}."}
        
        missing_after = new_df.isnull().sum().sum()
        if missing_after < missing_before:
            message += f" Total missing values reduced from {missing_before} to {missing_after}."
        
        save_dataframe_to_session(new_df)
        return {"success": True, "message": message, "rows": len(new_df), "cols": len(new_df.columns)}
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return {"success": False, "message": f"Error handling missing values: {e}"}

def convert_datatype(column, target_type):
    df = get_dataframe_from_session()
    if df is None:
        return {"success": False, "message": "No structured data loaded."}
    if column not in df.columns:
        return {"success": False, "message": f"Column '{column}' not found."}

    original_dtype = str(df[column].dtype)
    try:
        new_df = df.copy()
        if target_type == "numeric":
            new_df[column] = pd.to_numeric(new_df[column], errors='coerce')
        elif target_type == "datetime":
            new_df[column] = pd.to_datetime(new_df[column], errors='coerce')
        elif target_type == "string":
            new_df[column] = new_df[column].astype(str)
        elif target_type in ["int", "float"]: # Specific numeric types
            new_df[column] = pd.to_numeric(new_df[column], errors='coerce').astype(target_type)
        else:
            return {"success": False, "message": f"Unsupported target data type: {target_type}."}

        # Check if conversion introduced NaNs for non-nullable types and report
        if new_df[column].isnull().any() and original_dtype != 'object' and target_type not in ['numeric', 'datetime']:
             return {"success": False, "message": f"Conversion of '{column}' to {target_type} introduced missing values. Original values might not be compatible."}

        save_dataframe_to_session(new_df)
        return {"success": True, "message": f"Column '{column}' converted from {original_dtype} to {str(new_df[column].dtype)}.", "new_dtype": str(new_df[column].dtype)}
    except Exception as e:
        print(f"Error converting data type: {e}")
        return {"success": False, "message": f"Error converting data type for '{column}': {e}"}

def handle_duplicates_df(subset=None, keep='first'):
    df = get_dataframe_from_session()
    if df is None:
        return {"success": False, "message": "No structured data loaded."}

    original_rows = len(df)
    
    try:
        new_df = df.copy()
        if subset:
            if not all(col in new_df.columns for col in subset):
                return {"success": False, "message": f"One or more columns in subset not found: {subset}."}
            new_df.drop_duplicates(subset=subset, keep=keep, inplace=True)
            message_suffix = f" based on columns: {', '.join(subset)}"
        else:
            new_df.drop_duplicates(keep=keep, inplace=True)
            message_suffix = " across all columns"
        
        rows_affected = original_rows - len(new_df)
        save_dataframe_to_session(new_df)
        return {"success": True, "message": f"Removed {rows_affected} duplicate rows{message_suffix}. Remaining rows: {len(new_df)}.", "rows": len(new_df)}
    except Exception as e:
        print(f"Error handling duplicates: {e}")
        return {"success": False, "message": f"Error handling duplicates: {e}"}

def handle_outliers_iqr(column, method='remove', cap_range=1.5):
    df = get_dataframe_from_session()
    if df is None:
        return {"success": False, "message": "No structured data loaded."}
    if column not in df.columns:
        return {"success": False, "message": f"Column '{column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"success": False, "message": f"Column '{column}' is not numeric, cannot detect outliers."}

    original_rows = len(df)
    try:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - cap_range * IQR
        upper_bound = Q3 + cap_range * IQR

        new_df = df.copy()
        if method == 'remove':
            outlier_rows = new_df[(new_df[column] < lower_bound) | (new_df[column] > upper_bound)]
            new_df = new_df[~((new_df[column] < lower_bound) | (new_df[column] > upper_bound))]
            rows_affected = original_rows - len(new_df)
            message = f"Removed {rows_affected} rows with outliers in '{column}' based on IQR method."
        elif method == 'cap':
            capped_values = new_df[column].clip(lower=lower_bound, upper=upper_bound)
            num_capped = (capped_values != new_df[column]).sum()
            new_df[column] = capped_values
            message = f"Capped {num_capped} outliers in '{column}' based on IQR method."
        else:
            return {"success": False, "message": f"Unsupported outlier treatment method: {method}. Choose 'remove' or 'cap'."}
        
        save_dataframe_to_session(new_df)
        return {"success": True, "message": message, "rows": len(new_df)}
    except Exception as e:
        print(f"Error handling outliers: {e}")
        return {"success": False, "message": f"Error handling outliers: {e}"}


# --- Visualization Generation Functions ---

def generate_plot(plot_type, x_col, y_col=None, hue_col=None, title="Generated Plot", filename="plot.png"):
    df = get_dataframe_from_session()
    if df is None:
        return "No structured data loaded for plotting."
    # Validate columns only if they are provided and required for the plot type
    # Some plots (like hist) might only need x_col. Boxplot can take just y_col.
    required_cols = []
    if plot_type in ['line', 'scatter']:
        if not x_col or not y_col:
            return f"{plot_type} plot requires both X-axis and Y-axis columns."
        required_cols = [x_col, y_col]
    elif plot_type in ['bar', 'hist']:
        if not x_col:
            return f"{plot_type} plot requires an X-axis column."
        required_cols = [x_col]
    elif plot_type == 'box':
        # Box plot needs at least one of x or y
        if not x_col and not y_col:
            return "Box plot requires at least one column (X or Y axis)."
        if x_col and x_col not in df.columns:
            return f"Column '{x_col}' not found for plotting."
        if y_col and y_col not in df.columns:
            return f"Column '{y_col}' not found for plotting."
    
    for col in required_cols:
        if col and col not in df.columns:
            return f"Required column '{col}' not found in the dataset for '{plot_type}' plot."
    
    if hue_col and hue_col not in df.columns:
        return f"Hue column '{hue_col}' not found in the dataset for plotting."

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    try:
        if plot_type == 'bar':
            if y_col:
                sns.barplot(x=x_col, y=y_col, hue=hue_col, data=df)
                plt.ylabel(y_col)
            else: # Count plot for single categorical variable
                sns.countplot(x=x_col, hue=hue_col, data=df)
                plt.ylabel("Count")
            plt.xlabel(x_col)
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'line':
            sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=df)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'scatter':
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        elif plot_type == 'hist':
            sns.histplot(df[x_col], kde=True)
            plt.xlabel(x_col)
            plt.ylabel("Frequency")
        elif plot_type == 'box':
            if y_col and x_col: # Both X and Y provided
                sns.boxplot(x=x_col, y=y_col, data=df)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            elif y_col: # Only Y provided, make a single boxplot
                sns.boxplot(y=y_col, data=df)
                plt.ylabel(y_col)
            elif x_col: # Only X provided, make a single boxplot (assuming X is numeric)
                sns.boxplot(y=x_col, data=df)
                plt.ylabel(x_col)
        else:
            plt.close()
            return f"Unsupported plot type: {plot_type}"

        plt.title(title)
        plt.tight_layout()
        plot_path = os.path.join(app.config['STATIC_PLOT_FOLDER'], filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"DEBUG: Plot successfully saved to: {plot_path}")
        return f"Plot saved to: /{os.path.join(app.config['STATIC_PLOT_FOLDER'], filename)}"
    except Exception as e:
        plt.close()
        print(f"ERROR generating plot: {e}")
        return f"Error generating plot: {e}. Please check column names and data types."

# --- LLM Action Parsing and Execution ---

def parse_llm_action(response_text):
    action_prefix = "ACTION: "
    if action_prefix in response_text:
        try:
            action_json_str = response_text.split(action_prefix, 1)[1].strip()
            # Attempt to find the full JSON block if there's text after it
            if '}' in action_json_str:
                action_json_str = action_json_str[:action_json_str.rfind('}') + 1]
            else:
                print(f"WARNING: Incomplete JSON action from LLM. No closing brace found in: '{action_json_str}'")
                return None

            action_data = json.loads(action_json_str)
            print(f"DEBUG: Parsed LLM action: {action_data}")
            return action_data
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode LLM action JSON: {e} - Raw JSON attempt: '{action_json_str}'")
            return None
    return None

def execute_llm_action(action):
    result = {"text": "", "plot_url": None}
    print(f"DEBUG: Executing LLM action type: {action.get('type')}")

    if action["type"] == "plot":
        plot_filename = f"plot_{uuid.uuid4().hex}.png"
        plot_result_path = generate_plot(
            action.get("plot_type"),
            action.get("x_col"),
            action.get("y_col"),
            action.get("hue_col"),
            action.get("title", "Generated Plot"),
            plot_filename
        )
        if plot_result_path.startswith("Plot saved to: "):
            result["plot_url"] = plot_result_path.split("Plot saved to: ")[1]
            result["text"] = "Plot generated successfully by AI request."
        else:
            result["text"] = f"AI requested plot failed: {plot_result_path}"
    elif action["type"] == "analyze":
        analysis_type = action.get("analysis_type")
        if analysis_type == "aggregate":
            result["text"] = aggregate_data(
                action.get("group_cols", []),
                action.get("agg_col"),
                action.get("agg_func")
            )
        elif analysis_type == "filter":
            result["text"] = filter_data(
                action.get("column"),
                action.get("operator"),
                action.get("value")
            )
        elif analysis_type == "describe":
            result["text"] = describe_dataframe()
        elif analysis_type == "column_info":
            result["text"] = get_column_info(action.get("column_name"))
        else:
            result["text"] = f"Unknown analysis type: {analysis_type}"
    elif action["type"] == "clean": # NEW: Handle cleaning actions
        clean_action = action.get("action")
        if clean_action == "handle_missing_values":
            clean_result = handle_missing_values(
                action.get("column"),
                action.get("method"),
                action.get("value")
            )
            result["text"] = clean_result["message"]
        elif clean_action == "convert_datatype":
            clean_result = convert_datatype(
                action.get("column"),
                action.get("target_type")
            )
            result["text"] = clean_result["message"]
        elif clean_action == "handle_duplicates":
            clean_result = handle_duplicates_df(
                action.get("subset"),
                action.get("keep", 'first')
            )
            result["text"] = clean_result["message"]
        elif clean_action == "handle_outliers_iqr":
            clean_result = handle_outliers_iqr(
                action.get("column"),
                action.get("method", 'remove'),
                action.get("cap_range", 1.5)
            )
            result["text"] = clean_result["message"]
        else:
            result["text"] = f"Unknown cleaning action: {clean_action}"
    else:
        result["text"] = "Unsupported action type."

    return result

# --- Data Context for LLM ---

def get_data_context_for_llm():
    session_data = get_session_data()
    content_type = session_data["type"]
    context = ""
    print(f"DEBUG: Preparing data context for LLM. Current session type: {content_type}")

    if content_type == "dataframe":
        df = get_dataframe_from_session()
        if df is not None:
            schema_info = io.StringIO()
            df.info(buf=schema_info)
            schema_str = schema_info.getvalue()
            head_str = tabulate(df.head().astype(str), headers='keys', tablefmt='grid')

            # Add more detail about missing values for LLM
            missing_info = df.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            missing_summary = ""
            if not missing_cols.empty:
                missing_summary = "\n**Columns with Missing Values:**\n" + \
                                  tabulate(pd.DataFrame(missing_cols, columns=['Missing Count']), headers='keys', tablefmt='plain')
            else:
                missing_summary = "\nNo missing values detected in the dataset."


            context = (
                f"The user has provided a structured dataset (Pandas DataFrame) with {len(df)} rows and {len(df.columns)} columns.\n"
                f"Here is its schema:\n```\n{schema_str}\n```\n"
                f"And here are the first 5 rows:\n```\n{head_str}\n```\n"
                f"{missing_summary}\n"
                f"Carefully analyze the columns, their data types, and the sample values to understand the data content."
            )
        else:
            context = "No structured data has been loaded or it failed to parse correctly. This might be an empty or malformed file."
    elif content_type == "text":
        text_content = session_data["content"]
        if text_content is not None:
            truncated_content = text_content[:2000] + "..." if len(text_content) > 2000 else text_content
            context = (
                f"The user has provided an unstructured text document.\n"
                f"Here is a snippet of the document:\n```\n{truncated_content}\n```\n"
                f"Please read through the text to understand its content and identify key topics or entities."
            )
        else:
            context = "No text data has been loaded or extracted from the document."
    else:
        context = "No data has been loaded. Please upload a document first."
    return context

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("DEBUG: No file part in request.")
        return jsonify({"success": False, "message": "No file part", "type": None, "columns": []})
    file = request.files['file']
    if file.filename == '':
        print("DEBUG: No selected file.")
        return jsonify({"success": False, "message": "No selected file", "type": None, "columns": []})

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            print(f"DEBUG: File successfully saved to {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to save file {filepath}: {e}")
            return jsonify({"success": False, "message": f"Failed to save file: {e}", "type": None, "columns": []})

        # Clear previous session data and history on new upload
        # Also clean up old temp data files from previous sessions
        set_session_data({"content": None, "type": None})
        set_session_history([])
        clean_old_temp_data() # Clean up before potentially creating new temp data
        print("DEBUG: Session data and history cleared on new upload. Old temp data cleaned.")


        if upload_document(filepath):
            session_data_after_upload = get_session_data()
            document_type = session_data_after_upload["type"]
            columns = []
            column_dtypes = []
            if document_type == "dataframe":
                columns = get_dataframe_columns() # Only get columns if it's a dataframe
                column_dtypes = get_dataframe_column_dtypes() # Get column names and types

            print(f"DEBUG: Document processed successfully by upload_document. Final session type: {document_type}, Columns: {columns}")

            initial_message = "Document uploaded successfully. Analyzing data..."

            system_message = (
                "You are an intelligent data analysis agent. Your primary role is to "
                "understand the provided data (either structured or unstructured), "
                "summarize its content, and suggest types of analysis or questions that can be answered."
                "If it's structured data, highlight key columns, potential relationships, and suggest summary statistics or visualizations."
                "If it's unstructured text, identify main topics, entities, or potential insights."
                "Always start by confirming what type of data you have identified."
                "After any data cleaning or transformation, always provide a brief summary of the changes (e.g., 'X rows removed', 'Y values filled')."
                "If asked about feature engineering, you can suggest *potential* features given the column names, but do not generate an ACTION for it."
            )
            
            # Start timing for LLM overview generation
            llm_overview_start_time = time.time()

            user_message = (
                f"I have uploaded a document. Please provide an overview of the data it contains "
                f"and suggest some initial questions or analyses that could be performed. \n\n"
                f"DATA CONTEXT:\n{get_data_context_for_llm()}"
            )

            overview_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
            overview_response = get_llm_response(overview_messages)

            # End timing for LLM overview generation
            llm_overview_end_time = time.time()
            print(f"DEBUG: LLM overview generation finished in {llm_overview_end_time - llm_overview_start_time:.2f} seconds.")


            history = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}, {"role": "assistant", "content": overview_response}]
            set_session_history(history)

            return jsonify({
                "success": True,
                "message": initial_message,
                "overview": overview_response,
                "type": document_type,
                "columns": columns, # Simple list of names for basic plot/chat UIs
                "column_dtypes": column_dtypes # Detailed list for cleaning UIs
            })
        else:
            print(f"DEBUG: upload_document returned False for {filepath}. Final session type: {get_session_data()['type']}")
            return jsonify({"success": False, "message": "Failed to process document. It might be corrupted, malformed, or an unsupported file type.", "type": None, "columns": []})
    return jsonify({"success": False, "message": "Unknown error during upload.", "type": None, "columns": []})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    ingested_data = get_session_data()
    if ingested_data["content"] is None:
        print("DEBUG: Chat attempted, but no data in session.")
        return jsonify({"response": "Please upload a document first. No data is currently loaded."})

    conversation_history = get_session_history()

    # Define the system message, including available tools
    system_message = (
        "You are an intelligent data analysis agent. Your goal is to answer user questions about the provided data. "
        "You can analyze structured data (Pandas DataFrames) or extract information from unstructured text. "
        "When dealing with structured data, you have access to specific 'tools' for analysis, cleaning, and visualization. "
        "You MUST respond with a specific JSON action if you can fulfill the request using the tools. "
        "Otherwise, provide a direct textual answer based on your understanding of the data or explain why you cannot. "
        "Always be concise and helpful.\n\n"
        "Here's the current data context:\n" + get_data_context_for_llm() + "\n\n"
        "**AVAILABLE TOOLS:**\n"
        "1.  **To get basic statistics for a column:**\n"
        "    `ACTION: {\"type\": \"analyze\", \"analysis_type\": \"column_info\", \"column_name\": \"[column_name]\"}`\n"
        "2.  **To get descriptive statistics for the entire dataset:**\n"
        "    `ACTION: {\"type\": \"analyze\", \"analysis_type\": \"describe\"}`\n"
        "3.  **To aggregate data (e.g., sum, mean, count) by groups:**\n"
        "    `ACTION: {\"type\": \"analyze\", \"analysis_type\": \"aggregate\", \"group_cols\": [\"[column1]\"], \"agg_col\": \"[column_to_aggregate]\", \"agg_func\": \"[sum|mean|count|median|min|max]\"}`\n"
        "4.  **To filter data based on a condition:**\n"
        "    `ACTION: {\"type\": \"analyze\", \"analysis_type\": \"filter\", \"column\": \"[column_name]\", \"operator\": \"[>|<|==|>=|<=|!=|contains]\", \"value\": [value]}`\n"
        "5.  **To generate a bar chart:**\n"
        "    `ACTION: {\"type\": \"plot\", \"plot_type\": \"bar\", \"x_col\": \"[x_axis_column]\", \"y_col\": \"[y_axis_column]\", \"hue_col\": \"[optional_grouping_column]\", \"title\": \"[Plot Title]\"}`\n"
        "6.  **To generate a line chart:**\n"
        "    `ACTION: {\"type\": \"plot\", \"plot_type\": \"line\", \"x_col\": \"[x_axis_column]\", \"y_col\": \"[y_axis_column]\", \"hue_col\": \"[optional_grouping_column]\", \"title\": \"[Plot Title]\"}`\n"
        "7.  **To generate a scatter plot:**\n"
        "    `ACTION: {\"type\": \"plot\", \"plot_type\": \"scatter\", \"x_col\": \"[x_axis_column]\", \"y_col\": \"[y_axis_column]\", \"hue_col\": \"[optional_grouping_column]\", \"title\": \"[Plot Title]\"}`\n"
        "8.  **To generate a histogram:**\n"
        "    `ACTION: {\"type\": \"plot\", \"plot_type\": \"hist\", \"x_col\": \"[column_name]\", \"title\": \"[Plot Title]\"}`\n"
        "9.  **To generate a box plot:**\n"
        "    `ACTION: {\"type\": \"plot\", \"plot_type\": \"box\", \"x_col\": \"[x_axis_column]\", \"y_col\": \"[optional_y_axis_column]\", \"title\": \"[Plot Title]\"}`\n"
        "10. **To handle missing values:**\n"
        "    `ACTION: {\"type\": \"clean\", \"action\": \"handle_missing_values\", \"column\": \"[column_name|'all_columns']\", \"method\": \"[mean|median|mode|drop_row|drop_column|fill_value]\", \"value\": [optional_value_for_fill]}`\n"
        "    *Example: to fill 'Age' with mean: `{\"type\": \"clean\", \"action\": \"handle_missing_values\", \"column\": \"Age\", \"method\": \"mean\"}`*\n"
        "    *Example: to drop rows with missing values in 'Sales': `{\"type\": \"clean\", \"action\": \"handle_missing_values\", \"column\": \"Sales\", \"method\": \"drop_row\"}`*\n"
        "    *Example: to drop rows with any missing values: `{\"type\": \"clean\", \"action\": \"handle_missing_values\", \"column\": \"all_columns\", \"method\": \"drop_row\"}`*\n"
        "11. **To convert a column's data type:**\n"
        "    `ACTION: {\"type\": \"clean\", \"action\": \"convert_datatype\", \"column\": \"[column_name]\", \"target_type\": \"[numeric|datetime|string|int|float]\"}`\n"
        "    *Example: to convert 'Date' to datetime: `{\"type\": \"clean\", \"action\": \"convert_datatype\", \"column\": \"Date\", \"target_type\": \"datetime\"}`*\n"
        "12. **To handle duplicate rows:**\n"
        "    `ACTION: {\"type\": \"clean\", \"action\": \"handle_duplicates\", \"subset\": [\"[column1]\", \"[column2]\"], \"keep\": \"[first|last|false]\"}`\n"
        "    *`subset` is optional. If omitted, all columns are considered. `keep`: 'first' (default), 'last', or `false` (drop all duplicates).*`\n"
        "    *Example: to remove duplicate rows based on 'OrderID' and 'CustomerID': `{\"type\": \"clean\", \"action\": \"handle_duplicates\", \"subset\": [\"OrderID\", \"CustomerID\"], \"keep\": \"first\"}`*\n"
        "13. **To handle outliers in a numeric column (using IQR method):**\n"
        "    `ACTION: {\"type\": \"clean\", \"action\": \"handle_outliers_iqr\", \"column\": \"[numeric_column_name]\", \"method\": \"[remove|cap]\", \"cap_range\": [optional_float_default_1.5]}`\n"
        "    *`method`: 'remove' (drops rows with outliers) or 'cap' (caps outliers at bounds). `cap_range` controls the IQR multiplier (e.g., 1.5 for standard fences).*`\n"
        "    *Example: to remove outliers in 'Price' column: `{\"type\": \"clean\", \"action\": \"handle_outliers_iqr\", \"column\": \"Price\", \"method\": \"remove\"}`*\n"
        "\n**Important Considerations:**\n"
        "-   Always check if columns exist before suggesting an action.\n"
        "-   For numerical operations (like mean/median fill, plotting numeric axes, outlier handling), ensure the column is of a numeric data type. If not, suggest converting it first.\n"
        "-   If the data is unstructured text, you will not use these tools. Instead, answer directly from the text content.\n"
        "-   If the user's request cannot be fulfilled by the tools or direct text extraction, explain why politely.\n"
        "-   **If you provide an ACTION, do NOT include any other text besides the ACTION JSON block.**\n"
        "-   If you provide a direct answer, do NOT include an ACTION block."
        "-   **For direct textual answers or summaries of results, always use clear, natural language. Use Markdown for tables (`tabulate` provides this) and code blocks where appropriate, but avoid raw JSON or internal syntax.**"
    )

    if not conversation_history or conversation_history[0].get("role") != "system":
        conversation_history.insert(0, {"role": "system", "content": system_message})
    else:
        # Update the system message in the history if it has changed
        if conversation_history[0]["content"] != system_message:
            conversation_history[0]["content"] = system_message

    conversation_history.append({"role": "user", "content": user_message})
    
    # Start timing for LLM response
    llm_response_start_time = time.time()

    llm_response_content = get_llm_response(conversation_history)

    # End timing for LLM response
    llm_response_end_time = time.time()
    print(f"DEBUG: LLM chat response generation finished in {llm_response_end_time - llm_response_start_time:.2f} seconds.")

    display_response_for_history = llm_response_content
    # Clean up "ACTION:" prefix if it exists before adding to history
    if "ACTION:" in llm_response_content:
        display_response_for_history = llm_response_content.split("ACTION:", 1)[0].strip()
        if not display_response_for_history: # If only ACTION was present
            display_response_for_history = "Executing requested action..."

    conversation_history.append({"role": "assistant", "content": display_response_for_history})
    set_session_history(conversation_history) # Update history after LLM responds

    action = parse_llm_action(llm_response_content)

    if action:
        # Start timing for action execution
        action_execution_start_time = time.time()

        action_result = execute_llm_action(action)

        action_execution_end_time = time.time()
        print(f"DEBUG: LLM action execution finished in {action_execution_end_time - action_execution_start_time:.2f} seconds.")

        response_text_from_action = action_result["text"]
        plot_url = action_result["plot_url"]

        # Summarize the action result using LLM
        summary_prompt = (
            f"I executed the following action based on your suggestion: {action}\n"
            f"The result was:\n{response_text_from_action}\n\n"
            f"Please summarize this result in a concise, user-friendly way. "
            f"{'A plot was generated and saved as ' + plot_url + '.' if plot_url else ''} "
            f"Explain any limitations or insights clearly. Do not use 'ACTION:' in your response."
            f"Also, confirm the new shape of the DataFrame if it was modified (e.g., 'Now contains X rows and Y columns')."
        )
        current_messages_for_summary = list(conversation_history) # Copy history to avoid modifying it for this temporary prompt
        current_messages_for_summary.append({"role": "user", "content": summary_prompt})
        
        # Start timing for LLM summary
        llm_summary_start_time = time.time()

        summary_response = get_llm_response(current_messages_for_summary)

        # End timing for LLM summary
        llm_summary_end_time = time.time()
        print(f"DEBUG: LLM summary generation finished in {llm_summary_end_time - llm_summary_start_time:.2f} seconds.")

        # Append the summary to the main conversation history
        conversation_history.append({"role": "assistant", "content": summary_response})
        set_session_history(conversation_history)

        return jsonify({
            "response": summary_response,
            "plot_url": plot_url,
            "data_modified": True # Indicate to frontend that data might have changed
        })
    else:
        clean_llm_response = llm_response_content.replace("ACTION:", "").strip()
        return jsonify({
            "response": clean_llm_response,
            "plot_url": None,
            "data_modified": False
        })

@app.route('/get_columns_info', methods=['GET'])
def get_columns_info_route():
    columns_info = get_dataframe_column_dtypes()
    data_type = get_session_data()["type"]
    current_df = get_dataframe_from_session()
    num_rows = len(current_df) if current_df is not None else 0
    num_cols = len(current_df.columns) if current_df is not None else 0
    
    print(f"DEBUG: /get_columns_info requested. Type: {data_type}, Rows: {num_rows}, Cols: {num_cols}, Columns_info: {columns_info}")
    
    return jsonify({"columns_info": columns_info, "data_type": data_type, "rows": num_rows, "cols": num_cols})

@app.route('/generate_custom_plot', methods=['POST'])
def generate_custom_plot_route():
    data = request.json
    plot_type = data.get('plot_type')
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    hue_col = data.get('hue_col')
    title = data.get('title', 'Custom Generated Plot')

    print(f"DEBUG: Custom plot request received: type={plot_type}, x={x_col}, y={y_col}, hue={hue_col}, title={title}")

    if not plot_type:
        return jsonify({"success": False, "message": "Plot type is required."})

    df = get_dataframe_from_session()
    if df is None:
        print("DEBUG: get_dataframe_from_session returned None in generate_custom_plot_route. Cannot plot without structured data.")
        return jsonify({"success": False, "message": "No structured data loaded. Please upload a CSV or XLSX file."})

    plot_filename = f"custom_plot_{uuid.uuid4().hex}.png"
    plot_result = generate_plot(plot_type, x_col, y_col, hue_col, title, plot_filename)

    if plot_result.startswith("Plot saved to: "):
        plot_url = plot_result.split("Plot saved to: ")[1]
        response_message = f"Custom plot '{title}' generated successfully."
        # Update chat history for custom plot too
        conversation_history = get_session_history()
        conversation_history.append({"role": "user", "content": f"Generated custom plot: type={plot_type}, x={x_col}, y={y_col}, hue={hue_col} via UI."})
        conversation_history.append({"role": "assistant", "content": response_message + f" (Plot saved as {plot_url})."})
        set_session_history(conversation_history)
        return jsonify({"success": True, "message": response_message, "plot_url": plot_url})
    else:
        print(f"DEBUG: Plot generation failed with message: {plot_result}")
        return jsonify({"success": False, "message": plot_result})

@app.route('/apply_cleaning_action', methods=['POST'])
def apply_cleaning_action_route():
    data = request.json
    action_type = data.get('action_type') # e.g., "handle_missing_values"
    column = data.get('column')
    method = data.get('method')
    value = data.get('value')
    subset = data.get('subset')
    keep = data.get('keep')
    cap_range = data.get('cap_range')
    target_type = data.get('target_type')

    print(f"DEBUG: Cleaning action received: type={action_type}, column={column}, method={method}, value={value}")

    df = get_dataframe_from_session()
    if df is None:
        return jsonify({"success": False, "message": "No structured data loaded. Please upload a CSV or XLSX file."})

    result = {"success": False, "message": "Invalid cleaning action."}
    if action_type == "handle_missing_values":
        result = handle_missing_values(column, method, value)
    elif action_type == "convert_datatype":
        result = convert_datatype(column, target_type)
    elif action_type == "handle_duplicates":
        result = handle_duplicates_df(subset, keep)
    elif action_type == "handle_outliers_iqr":
        result = handle_outliers_iqr(column, method, cap_range)

    # After any successful cleaning, fetch and return updated column info
    if result["success"]:
        updated_columns_info = get_dataframe_column_dtypes()
        current_df = get_dataframe_from_session()
        result["new_rows"] = len(current_df) if current_df is not None else 0
        result["new_cols"] = len(current_df.columns) if current_df is not None else 0
        result["columns_info"] = updated_columns_info

        # Add to chat history that a cleaning action was performed by UI
        conversation_history = get_session_history()
        user_msg = f"Performed UI action: {action_type} on {column if column else 'data'} using method {method if method else ''}."
        if value: user_msg += f" (Value: {value})"
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": f"Data cleaning applied: {result['message']}. DataFrame now has {result['new_rows']} rows and {result['new_cols']} columns."})
        set_session_history(conversation_history)

    return jsonify(result)

@app.route('/download_data', methods=['GET'])
def download_data():
    df = get_dataframe_from_session()
    if df is None:
        return jsonify({"success": False, "message": "No data loaded or processed to download."}), 400

    try:
        # Determine filename and format
        filename = f"processed_data_{uuid.uuid4().hex}.csv" # Default to CSV
        file_path = os.path.join(app.config['TEMP_DATA_FOLDER'], filename)

        # You can add logic here to allow choosing CSV or XLSX based on a query parameter
        # For simplicity, let's just use CSV for now.
        df.to_csv(file_path, index=False)

        return send_file(file_path, as_attachment=True, download_name="processed_data.csv", mimetype='text/csv')
    except Exception as e:
        print(f"ERROR: Failed to prepare data for download: {e}")
        return jsonify({"success": False, "message": f"Error preparing data for download: {e}"}), 500


# Cleanup of temp_data folder on app exit or before restart (optional)
# This will remove all temporary data files when the Flask app starts
# You might want a more sophisticated cleanup for production (e.g., cron job)

def setup_app():
    # Clear temp_data folder on startup to ensure fresh state
    if os.path.exists(app.config['TEMP_DATA_FOLDER']):
        try:
            shutil.rmtree(app.config['TEMP_DATA_FOLDER'])
            print(f"INFO: Cleared existing temp data folder: {app.config['TEMP_DATA_FOLDER']}")
        except Exception as e:
            print(f"WARNING: Could not clear temp data folder on startup: {e}")
    os.makedirs(app.config['TEMP_DATA_FOLDER'], exist_ok=True)
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=port)
