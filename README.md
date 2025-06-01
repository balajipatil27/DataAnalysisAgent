# Data Analyst Agent

A Flask-based web application that serves as an intelligent data analyst assistant. Users can upload various file types (CSV, Excel, text documents), chat with an AI agent to get insights, generate visualizations, perform data cleaning operations, and conduct advanced statistical analyses like correlation and linear regression.

## Features

* **File Upload:** Upload `CSV`, `XLSX` (Excel), and `TXT` files.
    * For structured data (`CSV`, `XLSX`), the agent loads it into a Pandas DataFrame.
    * For unstructured data (`TXT`), the agent provides a summary.
  ![WhatsApp Image 2025-06-01 at 09 40 25_245df6b4](https://github.com/user-attachments/assets/efb53f07-744a-4c62-ba6d-98d5da518ed9)

* **Intelligent Chat Assistant:** Interact with a Google Gemini-powered AI model to:
    * Get an overview of your uploaded data or document.
    * Ask questions about your data.
    * Request specific plots (e.g., "show me a bar chart of sales by region").
    * Receive guidance on data cleaning.
    * **New:** Get interpretations of statistical analysis results (e.g., regression summaries).
  ![WhatsApp Image 2025-06-01 at 09 41 06_17197cc3](https://github.com/user-attachments/assets/3b6caaaf-e8de-49ec-8850-5b7e4ac6783f)

* **Custom Plot Generation:** Manually select columns and plot types (Bar, Line, Scatter, Histogram, Box) to generate custom visualizations.
  ![WhatsApp Image 2025-06-01 at 09 41 35_7028059a](https://github.com/user-attachments/assets/2f628ff4-44f4-4632-bd9b-672f8f14792d)
  ![WhatsApp Image 2025-06-01 at 09 42 49_2ebbc655](https://github.com/user-attachments/assets/c40bed1b-4489-4023-9bbb-5d3e083fa84c)

* **Data Cleaning & Preprocessing:**
    * **Missing Values:** Handle missing values by dropping rows/columns, or filling with mean, median, mode, or a specific value.
    * **Data Type Conversion:** Convert column data types to numeric, integer, datetime, or string.
    * **Duplicate Handling:** Remove duplicate rows, optionally based on a subset of columns, and specify which occurrence to keep.
    * **Outlier Handling (IQR Method):** Identify and handle outliers in numeric columns by removal or capping values using the Interquartile Range method.
  ![WhatsApp Image 2025-06-01 at 09 43 51_3c3236d1](https://github.com/user-attachments/assets/0521a045-0cd8-48eb-94ef-d18825df5c30)


* **Download Processed Data:** Download your cleaned and processed DataFrame as a CSV file.
  ![WhatsApp Image 2025-06-01 at 09 44 24_cf65d8d2](https://github.com/user-attachments/assets/3e1896a3-d706-4240-bee6-c1643632e5ea)

## Technologies Used

* **Backend:** Flask (Python)
* **Data Manipulation:** Pandas
* **Numerical Operations:** NumPy
* **Statistical Modeling:**
    * Scikit-learn (for potential future basic models, though `statsmodels` is used for regression summary here)
    * Statsmodels (for robust statistical model summaries in regression)
* **Visualization:** Matplotlib, Seaborn
* **AI/LLM:** Google Gemini Pro API
* **Frontend:** HTML, CSS (Bootstrap 5), JavaScript (for AJAX interactions and dynamic UI)
* **Markdown Rendering:** Marked.js

## Setup and Installation

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### 1. Clone the Repository

```bash
git clone <repository-url> # Replace with your actual repository URL
cd dataanalystagent
