#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import boto3


# # Good Companies - Stats By Stage Calculation

# <hr style="border: 3px solid black;">

# In[16]:


# ---------------------------------------------------------------------
# PART 1: Generate stats_by_stage from company data analysis
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from io import StringIO
import boto3
from botocore.exceptions import ClientError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# ---------------- AWS helpers ----------------
s3 = boto3.client('s3')
BUCKET_NAME   = "good-companies"
CSV_FILE_KEY  = "affinity_good_companies_cleaned.csv"

def read_csv_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return pd.DataFrame()
        raise

def write_csv_to_s3(df, bucket, key):
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

# --------------- cleaning helpers ------------
def clean_most_recent_funding_date(date_str, date_evaluated):
    """
    Robust date‐parser that normalises 'Last Funding Date'.
    Returns days between date_evaluated and the funding date, or np.nan.
    """
    try:
        # Parse funding date
        date = pd.to_datetime(date_str, errors="coerce")
        # fix 2-digit years like 3/10/22
        if pd.notna(date) and date.year < 2000:
            date = pd.to_datetime(date_str, format="%m/%d/%y", errors="coerce")
        
        # Parse evaluation date
        eval_date = pd.to_datetime(date_evaluated, errors="coerce")
        if pd.notna(eval_date) and eval_date.year < 2000:
            eval_date = pd.to_datetime(date_evaluated, format="%m/%d/%y", errors="coerce")
        
        # Calculate days difference if both dates are valid
        if pd.notna(date) and pd.notna(eval_date):
            return (eval_date - date).days
        return np.nan
    except Exception:
        return np.nan

def clean_founded_year(val, date_evaluated):
    """
    Accepts anything (YYYY, 'YYYY-MM-DD', etc.) and
    returns days between date_evaluated and founded date, or np.nan.
    """
    try:
        # Parse evaluation date
        eval_date = pd.to_datetime(date_evaluated, errors="coerce")
        if pd.notna(eval_date) and eval_date.year < 2000:
            eval_date = pd.to_datetime(date_evaluated, format="%m/%d/%y", errors="coerce")
        
        if pd.isna(eval_date):
            return np.nan
        
        # Parse founded date
        # direct numeric year?
        yr = pd.to_numeric(val, errors="coerce")
        if pd.notna(yr):
            yr = int(yr)
            # Convert year to January 1st of that year
            founded_date = pd.Timestamp(year=yr, month=1, day=1)
        else:
            # try full date
            founded_date = pd.to_datetime(val, errors="coerce")
            if pd.notna(founded_date):
                yr = founded_date.year
            else:
                return np.nan
        
        # sanity check plausible range
        if yr < 1900 or yr > eval_date.year:
            return np.nan
            
        # Calculate days difference
        return (eval_date - founded_date).days
    except Exception:
        return np.nan

# ------------------ load & split --------------
df = read_csv_from_s3(BUCKET_NAME, CSV_FILE_KEY)
df["Last Funding Type"] = df["Last Funding Type"].str.upper()

seed_companies     = df[df["Last Funding Type"].isin(["SEED", "PRE_SEED"])]
series_a_companies = df[df["Last Funding Type"].isin(["SERIES_A"])]
series_b_companies = df[df["Last Funding Type"] == "SERIES_B"]

# ------------- metric registration -----------
numerical_cols = [
    # existing
    "Headcount % (365d)", "Headcount (Sales) % (365d)",
    "Headcount % (180d)", "Web Traffic % (365d)",
    "Web Traffic % (180d)", "Last Funding Total",
    "Funding Total", "Last Round Post Val",
    "HeadCount", "Days Before Last Funding",
    "Headcount (Sales) % (180d)", "Headcount % (2 Years)",
    "HeadCount Engineering_1Y_Growth_Percent",
    "HeadCount Operations_1Y_Growth_Percent",
    "LinkedIn Followers_1Y_Growth_Percent",
    "Days Since Founded",  # NEW
]

columns = [
    "HeadCount", "Last Funding Total", "Funding Total",
    "Headcount % (365d)", "Headcount % (180d)", "Headcount % (2 Years)",
    "Web Traffic % (365d)", "Web Traffic % (180d)", "Last Round Post Val",
    "Headcount (Sales) % (365d)", "Headcount (Sales) % (180d)",
    "Days Before Last Funding",
    "HeadCount Engineering_1Y_Growth_Percent",
    "HeadCount Operations_1Y_Growth_Percent",
    "LinkedIn Followers_1Y_Growth_Percent",
    "Days Since Founded",  # NEW
]

column_name_mapping = {
    # existing
    "HeadCount": "Total Employees",
    "Last Funding Total": "Last Funding Amount",
    "Funding Total": "Total Raised",
    "Headcount % (365d)": "1 Year Headcount Growth %",
    "Headcount % (180d)": "180d Headcount Growth %",
    "Headcount % (2 Years)": "2 Year Headcount Growth %",
    "Web Traffic % (365d)": "1 Year Web Traffic Growth %",
    "Web Traffic % (180d)": "180d Web Traffic Growth %",
    "Last Round Post Val": "Last Round Valuation",
    "Headcount (Sales) % (365d)": "1 Year Sales %",
    "Headcount (Sales) % (180d)": "180d Sales %",
    "Days Before Last Funding": "Days Before Last Funding",
    "HeadCount Engineering_1Y_Growth_Percent": "1 Year Engineering Headcount %",
    "HeadCount Operations_1Y_Growth_Percent": "1 Year Operations Headcount %",
    "LinkedIn Followers_1Y_Growth_Percent": "1 Year Linkedin Followers %",
    "Days Since Founded": "Days Since Founded",  # NEW
}

# ------------- core computation --------------
def process_funding_round(df_group, eps=1.5, min_samples=2):
    """
    Cleans, clusters, and summarises a subset (SEED / A / B).
    """
    df_group = df_group.copy()

    # -- clean dates & derived fields -----------------------
    # Calculate days using Date Evaluated as reference
    if "Date Evaluated" in df_group.columns and "Last Funding Date" in df_group.columns:
        df_group["Days Before Last Funding"] = df_group.apply(
            lambda row: clean_most_recent_funding_date(row["Last Funding Date"], row["Date Evaluated"]), 
            axis=1
        )
        df_group.drop(columns=["Last Funding Date"], inplace=True)

    # Calculate days since founded using Date Evaluated as reference
    if "Date Evaluated" in df_group.columns:
        if "Founded Year" in df_group.columns:
            df_group["Days Since Founded"] = df_group.apply(
                lambda row: clean_founded_year(row["Founded Year"], row["Date Evaluated"]), 
                axis=1
            )
        elif "Founding Date" in df_group.columns:
            df_group["Days Since Founded"] = df_group.apply(
                lambda row: clean_founded_year(row["Founding Date"], row["Date Evaluated"]), 
                axis=1
            )

    # -- numeric conversion and infinity handling -----------
    for col in numerical_cols:
        if col in df_group.columns:
            df_group[col] = pd.to_numeric(df_group[col], errors="coerce")
            df_group[col] = df_group[col].replace([np.inf, -np.inf], np.nan)

    # -- impute / scale / PCA -------------------------------
    available_numerical_cols = [col for col in numerical_cols if col in df_group.columns]
    
    if not available_numerical_cols:
        print("Warning: No numerical columns available for processing")
        return {}
    
    imputer      = SimpleImputer(strategy="median")
    df_imputed   = imputer.fit_transform(df_group[available_numerical_cols])
    scaler       = StandardScaler()
    scaled_data  = scaler.fit_transform(df_imputed)

    pca   = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # -- feature weights from PCA ---------------------------
    loadings    = np.abs(pca.components_)
    weights_raw = loadings.sum(axis=0)
    weights_raw /= weights_raw.mean()       # average weight == 1
    weights_dict = dict(zip(available_numerical_cols, np.round(weights_raw, 2)))

    # FORCE Days Since Founded weight = 1 if it exists
    if "Days Since Founded" in weights_dict:
        weights_dict["Days Since Founded"] = 1.0

    # -- DBSCAN, keep cluster 0 -----------------------------
    clusters      = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pca_data)
    df_group["Cluster"] = clusters
    cluster0 = df_group[df_group["Cluster"] == 0].copy()
    if cluster0.empty:
        print("Warning: Cluster 0 empty, using full group.")
        cluster0 = df_group.copy()

    # -- build summary stats --------------------------------
    summary = {}
    for col in columns:
        if col in cluster0.columns:
            friendly = column_name_mapping[col]
            cluster0[col] = pd.to_numeric(cluster0[col], errors="coerce")
            summary[friendly] = {
                "5th":    round(cluster0[col].quantile(0.05), 2),
                "95th":   round(cluster0[col].quantile(0.95), 2),
                "median": round(cluster0[col].median(),       2),
            }

    summary["weights"] = weights_dict
    return summary

# --------------- run for each stage -----------------------
seed_summary     = process_funding_round(seed_companies,     eps=1.5, min_samples=2)
series_a_summary = process_funding_round(series_a_companies, eps=1.5, min_samples=2)
series_b_summary = process_funding_round(series_b_companies, eps=1.5, min_samples=3)

stats_by_stage = {
    "SEED":      seed_summary,
    "SERIES A":  series_a_summary,
    "SERIES B":  series_b_summary,
}

# --- convert weight keys to friendly names & store top-9 ------------
for stage, d in stats_by_stage.items():
    # 1) map raw → friendly names
    d["weights"] = {
        column_name_mapping.get(k, k): v
        for k, v in d["weights"].items()
    }
    # 2) capture the names of the 9 largest weights for quick lookup
    d["top9"] = [
        k for k, _ in sorted(
            d["weights"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:9]
    ]


# In[17]:


stats_by_stage


# <hr style="border: 3px solid black;">
# 

# # Model Code

# In[33]:


# ---------------------------------------------------------------------
# PART 2: Use stats_by_stage for company scoring
# ---------------------------------------------------------------------

# ---------------------------
# SCORING HELPER FUNCTIONS
# ---------------------------
def custom_scoring(value, percentile_5, avg, percentile_95, max_score):
    """
    Output is clamped between 0 and max_score.
    Quadratic decrease from the 5th percentile to the average,
    and linear decrease from the average to the 95th percentile.
    """
    y1_max = max_score        # Score at the 5th percentile
    y1_min = max_score * 0.6  # Score at the average (e.g., 60% of max score)
    y2_min = 0                # Score at the 95th percentile

    if value <= avg:
        x_ratio = (value - percentile_5) / (avg - percentile_5) if avg != percentile_5 else 0
        x_ratio = max(0, min(x_ratio, 1))
        y = y1_max - (y1_max - y1_min) * (x_ratio ** 2)
    elif value <= percentile_95:
        x_ratio = (percentile_95 - value) / (percentile_95 - avg) if percentile_95 != avg else 0
        x_ratio = max(0, min(x_ratio, 1))
        y = y1_min * x_ratio
    else:
        y = y2_min
    return max(y2_min, min(y, y1_max))


def linear_decay(value, avg, percentile_5, percentile_95, max_score):
    """
    Output is clamped between 0 and max_score.
    If value <= avg => scale linearly up to max_score near the 5th percentile,
    If value > avg  => scale linearly down to 0 near the 95th percentile.
    """
    if value <= avg:
        denom = (avg - percentile_5) or 1e-9
        score = max_score - ((avg - value) / denom) * max_score
        return max(0, min(score, max_score))
    else:
        denom = (percentile_95 - avg) or 1e-9
        score = max_score - ((value - avg) / denom) * max_score
        return max(0, min(score, max_score))

# ---- Inverted percentile helpers (high value = high score) ----
# ------------------------------------------------------------------
def custom_scoring_inv(value, p5, avg, p95, max_score):
    """
    Linear scale:
        • max_score at/above the 95th percentile
        • 0 at/below the 5th percentile
        • straight‐line interpolation in between
    """
    if value >= p95:      # top performers
        return max_score
    if value <= p5:       # bottom performers
        return 0

    # linear interpolation between p5 → p95
    return (value - p5) / (p95 - p5) * max_score


def custom_scoring_inv_raw(value, p5, avg, p95):
    """Same linear rule, but on a 0‑1 scale."""
    return custom_scoring_inv(value, p5, avg, p95, 1.0)


# ---------------------------
# Helper: Get percentiles and average
# ---------------------------
def get_percentiles_and_avg(stage_data, metric_name):
    """
    Returns (p5, p95, avg) for the given metric_name.
    Uses 'median' as the average.
    """
    metric_stats = stage_data.get(metric_name, {})
    p5 = metric_stats.get('5th', 0)
    p95 = metric_stats.get('95th', 0)
    avg_val = metric_stats.get('median', 0)
    return p5, p95, avg_val


# ---------------------------
# Helper: Get metric weight (no fallback; assumes key exists)
# ---------------------------
def get_metric_weight(stage_data, metric_name):
    return 10.0 * stage_data['weights'][metric_name]


# Helper function to evaluate offshore data criteria
def evaluate_offshore_data(offshore_str):
    if not isinstance(offshore_str, str) or offshore_str.strip() == '':
        return False  # Skip empty or non-string values
    
    western_countries = ['United States', 'USA', 'U.S.A.', 'Israel', 'Canada', 'United Kingdom', 'UK', 'U.K.']
    
    # Parse the offshore data string into country-percentage pairs
    country_percentages = {}
    try:
        parts = offshore_str.split(';')
        for part in parts:
            country_info = part.strip()
            if ':' in country_info and '(' in country_info and ')' in country_info:
                country = country_info.split(':')[0].strip()
                percentage_str = country_info.split('(')[1].split(')')[0].replace('%', '').strip()
                percentage = float(percentage_str)
                country_percentages[country] = percentage
    except Exception:
        return False  # If parsing fails, skip this entry
    
    # Calculate total percentage for western countries
    western_percentage = sum(country_percentages.get(country, 0) 
                            for country in western_countries 
                            if country in country_percentages)
    
    # Check if western countries percentage is too low (≤35%)
    if western_percentage <= 35:
        # Check if India has ≥40%
        if country_percentages.get('India', 0) >= 40:
            return False  # Exclude this company
        
        # Calculate total percentage for all non-western countries
        non_western_percentage = sum(percentage
                                    for country, percentage in country_percentages.items()
                                    if country not in western_countries)
        
        # Check if combined non-western percentage is ≥70%
        if non_western_percentage >= 70:
            return False  # Exclude this company
            
        # Keep the original check for any single non-western country having ≥70%
        for country, percentage in country_percentages.items():
            if country not in western_countries and country != 'India' and percentage >= 70:
                return False  # Exclude this company
    
    # If we passed all exclusion criteria
    return True

# NEW FUNCTIONS FOR RAW SCORES
def custom_scoring_raw(value, percentile_5, avg, percentile_95):
    """
    Returns raw score on a 0-1 scale without applying weight.
    Quadratic decrease from the 5th percentile to the average,
    and linear decrease from the average to the 95th percentile.
    """
    y1_max = 1.0        # Score at the 5th percentile
    y1_min = 0.6        # Score at the average (e.g., 60% of max score)
    y2_min = 0          # Score at the 95th percentile

    if value <= avg:
        x_ratio = (value - percentile_5) / (avg - percentile_5) if avg != percentile_5 else 0
        x_ratio = max(0, min(x_ratio, 1))
        y = y1_max - (y1_max - y1_min) * (x_ratio ** 2)
    elif value <= percentile_95:
        x_ratio = (percentile_95 - value) / (percentile_95 - avg) if percentile_95 != avg else 0
        x_ratio = max(0, min(x_ratio, 1))
        y = y1_min * x_ratio
    else:
        y = y2_min
    return max(y2_min, min(y, y1_max))


def linear_decay_raw(value, avg, percentile_5, percentile_95):
    """
    Returns raw score on a 0-1 scale without applying weight.
    If value <= avg => scale linearly up to 1.0 near the 5th percentile,
    If value > avg  => scale linearly down to 0 near the 95th percentile.
    """
    if value <= avg:
        denom = (avg - percentile_5) or 1e-9
        score = 1.0 - ((avg - value) / denom) * 1.0
        return max(0, min(score, 1.0))
    else:
        denom = (percentile_95 - avg) or 1e-9
        score = 1.0 - ((value - avg) / denom) * 1.0
        return max(0, min(score, 1.0))


def location_kill_metric(row):
    """
    Returns a penalty based on the company's location (using up to three columns):
      - If any location is in the safe list, returns 0 (no penalty).
      - Else if any location is in the moderate list, returns -25.
      - Else if at least one location is provided but not in the above lists, returns -50.
      - If none of the three location columns has data, returns 0.
    """
    location_columns = [
        'Deal Room Location (Country)',
        'Affinity Location (Country)',
        'HQ Location (Country)'
    ]
    locations = [row.get(col) for col in location_columns if pd.notna(row.get(col))]

    safe_countries = ['United States', 'Israel', 'Canada', 'United Kingdom']
    moderate_countries = [
        'Austria', 'Belgium', 'France', 'Germany', 'Netherlands', 'Switzerland',
        'Ireland', 'Spain', 'Portugal', 'Italy', 'Czech Republic', 'Poland',
        'Slovakia', 'Slovenia', 'Croatia', 'Ukraine'
    ]

    if locations:
        if any(loc in safe_countries for loc in locations):
            return 0
        elif any(loc in moderate_countries for loc in locations):
            return -25
        else:
            return -50
    else:
        return 0
    
    
# ---------------------------------------------------------------------
# 3)  NEW  →  helper that says “is the metric value present?”
# ---------------------------------------------------------------------
# Map friendly metric names → dataframe column names
metric_to_column = {
    'Total Employees'              :'Total Employees',
    'Last Funding Amount'          :'Last Funding Amount',
    'Total Raised'                 :'Total Raised',
    '1 Year Headcount Growth %'    :'1 Year Headcount △%',
    '180d Headcount Growth %'      :'180d Headcount △%',
    '2 Year Headcount Growth %'    :'2 Year Headcount Growth',
    '180d Web Traffic Growth %'    :'180d Web Traffic △%',
    '1 Year Web Traffic Growth %'  :'1 Year Web Traffic △%',
    'Last Round Valuation'         :'Last Round Valuation',
    '1 Year Sales %'               :'1 Year Sales △%',
    '180d Sales %'                 :'180d Sales Growth △%',
    'Days Before Last Funding'     :'Most Recent Funding',   # date used indirectly
    '1 Year Engineering Headcount %' : '1 Year Engineering Headcount △%',
    '1 Year Operations Headcount %'  : '1 Year Operations Headcount △%',
    '1 Year Linkedin Followers %'    : '1 Year Linkedin Followers △%',
    'Days Since Founded'                   : 'Founded Date'          # raw integer year

}

def metric_is_missing(row, metric_name, has_lr_val):
    """True if the metric is absent or zero."""
    if metric_name == 'Last Round Valuation':
        return not has_lr_val
    col = metric_to_column.get(metric_name)
    if col is None:
        return True
    val = row.get(col)
    # treat 0 (numeric or string) as missing
    try:
        numeric_val = pd.to_numeric(val)
    except Exception:
        numeric_val = np.nan
    return pd.isna(val) or numeric_val == 0


# ---------------------------
# Helper: Get percentiles and average
# ---------------------------
def get_percentiles_and_avg(stage_data, metric_name):
    """
    Returns (p5, p95, avg) for the given metric_name.
    Uses 'median' as the average.
    """
    metric_stats = stage_data.get(metric_name, {})
    p5 = metric_stats.get('5th', 0)
    p95 = metric_stats.get('95th', 0)
    avg_val = metric_stats.get('median', 0)
    return p5, p95, avg_val


# ---------------------------
# Helper: Get metric weight (no fallback; assumes key exists)
# ---------------------------
def get_metric_weight(stage_data, metric_name):
    return 10.0 * stage_data['weights'][metric_name]


# ---------------------------
# MAIN SCORE CALCULATION FUNCTION
# ---------------------------
def calculate_score(row):
    """
    Calculates a score for `row` using dictionary weights and static metrics.
    
    The stage is determined from the 'Last Round' column. For that stage,
    the dictionary-based total possible points equal the sum of (10×weight) for
    each metric in the stage's dictionary. In addition, four static metrics contribute
    a fixed 25 points (Location Score: 10, Founded Year Score: 10, Business Model Score: 5,
    and Kill Metrics Score is best-case 0). Thus, the overall Total Possible Points is:
    
         dict_total_possible + 25
    
    Each stage (SEED, SERIES A, SERIES B) will have a distinct total possible points value.
    """
    
    # ADD THIS CHECK AT THE VERY BEGINNING:
    # Check if Last Round is VENTURE_UNKNOWN and return zero score immediately
    last_round_val = row.get('Last Round', "")
    if pd.notna(last_round_val) and str(last_round_val).upper() == 'VENTURE_UNKNOWN':
        # Return a dictionary with all scores set to 0
        zero_subscores = {
            'Kill Metrics Score': 0,
            'Small Company Penalty': 0,
            'Location Kill Score': 0,
            'Offshore Score': 0,
            'Location Score': 0,
            'Founded Year Score': 0,
            'Business Model Score': 0,
            'Total Employees Score': 0,
            'Last Funding Amount Score': 0,
            'Total Raised Score': 0,
            '1 Year Headcount Growth Score': 0,
            '180d Headcount Growth Score': 0,
            '2 Year Headcount Growth Score': 0,
            '180d Web Traffic Growth Score': 0,
            '1 Year Web Traffic Growth Score': 0,
            'Last Round Valuation Score': 0,
            '1 Year Sales Growth Score': 0,
            '180d Sales Growth Score': 0,
            'Last Fundraise Date Score': 0,
            '1 Year Engineering Headcount Growth Score': 0,
            '1 Year Operations Headcount Growth Score': 0,
            '1 Year Linkedin Followers Growth Score': 0,
            'Total Score': 0,
            'Normalized Score': 0,
            'Total Possible Points': 100,  # Set a reasonable denominator
            'Scoring Explanation': "Score is 0.0 => VENTURE_UNKNOWN funding round",
            'Last Fundraise Date Score (x10)': 0,
            'Total Raise Penalty': 0
        }
        
        # Add all raw scores as 0
        raw_score_keys = [
            'Kill Metrics Raw Score', 'Small Company Penalty Raw Score', 'Location Kill Raw Score',
            'Offshore Raw Score', 'Location Raw Score', 'Founded Year Raw Score', 'Business Model Raw Score',
            'Total Employees Raw Score', 'Last Funding Amount Raw Score', 'Total Raised Raw Score',
            '1 Year Headcount Growth Raw Score', '180d Headcount Growth Raw Score', '2 Year Headcount Growth Raw Score',
            '180d Web Traffic Growth Raw Score', '1 Year Web Traffic Growth Raw Score', 'Last Round Valuation Raw Score',
            '1 Year Sales Growth Raw Score', '180d Sales Growth Raw Score', 'Last Fundraise Date Raw Score',
            '1 Year Engineering Headcount Growth Raw Score', '1 Year Operations Headcount Growth Raw Score',
            '1 Year Linkedin Followers Growth Raw Score', 'Days Since Founded Raw Score', 'Total Raise Penalty Raw Score'
        ]
        
        for key in raw_score_keys:
            zero_subscores[key] = 0
            
        return zero_subscores
    # END OF NEW CHECK
    
    
    # NEW CHECK: Set score to 0 for any funding round that's not PRE_SEED, SEED, SERIES_A, or SERIES_B
    valid_rounds = {'PRE_SEED', 'SEED', 'SERIES_A', 'SERIES_B'}
    if pd.notna(last_round_val):
        upper_round = str(last_round_val).upper()
        # Check if the round contains any of our valid stages
        is_valid_round = any(valid_stage in upper_round for valid_stage in valid_rounds)

        if not is_valid_round:
            # Return a dictionary with all scores set to 0
            zero_subscores = {
                'Kill Metrics Score': 0,
                'Small Company Penalty': 0,
                'Location Kill Score': 0,
                'Offshore Score': 0,
                'Location Score': 0,
                'Founded Year Score': 0,
                'Business Model Score': 0,
                'Total Employees Score': 0,
                'Last Funding Amount Score': 0,
                'Total Raised Score': 0,
                '1 Year Headcount Growth Score': 0,
                '180d Headcount Growth Score': 0,
                '2 Year Headcount Growth Score': 0,
                '180d Web Traffic Growth Score': 0,
                '1 Year Web Traffic Growth Score': 0,
                'Last Round Valuation Score': 0,
                '1 Year Sales Growth Score': 0,
                '180d Sales Growth Score': 0,
                'Last Fundraise Date Score': 0,
                '1 Year Engineering Headcount Growth Score': 0,
                '1 Year Operations Headcount Growth Score': 0,
                '1 Year Linkedin Followers Growth Score': 0,
                'Total Score': 0,
                'Normalized Score': 0,
                'Total Possible Points': 100,
                'Scoring Explanation': f"Score is 0.0 => {last_round_val} is not a valid funding round",
                'Last Fundraise Date Score (x10)': 0,
                'Total Raise Penalty': 0

            }

            # Add all raw scores as 0
            raw_score_keys = [
                'Kill Metrics Raw Score', 'Small Company Penalty Raw Score', 'Location Kill Raw Score',
                'Offshore Raw Score', 'Location Raw Score', 'Founded Year Raw Score', 'Business Model Raw Score',
                'Total Employees Raw Score', 'Last Funding Amount Raw Score', 'Total Raised Raw Score',
                '1 Year Headcount Growth Raw Score', '180d Headcount Growth Raw Score', '2 Year Headcount Growth Raw Score',
                '180d Web Traffic Growth Raw Score', '1 Year Web Traffic Growth Raw Score', 'Last Round Valuation Raw Score',
                '1 Year Sales Growth Raw Score', '180d Sales Growth Raw Score', 'Last Fundraise Date Raw Score',
                '1 Year Engineering Headcount Growth Raw Score', '1 Year Operations Headcount Growth Raw Score',
                '1 Year Linkedin Followers Growth Raw Score', 'Days Since Founded Raw Score', 'Total Raise Penalty Raw Score'
            ]

            for key in raw_score_keys:
                zero_subscores[key] = 0

            return zero_subscores
    # END OF VALID ROUNDS CHECK
    
    subscores = {
        'Kill Metrics Score': 0,
        'Small Company Penalty': 0,
        'Location Kill Score': 0,
        'Offshore Score': 0,
        'Location Score': 0,
        'Founded Year Score': 0,
        'Business Model Score': 0,
        'Total Employees Score': 0,
        'Last Funding Amount Score': 0,
        'Total Raised Score': 0,
        '1 Year Headcount Growth Score': 0,
        '180d Headcount Growth Score': 0,
        '2 Year Headcount Growth Score': 0,
        '180d Web Traffic Growth Score': 0,
        '1 Year Web Traffic Growth Score': 0,
        'Last Round Valuation Score': 0,
        '1 Year Sales Growth Score': 0,
        '180d Sales Growth Score': 0,
        'Last Fundraise Date Score': 0,
        '1 Year Engineering Headcount Growth Score': 0,
        '1 Year Operations Headcount Growth Score': 0,
        '1 Year Linkedin Followers Growth Score': 0,
        'Total Raise Penalty': 0
    }
    
    # Initialize raw scores dictionary
    raw_scores = {
        'Kill Metrics Raw Score': 0,
        'Small Company Penalty Raw Score': 0,
        'Location Kill Raw Score': 0,
        'Offshore Raw Score': 0,
        'Location Raw Score': 0,
        'Founded Year Raw Score': 0,
        'Business Model Raw Score': 0,
        'Total Employees Raw Score': 0,
        'Last Funding Amount Raw Score': 0,
        'Total Raised Raw Score': 0,
        '1 Year Headcount Growth Raw Score': 0,
        '180d Headcount Growth Raw Score': 0,
        '2 Year Headcount Growth Raw Score': 0,
        '180d Web Traffic Growth Raw Score': 0,
        '1 Year Web Traffic Growth Raw Score': 0,
        'Last Round Valuation Raw Score': 0,
        '1 Year Sales Growth Raw Score': 0,
        '180d Sales Growth Raw Score': 0,
        'Last Fundraise Date Raw Score': 0,
        '1 Year Engineering Headcount Growth Raw Score': 0,
        '1 Year Operations Headcount Growth Raw Score': 0,
        '1 Year Linkedin Followers Growth Raw Score': 0,
        'Total Raise Penalty Raw Score': 0
    }
    
    # --- Static Metrics ---
    # 1) Kill Metrics by Investors
    if pd.notna(row.get('Current Investors')):
        investors = [inv.strip() for inv in row['Current Investors'].split(';')]
        if any(inv in ['Sequoia Capital', 'Benchmark', 'Thrive Capital'] for inv in investors):
            raw_scores['Kill Metrics Raw Score'] = -10.0  # Full negative score
            subscores['Kill Metrics Score'] -= 0
        elif any(inv in ['Battery Ventures', 'Andreessen Horowitz', 'Lightspeed Venture Partners'] for inv in investors):
            raw_scores['Kill Metrics Raw Score'] = -0.1  # Partial negative score
            subscores['Kill Metrics Score'] -= 0

    # 2) Location Kill Score (penalty; not included in static possible)
    kill_score = location_kill_metric(row)
    subscores['Location Kill Score'] = kill_score
    if kill_score == 0:
        raw_scores['Location Kill Raw Score'] = 0
    elif kill_score == -25:
        raw_scores['Location Kill Raw Score'] = -0.25
    elif kill_score == -50:
        raw_scores['Location Kill Raw Score'] = -0.5

    # 3) Location Score (static; max 10)
    if pd.notna(row.get('HQ Location (Country)')):
        country = row['HQ Location (Country)']
        if country == 'United States':
            raw_scores['Location Raw Score'] = 1.0  # 10/10
            subscores['Location Score'] += 10
        elif country in ['Israel', 'Canada']:
            raw_scores['Location Raw Score'] = 0.6  # 6/10
            subscores['Location Score'] += 6
        elif country in ['United Kingdom', 'Australia', 'UK']:
            raw_scores['Location Raw Score'] = 0.4  # 4/10
            subscores['Location Score'] += 4
        elif country in ['India', 'Germany', 'France']:
            raw_scores['Location Raw Score'] = 0.0
            subscores['Location Score'] += 0

    # 4) Founded Year Score (static; max 10)
    # if pd.notna(row.get('Founded Date')):
    #     founded_year = pd.to_datetime(row['Founded Date'], errors='coerce').year
    #     if pd.notna(founded_year):
    #         current_year = pd.Timestamp.now().year
    #         years_since_founded = current_year - founded_year
    #         if 3 <= years_since_founded <= 6:
    #             raw_scores['Founded Year Raw Score'] = 1.0  # 10/10
    #             subscores['Founded Year Score'] += 10
    #         elif 6 < years_since_founded <= 8:
    #             raw_scores['Founded Year Raw Score'] = 0.5  # 5/10
    #             subscores['Founded Year Score'] += 5
    #         elif 2 <= years_since_founded < 3:
    #             raw_scores['Founded Year Raw Score'] = 0.0  # 0/10
    #             subscores['Founded Year Score'] += 0
    #         elif years_since_founded < 2:
    #             raw_scores['Founded Year Raw Score'] = -0.5  # -5/10
    #             subscores['Founded Year Score'] -= 5
    #         elif years_since_founded > 8:
    #             raw_scores['Founded Year Raw Score'] = -1.0  # -10/10
    #             subscores['Founded Year Score'] -= 10

        # ────────── Founded Year (percentile-based MAIN metric) ──────────



    # 5) Business Model Score (static; max 5)
    if pd.notna(row.get('Business Model')):
        bm = row['Business Model'].strip().lower()
        if 'vertical saas' in bm:
            raw_scores['Business Model Raw Score'] = 1.0  # 5/5
            subscores['Business Model Score'] += 10
        elif 'horizontal saas' in bm:
            raw_scores['Business Model Raw Score'] = 0.0  # 0/5
            subscores['Business Model Score'] += 0

    # 6) Offshore Score  – penalty if evaluate_offshore_data() fails
    offshore_field = row.get('OffShore Data')   # <-- adjust to your column name
    if pd.notna(offshore_field):
        if not evaluate_offshore_data(offshore_field):
            subscores['Offshore Score'] -= 50
            raw_scores['Offshore Raw Score'] = -5.0   # -50 scaled to –5 on 0-10 raw scale

    # 7) Total Raise Penalty - deduct 50 points if total raise >= $45M
    # ------------------------------------------------------------------
    #  SINGLE kill-metric gate – apply once if *any* rule trips
    # ------------------------------------------------------------------
    total_raise_penalty = 0                 # default = no penalty
    kill_triggered      = False             # helper flag
    
    # --- Rule 1: Total Raised ≥ $45 M
    tot_raised = pd.to_numeric(row.get('Total Raised'), errors='coerce')
    if pd.notna(tot_raised) and tot_raised >= 45_000_000:
        kill_triggered = True
    
    # --- Rule 2: Last-Round Valuation ≥ $175 M
    lrv = pd.to_numeric(row.get('Last Round Valuation'), errors='coerce')
    if pd.notna(lrv) and lrv >= 175:
        kill_triggered = True
    
    # --- Rule 3-5: Oversized *last* raise by funding stage
    fund_type = str(row.get('Last Round', '')).upper()
    last_amt  = pd.to_numeric(row.get('Last Funding Amount'), errors='coerce')
    
    if pd.notna(last_amt):
        if fund_type in {'SEED', 'PRE_SEED'} and last_amt > 25_000_000:
            kill_triggered = True
        elif fund_type == 'SERIES_A' and last_amt > 25_000_000:
            kill_triggered = True
        elif fund_type == 'SERIES_B' and last_amt > 25_000_000:
            kill_triggered = True
    
    # --- Apply penalty once if any rule matched
    if kill_triggered:
        total_raise_penalty = -100                         # -100 pts
        raw_scores['Total Raise Penalty Raw Score'] = -10.0  # -10 on 0-10 scale


    # --- Determine Stage from 'Last Round' ---
    last_round_val = row.get('Last Round', "")
    stage = 'SERIES A'  # default
    if pd.notna(last_round_val):
        upper_round = str(last_round_val).upper()
        if 'SEED' in upper_round:
            stage = 'SEED'
        elif 'SERIES_B' in upper_round or 'SERIES B' in upper_round:
            stage = 'SERIES B'
        elif 'SERIES_A' in upper_round or 'SERIES A' in upper_round:
            stage = 'SERIES A'
    
    # Use dynamically generated stats_by_stage rather than hardcoded values
    stage_data = stats_by_stage.get(stage, stats_by_stage['SERIES A'])

    # ── Last-round valuation availability check  ───────────────
    lr_val = pd.to_numeric(row.get('Last Round Valuation'), errors='coerce')
    has_lr_val = pd.notna(lr_val) and lr_val > 0
    
    # --- Dictionary-based Metrics ---
    # Compute dictionary total possible points (distinct per stage)
    #dict_total_possible = sum(get_metric_weight(stage_data, metric) for metric in stage_data['weights'])

    # ── Denominator for dictionary metrics  ────────────────────
    # ── figure dynamic denominator with top-9 missing-data rule ─────────
    # ── dynamic denominator with LFA-seed fallback & top-9 rule ───────
    stage_top9 = stage_data.get('top9', [])
    dict_total_possible = 0
    
    for metric in stage_data['weights']:
        # Skip LRV if missing (old rule)
        if metric == 'Last Round Valuation' and not has_lr_val:
            continue
    
        # Special presence-check for LFA in SEED
        if metric == 'Last Funding Amount' and stage == 'SEED':
            lfa_val = pd.to_numeric(row.get('Last Funding Amount'), errors='coerce')
            if pd.isna(lfa_val) or lfa_val == 0:
                lfa_val = pd.to_numeric(row.get('Total Raised'), errors='coerce')
            missing = pd.isna(lfa_val) or lfa_val == 0
        else:
            # generic check (treat 0 as missing)
            missing = metric in stage_top9 and (
                pd.isna(row.get(metric_to_column.get(metric))) or
                pd.to_numeric(row.get(metric_to_column.get(metric)), errors='coerce') == 0
            )
    
        # top-9 rule: if missing & in top-9 → omit from possible points
        if metric in stage_top9 and missing:
            continue
    
        dict_total_possible += get_metric_weight(stage_data, metric)

    # Compute dictionary total possible points (distinct per stage)
#     if include_last_round_valuation:
#         # Include all metrics
#         dict_total_possible = sum(get_metric_weight(stage_data, metric) for metric in stage_data['weights'])
#     else:
#         # Exclude Last Round Valuation from total possible
#         dict_total_possible = sum(get_metric_weight(stage_data, metric) for metric in stage_data['weights'] 
#                                   if metric != 'Last Round Valuation')
    
    # Accumulate dictionary-based scores
    dict_total_score = 0
    
    # ────────── Founded Year (percentile-based MAIN metric) ──────────
#     fy_val = pd.to_numeric(row.get('Founded Year'), errors='coerce')
#     if pd.notna(fy_val):
#         p5, p95, avg = get_percentiles_and_avg(stage_data, 'Founded Year')
#         # Convert founded year to days since founding
#         founded_date = pd.to_datetime(f"{int(fy_val)}-01-01")  # Assume January 1st of the founded year
#         days_since_founded = (pd.Timestamp.now() - founded_date).days
#         raw = linear_decay_raw(days_since_founded, avg, p5, p95)
#         raw_scores['Founded Year Raw Score'] = raw
#         sc = raw * get_metric_weight(stage_data, 'Founded Year')   # weight = 10×1
#         subscores['Founded Year Score'] = sc
#         dict_total_score += sc

    # ────────── Days Since Founded (percentile-based MAIN metric) ──────────
    # ────────── Days Since Founded (percentile-based MAIN metric) ──────────
    founded_date_str = row.get('Founded Date')
    if pd.notna(founded_date_str) and founded_date_str != 'N/A':
        try:
            # Parse the full date from Affinity (e.g., "01/15/2020")
            founded_date = pd.to_datetime(founded_date_str, errors='coerce')
            if pd.notna(founded_date):
                p5, p95, avg = get_percentiles_and_avg(stage_data, 'Days Since Founded')
                # Calculate days since founding
                days_since_founded = (pd.Timestamp.now() - founded_date).days

                # Special scoring logic: if between 5th & 95th percentile, give 10 points
                if p5 <= days_since_founded <= p95:
                    raw = 1.0  # Full raw score
                    sc = 10.0  # Direct 10 points instead of using weight
                else:
                    raw = linear_decay_raw(days_since_founded, avg, p5, p95)
                    sc = raw * get_metric_weight(stage_data, 'Days Since Founded')

                raw_scores['Days Since Founded Raw Score'] = raw
                subscores['Founded Year Score'] = sc  # Keep display name as Founded Year Score
                dict_total_score += sc
        except Exception:
            pass  # Skip if date parsing fails

    # Total Raised Score
    total_raised_weight = get_metric_weight(stage_data, 'Total Raised')
    if pd.notna(row.get('Total Raised')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, 'Total Raised')
        val = pd.to_numeric(row.get('Total Raised'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_raw(val, p5, avg, p95)
        raw_scores['Total Raised Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * total_raised_weight
        subscores['Total Raised Score'] = score
        dict_total_score += score

    # Last Round Valuation Score
    lrv_weight = get_metric_weight(stage_data, 'Last Round Valuation')
    
    if has_lr_val:
        p5, p95, avg = get_percentiles_and_avg(stage_data, 'Last Round Valuation')
        raw_score = custom_scoring_raw(lr_val, p5, avg, p95)
        subscores['Last Round Valuation Score'] = raw_score * lrv_weight
        raw_scores['Last Round Valuation Raw Score'] = raw_score
        dict_total_score += subscores['Last Round Valuation Score']
    else:
        # **Penalty when the metric is missing**
        PENALTY_FRACTION = 0.0          # ← 0 % of its potential
        penalty_points = -PENALTY_FRACTION * lrv_weight
        subscores['Last Round Valuation Score'] = penalty_points
        raw_scores['Last Round Valuation Raw Score'] = -PENALTY_FRACTION
        dict_total_score += penalty_points

    # Total Employees Score
    max_te = get_metric_weight(stage_data, 'Total Employees')
    if pd.notna(row.get('Total Employees')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, 'Total Employees')
        employees = pd.to_numeric(row['Total Employees'], errors='coerce') or 0
        
        # Calculate raw score
        raw_score = linear_decay_raw(employees, avg, p5, p95)
        raw_scores['Total Employees Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_te
        subscores['Total Employees Score'] = score
        dict_total_score += score
        
        
        # ────────── Small-company penalty (moved inside this block) ────────────────────
        if employees < p5:
            subscores['Small Company Penalty']          = -50
            raw_scores['Small Company Penalty Raw Score'] = -5.0   # –50 scaled to 0-10 raw
            dict_total_score += -50
    
    subscores['Total Raise Penalty'] = total_raise_penalty

    # ────────── Last Funding Amount Score ─────────────────────────
    max_lfa = get_metric_weight(stage_data, 'Last Funding Amount')

    # value logic: in SEED, fall back to Total Raised if LFA missing/zero
    lfa_val = pd.to_numeric(row.get('Last Funding Amount'), errors='coerce')
    if stage == 'SEED' and (pd.isna(lfa_val) or lfa_val == 0):
        lfa_val = pd.to_numeric(row.get('Total Raised'), errors='coerce')

    if pd.notna(lfa_val) and lfa_val > 0:
        p5, p95, avg = get_percentiles_and_avg(stage_data, 'Last Funding Amount')
        raw_score = linear_decay_raw(lfa_val, avg, p5, p95)
        raw_scores['Last Funding Amount Raw Score'] = raw_score
        sc = raw_score * max_lfa
        subscores['Last Funding Amount Score'] = sc
        dict_total_score += sc
    # else → treated as “missing”; no score added


    # 1 Year Headcount Growth Score
    max_1yr_hc = get_metric_weight(stage_data, '1 Year Headcount Growth %')
    if pd.notna(row.get('1 Year Headcount △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '1 Year Headcount Growth %')
        val = pd.to_numeric(row.get('1 Year Headcount △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Headcount Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_1yr_hc
        subscores['1 Year Headcount Growth Score'] = score
        dict_total_score += score

    # 180d Headcount Growth Score
    max_180d_hc = get_metric_weight(stage_data, '180d Headcount Growth %')
    if pd.notna(row.get('180d Headcount △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '180d Headcount Growth %')
        val = pd.to_numeric(row.get('180d Headcount △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['180d Headcount Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_180d_hc
        subscores['180d Headcount Growth Score'] = score
        dict_total_score += score

    # 2 Year Headcount Growth Score
    max_2yr_hc = get_metric_weight(stage_data, '2 Year Headcount Growth %')
    if pd.notna(row.get('2 Year Headcount Growth')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '2 Year Headcount Growth %')
        val = pd.to_numeric(row.get('2 Year Headcount Growth'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['2 Year Headcount Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_2yr_hc
        subscores['2 Year Headcount Growth Score'] = score
        dict_total_score += score

    # 180d Web Traffic Growth Score
    max_180d_web = get_metric_weight(stage_data, '180d Web Traffic Growth %')
    if pd.notna(row.get('180d Web Traffic △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '180d Web Traffic Growth %')
        val = pd.to_numeric(row.get('180d Web Traffic △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['180d Web Traffic Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_180d_web
        subscores['180d Web Traffic Growth Score'] = score
        dict_total_score += score

    # 1 Year Web Traffic Growth Score
    max_1yr_web = get_metric_weight(stage_data, '1 Year Web Traffic Growth %')
    if pd.notna(row.get('1 Year Web Traffic △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '1 Year Web Traffic Growth %')
        val = pd.to_numeric(row.get('1 Year Web Traffic △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Web Traffic Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_1yr_web
        subscores['1 Year Web Traffic Growth Score'] = score
        dict_total_score += score

    # Last Fundraise Date Score
    max_days_since_fund = get_metric_weight(stage_data, 'Days Before Last Funding')
    if pd.notna(row.get('Most Recent Funding')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, 'Days Before Last Funding')
        days_since_funding = (pd.Timestamp.now() - pd.to_datetime(row['Most Recent Funding'])).days
        inverted_value = avg - (days_since_funding - avg)  # fewer days => better
        
        # Calculate raw score
        raw_score = linear_decay_raw(inverted_value, avg, p5, p95)
        raw_scores['Last Fundraise Date Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_days_since_fund
        subscores['Last Fundraise Date Score'] = score
        dict_total_score += score

    # 1 Year Sales Growth Score
    max_1yr_sales = get_metric_weight(stage_data, '1 Year Sales %')
    if pd.notna(row.get('1 Year Sales △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '1 Year Sales %')
        val = pd.to_numeric(row.get('1 Year Sales △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Sales Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_1yr_sales
        subscores['1 Year Sales Growth Score'] = score
        dict_total_score += score

    # 180d Sales Growth Score
    max_180d_sales = get_metric_weight(stage_data, '180d Sales %')
    if pd.notna(row.get('180d Sales Growth △%')):
        p5, p95, avg = get_percentiles_and_avg(stage_data, '180d Sales %')
        val = pd.to_numeric(row.get('180d Sales Growth △%'), errors='coerce') or 0
        
        # Calculate raw score
        raw_score = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['180d Sales Growth Raw Score'] = raw_score
        
        # Calculate weighted score
        score = raw_score * max_180d_sales
        subscores['180d Sales Growth Score'] = score
        dict_total_score += score

    # ────────── 1 Year Engineering Headcount Growth ───────────────
    max_eng_hc = get_metric_weight(stage_data,
                                   '1 Year Engineering Headcount %')
    eng_col = '1 Year Engineering Headcount △%'      # ← DF column name
    if pd.notna(row.get(eng_col)):
        p5, p95, avg = get_percentiles_and_avg(stage_data,
                                               '1 Year Engineering Headcount %')
        val = pd.to_numeric(row.get(eng_col), errors='coerce') or 0
        raw = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Engineering Headcount Growth Raw Score'] = raw
        sc = raw * max_eng_hc
        subscores['1 Year Engineering Headcount Growth Score'] = sc
        dict_total_score += sc

    # ────────── 1 Year Operations Headcount Growth ────────────────
    max_ops_hc = get_metric_weight(stage_data,
                                   '1 Year Operations Headcount %')
    ops_col = '1 Year Operations Headcount △%'        # ← DF column name
    if pd.notna(row.get(ops_col)):
        p5, p95, avg = get_percentiles_and_avg(stage_data,
                                               '1 Year Operations Headcount %')
        val = pd.to_numeric(row.get(ops_col), errors='coerce') or 0
        raw = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Operations Headcount Growth Raw Score'] = raw
        sc = raw * max_ops_hc
        subscores['1 Year Operations Headcount Growth Score'] = sc
        dict_total_score += sc

    # ────────── 1 Year LinkedIn Followers Growth ──────────────────
    max_li = get_metric_weight(stage_data,
                               '1 Year Linkedin Followers %')
    li_col = '1 Year Linkedin Followers △%'           # ← DF column name
    if pd.notna(row.get(li_col)):
        p5, p95, avg = get_percentiles_and_avg(stage_data,
                                               '1 Year Linkedin Followers %')
        val = pd.to_numeric(row.get(li_col), errors='coerce') or 0
        raw = custom_scoring_inv_raw(val, p5, avg, p95)
        raw_scores['1 Year Linkedin Followers Growth Raw Score'] = raw
        sc = raw * max_li
        subscores['1 Year Linkedin Followers Growth Score'] = sc
        dict_total_score += sc

#     # ────────── Small-company penalty ────────────────────────────
#     if pd.notna(row.get('Total Employees')) and employees < p5:
#         subscores['Small Company Penalty']          = -50
#         raw_scores['Small Company Penalty Raw Score'] = -5.0   # –50 scaled to 0-10 raw
#         # direct subtraction from overall total:
#         dict_total_score += -50


    # --- Compute Overall Totals ---
    # Static metrics (for normalization) have a maximum of:
    # Kill Metrics Score: 0 (best case), Location Score: 10, Founded Year Score: 10, Business Model Score: 10
    additional_possible = 0 + 10 + 10  # = 20
    additional_score = (
         subscores['Kill Metrics Score'] +
         subscores['Location Score'] +
         subscores['Business Model Score']
    )
    
    # Total possible points (distinct per stage) is:
    total_possible_points = dict_total_possible + additional_possible

    # Final overall score includes all dictionary-based scores, the static additional score, and the penalty from Location Kill Score.
    final_total_score = dict_total_score + additional_score + subscores['Location Kill Score'] + subscores['Offshore Score'] + subscores['Total Raise Penalty']

    # Normalized score computed on the basis of total possible points.
    normalized_score = (final_total_score / total_possible_points) * 100 if total_possible_points > 0 else 0

    subscores['Total Score'] = final_total_score
    subscores['Normalized Score'] = normalized_score
    subscores['Total Possible Points'] = total_possible_points
    
    # --- Explanation of top/bottom drivers using only performance metrics ---
    # Define the excluded keys for reasoning
    excluded_keys_for_reasoning = {
        'Business Model Score', 'Location Score', 'Founded Year Score',
        'Total Employees Score', 'Kill Metrics Score',
        'Total Score', 'Normalized Score', 'Total Possible Points', 
        'Total Raised', 'Last Funding Amount','Offshore Score', 'Small Company Penalty',
    }
    
    # Filter subscores to only include metrics we want for reasoning
    metric_subscores = {
        k: v for k, v in subscores.items() if k not in excluded_keys_for_reasoning
    }
    
    # Sort for top and bottom drivers
    desc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1], reverse=True)
    asc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1])
    
    if normalized_score > 40:
        scoring_category = "GOOD"
        drivers_to_explain = desc_sorted[:3]
    elif 35 < normalized_score < 40:
        scoring_category = "MIDDLE"
        drivers_to_explain = desc_sorted[:2] + asc_sorted[:1]
    else:
        scoring_category = "BAD"
        drivers_to_explain = asc_sorted[:3]

    explanations = []
    for (m_name, val) in drivers_to_explain:
        sign = "+" if val >= 0 else ""
        explanations.append(f"{m_name} => {sign}{val:.2f} pts")
    if scoring_category == "GOOD":
        explanation = (
            f"Score is {normalized_score:.1f} => GOOD. "
            f"Top 3 drivers: " + "; ".join(explanations)
        )
    elif scoring_category == "MIDDLE":
        explanation = (
            f"Score is {normalized_score:.1f} => MIDDLE (35-40). "
            f"2 top + 1 bottom: " + "; ".join(explanations)
        )
    else:
        explanation = (
            f"Score is {normalized_score:.1f} => BAD (<35). "
            f"Lowest 3 drivers: " + "; ".join(explanations)
        )
    subscores['Scoring Explanation'] = explanation

    # (Optional) Retain the Last Fundraise Date Score multiplied by 10 for further reporting
    subscores['Last Fundraise Date Score (x10)'] = subscores['Last Fundraise Date Score'] * 10

    # Add raw scores to the subscores dictionary
    for key, value in raw_scores.items():
        subscores[key] = value * 10

    return subscores


# <hr style="border: 3px solid black;">
# 

# # Get RAW Data From Affinity

# In[10]:


import requests
import pandas as pd
import pytz
from datetime import datetime

# Affinity API Credentials
API_BASE_URL = "https://api.affinity.co/v2"
API_KEY = "_1EA5jw1tabci0t38DXPz-VcIW0Pz_CmTCbbMsPQsOg"

# Define the List ID and View ID
LIST_ID = "151949"  # Change this to "117968" when pulling for the other list
VIEW_ID = "1048283"  # Change this accordingly for the other list

# # Define the fields to retrieve
# FIELDS_TO_RETRIEVE = {
#     "Total Employees": "field-4135365",
#     "Last Funding Amount": "field-4135382",
#     "Total Raised": "field-4135383",
#     "1 Year Headcount Growth %": "field-4135397",
#     "180d Headcount Growth %": "field-4135396",
#     "2 Year Headcount Growth %": "field-4975108",
#     "1 Year Web Traffic Growth %": "field-4642225",
#     "180d Web Traffic Growth %": "field-4642226",
#     "Last Round Valuation": "field-4739650",
#     "1 Year Sales %": "field-4135400",
#     "180d Sales %": "field-4135399",
#     "Most Recent Funding": "field-4135385",
#     "Business Model": "field-3602201",
#     "HQ Location (Country)": "field-4135401",
#     "Founded Date": "field-4646931",
#     "Current Investors": "field-4135384",
#     "Company Status": "field-4788673"  
# }

FIELDS_TO_RETRIEVE = {
    "Total Employees": "field-4135365",
    "Last Funding Amount": "field-4135382",
    "Total Raised": "field-4135383",
    "1 Year Headcount △%": "field-4135397",
    "180d Headcount △%": "field-4135396",
    "2 Year Headcount Growth": "field-4975108",
    "1 Year Web Traffic △%": "field-4642225",
    "180d Web Traffic △%": "field-4642226",
    "Last Round Valuation": "field-4739650",
    "1 Year Sales △%": "field-4135400",
    "180d Sales Growth △%": "field-4135399",
    "Most Recent Funding": "field-4135385",
    "Business Model": "field-3602201",
    "HQ Location (Country)": "field-4135401",
    "Founded Date": "field-4646931",
    "Current Investors": "field-5308281", #this is actaully the new previus investor column
    "Company Status": "field-4788673", 
    "Last Round": "field-4135361",
    "End Market": "field-3602465",
    "OffShore Data": "field-5084909",
    "1 Year Linkedin Followers △%": "field-5195507",
    "1 Year Engineering Headcount △%": "field-5195508",
    "1 Year Operations Headcount △%": "field-5195509"
}

# Initialize headers
headers = {"Authorization": f"Bearer {API_KEY}"}

# API URL for fetching list entries
entries_url = f"{API_BASE_URL}/lists/{LIST_ID}/saved-views/{VIEW_ID}/list-entries"

# Fetch all entries from the saved view
entries = []
next_url = entries_url  # Start with the first request

while next_url:
    response = requests.get(next_url, headers=headers)
    
    # Handle potential API errors
    if response.status_code != 200:
        raise Exception(f"Error fetching list entries: {response.status_code}, {response.text}")

    data = response.json()
    entries.extend(data.get("data", []))
    
    # Get next URL for pagination
    next_url = data.get("pagination", {}).get("nextUrl")

# Function to format dates from UTC to Pacific Time (PT)
def format_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return ""
    try:
        utc_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        pacific_time = utc_date.astimezone(pytz.timezone("America/Los_Angeles"))
        return pacific_time.strftime("%m/%d/%Y")
    except Exception:
        return date_str

def extract_field_value(field, field_name=None):
    """Extracts the correct value from different field formats."""
    
    if field_name == "Last Round":
        if isinstance(field, dict):
            # Check if it's directly the dropdown object
            if "text" in field:
                return field.get("text", "N/A")
            # Or if it's nested in the data attribute
            elif "data" in field:
                data = field.get("data")
                if isinstance(data, dict) and "text" in data:
                    return data.get("text", "N/A")
        return "N/A"
    
    # For End Market field which may come as a list of dropdown objects
    if field_name == "End Market":
        if isinstance(field, list) and field:
            # Extract text values from all items in the list
            market_values = []
            for item in field:
                if isinstance(item, dict) and "text" in item:
                    market_values.append(item.get("text", "N/A"))
            return ", ".join(market_values) if market_values else "N/A"
        elif isinstance(field, dict):
            # Handle if it's in the data field
            if "data" in field:
                data = field.get("data")
                if isinstance(data, list):
                    market_values = []
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            market_values.append(item.get("text", "N/A"))
                    return ", ".join(market_values) if market_values else "N/A"
                elif isinstance(data, dict) and "text" in data:
                    return data.get("text", "N/A")
            # Handle if text is directly in the dict
            elif "text" in field:
                return field.get("text", "N/A")
        return "N/A"
    
    # For Company Status, use the same approach as the working code
    if field_name == "Company Status":
        if isinstance(field, dict):
            data = field.get("data", "N/A")  # Extract 'data' field

            if isinstance(data, dict):  # Single dropdown option
                return data.get("text", "N/A")  
            elif isinstance(data, list):  # Multiple dropdown options
                return ", ".join([item.get("text", "N/A") for item in data if isinstance(item, dict)])
        
        return "N/A"  # Default if no valid data found
    
    # Special handling for Business Model
    if field_name == "Business Model":
        if isinstance(field, list):
            business_models = [item.get("text", "N/A") for item in field if isinstance(item, dict)]
            return ", ".join(business_models) if business_models else "N/A"
        elif isinstance(field, dict) and "data" in field and isinstance(field["data"], list):
            business_models = [item.get("text", "N/A") for item in field["data"] if isinstance(item, dict)]
            return ", ".join(business_models) if business_models else "N/A"
        return field.get("text", "N/A")
    
    # Special handling for Current Investors
    if field_name == "Current Investors":
        if isinstance(field, list):
            investors = [item.get("text", "N/A") for item in field if isinstance(item, dict)]
            return "; ".join(investors) if investors else "N/A"
        elif isinstance(field, dict) and "data" in field and isinstance(field["data"], list):
            investors = [item.get("text", "N/A") for item in field["data"] if isinstance(item, dict)]
            return "; ".join(investors) if investors else "N/A"
        return field.get("text", "N/A")
    
    # Handle date fields
    if isinstance(field, dict):
        if field_name in ["Most Recent Funding", "Founded Date"]:
            return format_date(field.get("data", "N/A"))
        if field_name == "HQ Location (Country)":
            data = field.get("data")
            if data is None:
                return "N/A"
            if isinstance(data, dict):
                return data.get("country", "N/A")
            return str(data)
        return field.get("data", "N/A")
    
    # Handle generic list values
    elif isinstance(field, list) and field:
        return field[0].get("data", "N/A") if isinstance(field[0], dict) else "N/A"
    
    return str(field) if field else "N/A"


# Process the retrieved data
extracted_data = []
for entry in entries:
    entity = entry.get("entity", {})
    row_data = {
        "Affinity Row ID": entry.get("id", "Unknown"),  # Get the Affinity row ID
        "Organization Id": entity.get("id", "Unknown"),
        "Name": entity.get("name", "Unknown"),
        "Website": entity.get("domain", "Unknown")
    }
    
    # Access the correct field structure
    entity_fields = entity.get("fields", [])
    
    # Convert list of field dicts into a dictionary {field_id: field_value}
    fields_dict = {f["id"]: f.get("value", {}) for f in entity_fields}
    
    # Extract only the required fields with special handling
    for field_name, field_id in FIELDS_TO_RETRIEVE.items():
        field_value = extract_field_value(fields_dict.get(field_id, {}), field_name)  # Extract using function
        row_data[field_name] = field_value

    extracted_data.append(row_data)

# Convert extracted data into a Pandas DataFrame
df = pd.DataFrame(extracted_data)


# <hr style="border: 3px solid black;">
# 

# # Getting VC List Data from Affinity

# In[14]:


import requests
import pandas as pd

# Affinity API Credentials
API_BASE_URL = "https://api.affinity.co/v2"
API_KEY = "_1EA5jw1tabci0t38DXPz-VcIW0Pz_CmTCbbMsPQsOg"
LIST_ID = "117968"
VIEW_ID = "610761"  # Specific Saved View ID

# Define the fields to retrieve
FIELDS_TO_RETRIEVE = {
    "Status": "field-1986685"
}

# Initialize headers
headers = {"Authorization": f"Bearer {API_KEY}"}

# API URL for fetching list entries from the specific view
entries_url = f"{API_BASE_URL}/lists/{LIST_ID}/saved-views/{VIEW_ID}/list-entries"

# Fetch all entries from the saved view
entries = []
next_url = entries_url  # Start with the first request

while next_url:
    response = requests.get(next_url, headers=headers)
    
    # Handle potential API errors
    if response.status_code != 200:
        raise Exception(f"Error fetching list entries: {response.status_code}, {response.text}")

    data = response.json()
    entries.extend(data.get("data", []))
    
    # Get next URL for pagination
    next_url = data.get("pagination", {}).get("nextUrl")


def extract_field_value(field):
    """Extracts the correct value from different field formats."""
    if isinstance(field, dict):
        data = field.get("data", "N/A")  # Extract 'data' field

        if isinstance(data, dict):  # Single dropdown option
            return data.get("text", "N/A")  
        elif isinstance(data, list):  # Multiple dropdown options
            return ", ".join([item.get("text", "N/A") for item in data if isinstance(item, dict)])
    
    return "N/A"  # Default if no valid data found


# Process the retrieved data
extracted_data = []
for entry in entries:
    entity = entry.get("entity", {})
    row_data = {
        "Affinity Row ID": entry.get("id", "Unknown"),  # Get the Affinity row ID
        "Organization Id": entity.get("id", "Unknown"),
        "Name": entity.get("name", "Unknown"),
        "Website": entity.get("domain", "Unknown")
    }
    
    # Access the correct field structure
    entity_fields = entity.get("fields", [])
    
    # Convert list of field dicts into a dictionary {field_id: field_value}
    fields_dict = {f["id"]: f.get("value", {}) for f in entity_fields}
    
    # Extract only the required fields
    for field_name, field_id in FIELDS_TO_RETRIEVE.items():
        field_value = extract_field_value(fields_dict.get(field_id, {}))
        row_data[field_name] = field_value

    extracted_data.append(row_data)

# Convert extracted data into a Pandas DataFrame
vc_list = pd.DataFrame(extracted_data)


# <hr style="border: 3px solid black;">
# 
# # Running Line by Line: Scores and Subscores

# In[15]:


import pandas as pd
import numpy as np
from datetime import datetime
import re

# Assuming all prerequisite functions and data processing are already in place:
# - The calculate_score function is defined
# - Dataframes df and vc_list are loaded
# - Raw data is properly preprocessed

# STEP 1: Apply initial score calculation to get subscores
print("Step 1: Calculating initial subscores...")
subscore_df = df.apply(lambda row: calculate_score(row), axis=1, result_type='expand')

# Ensure no duplicate columns when combining with original dataframe
df = df.loc[:, ~df.columns.duplicated(keep='first')]
new_cols = subscore_df.columns.difference(df.columns)
overwrite_cols = subscore_df.columns.intersection(df.columns)

# Add new columns from subscores
if not new_cols.empty:
    df = pd.concat([df, subscore_df[new_cols]], axis=1)
    
# Update existing columns with new values
for col in overwrite_cols:
    df[col] = subscore_df[col]

# Remove any duplicate columns again (just to be safe)
df = df.loc[:, ~df.columns.duplicated(keep='last')]

# -----------------------------------------------------------------
#  NEW: limit the percentile calc to Seed, Series A and Series B
# -----------------------------------------------------------------
STAGE_COL      = 'Last Round'           # change if your column name differs
TARGET_STAGES  = {'seed', 'series a', 'series b'}

# helper – returns a copy filtered to the target stages
def stage_filtered(df, col=STAGE_COL, stages=TARGET_STAGES):
    return (
        df.loc[                                   # keep row if stage matches
            df[col]
              .astype(str)                        # guard against NaNs
              .str.strip()
              .str.lower()
              .isin(stages)
        ]
        .copy()
    )

df_stage = stage_filtered(df)

if df_stage.empty:
    raise ValueError(
        f'No rows found with {STAGE_COL} in {TARGET_STAGES}. '
        'Check column name / stage labels.'
    )
print(f"Percentile stats based on {len(df_stage):,} Seed/Series-A/B companies")

# STEP 2: Calculate distribution statistics including 75th percentile
print("Step 2: Calculating distribution statistics...")

# Define weighted score metrics only (exclude raw scores)
weighted_metrics_to_analyze = [
    "Offshore Score",
    "Location Score",
    "Founded Year Score",
    "Last Fundraise Date Score",
    "Total Employees Score", 
    "Last Funding Amount Score",
    "Total Raised Score",
    "1 Year Headcount Growth Score",
    "180d Headcount Growth Score",
    "2 Year Headcount Growth Score",
    "180d Web Traffic Growth Score",
    "1 Year Web Traffic Growth Score",
    "Last Round Valuation Score",
    "1 Year Sales Growth Score",
    "180d Sales Growth Score",
    "1 Year Engineering Headcount Growth Score",    # ADD THIS
    "1 Year Operations Headcount Growth Score",     # ADD THIS
    "1 Year Linkedin Followers Growth Score",       # ADD THIS
    "Total Raise Penalty",
    "Small Company Penalty",                        # ADD THIS
    "Business Model Score",
    "Normalized Score"
]


# Filter to only columns that exist in the dataframe
columns_present = [col for col in weighted_metrics_to_analyze if col in df.columns]
df_numeric = df_stage[columns_present].select_dtypes(include=['number'])

# Calculate distribution statistics
summary_data = []
percentile_75th = {}

for column in df_numeric.columns:
    if df_numeric[column].notna().sum() > 0:  # Only process columns with data
        try:
            p75 = df_numeric[column].quantile(0.75)
            percentile_75th[column] = p75
            
            summary_data.append({
                'Variable': column,
                'Min': df_numeric[column].min(),
                '5th Percentile': df_numeric[column].quantile(0.05),
                '25th Percentile': df_numeric[column].quantile(0.25),
                'Median': df_numeric[column].median(),
                '75th Percentile': p75,
                '95th Percentile': df_numeric[column].quantile(0.95),
                'Max': df_numeric[column].max()
            })
        except Exception as e:
            print(f"Warning: Could not calculate statistics for column '{column}': {e}")

# Save distribution statistics
summary_df = pd.DataFrame(summary_data)
write_csv_to_s3(summary_df, "company-scoring", "subscores_distribution.csv")
print(f"Saved distribution statistics to subscores_distribution.csv")

# Calculate 75th percentile for raw scores (for later use in the code)
# Define raw score metrics
raw_metrics_to_analyze = [
    "Location Raw Score",
    "Founded Year Raw Score",
    "Last Fundraise Date Raw Score",
    "Total Employees Raw Score",
    "Last Funding Amount Raw Score",
    "Total Raised Raw Score",
    "1 Year Headcount Growth Raw Score",
    "180d Headcount Growth Raw Score",
    "2 Year Headcount Growth Raw Score",
    "180d Web Traffic Growth Raw Score",
    "1 Year Web Traffic Growth Raw Score",
    "Last Round Valuation Raw Score",
    "1 Year Sales Growth Raw Score",
    "180d Sales Growth Raw Score",
    "1 Year Engineering Headcount Growth Raw Score",    # ADD THIS
    "1 Year Operations Headcount Growth Raw Score",     # ADD THIS
    "1 Year Linkedin Followers Growth Raw Score",       # ADD THIS
    "Small Company Penalty Raw Score",                  # ADD THIS
    "Total Raise Penalty Raw Score",
    "Business Model Raw Score",
    "Kill Metrics Raw Score",
    "Location Kill Raw Score",
    "Offshore Raw Score"
]

# Filter to only raw score columns that exist in the dataframe
raw_columns_present = [col for col in raw_metrics_to_analyze if col in df.columns]
raw_df_numeric = df_stage[raw_columns_present].select_dtypes(include=['number'])

# Calculate 75th percentiles for raw scores and add to percentile_75th dictionary
for column in raw_df_numeric.columns:
    if raw_df_numeric[column].notna().sum() > 0:
        try:
            p75 = raw_df_numeric[column].quantile(0.75)
            percentile_75th[column] = p75
        except Exception as e:
            print(f"Warning: Could not calculate 75th percentile for column '{column}': {e}")
            
            
# STEP 3: Update scoring explanations based on the 75th percentile
print("Step 3: Updating scoring explanations based on 75th percentile...")

# Get the 75th percentile value for Normalized Score
normalized_score_75th = percentile_75th.get('Normalized Score', 0)
threshold_middle = normalized_score_75th - 10  # 10 points below 75th percentile

# def update_score_explanation(row):
#     """
#     Updates the scoring explanation based on normalized score thresholds:
#     - GOOD: Above 75th percentile
#     - MIDDLE: Between 75th percentile and 10 points below
#     - BAD: Below 10 points from 75th percentile
#     """
#     normalized_score = row.get('Normalized Score', 0)
    
#     # Define excluded keys for reasoning
#     excluded_keys_for_reasoning = {
#         'Business Model Score', 'Location Score', 'Founded Year Score',
#         'Total Employees Score', 'Kill Metrics Score',
#         'Total Score', 'Normalized Score', 'Total Possible Points', 
#         'Total Raised', 'Last Funding Amount', 'Location Kill Score'
#     }
    
#     # Filter subscores to only include metrics we want for reasoning
#     metric_subscores = {}
#     for k, v in row.items():
#         if k.endswith('Score') and k not in excluded_keys_for_reasoning and isinstance(v, (int, float)):
#             metric_subscores[k] = v
    
#     # Sort for top and bottom drivers
#     desc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1], reverse=True)
#     asc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1])
    
#     # Determine scoring category based on 75th percentile thresholds
#     if normalized_score >= normalized_score_75th:
#         scoring_category = "GOOD"
#         drivers_to_explain = desc_sorted[:3]  # Top 3 drivers for GOOD
#     elif threshold_middle <= normalized_score < normalized_score_75th:
#         scoring_category = "MIDDLE"
#         drivers_to_explain = desc_sorted[:2] + asc_sorted[:1]  # 2 top + 1 bottom for MIDDLE
#     else:
#         scoring_category = "BAD"
#         drivers_to_explain = asc_sorted[:3]  # Bottom 3 drivers for BAD
    
#     # Format explanations
#     explanations = []
#     for (m_name, val) in drivers_to_explain:
#         sign = "+" if val >= 0 else ""
#         explanations.append(f"{m_name} => {sign}{val:.2f} pts")
        
#     if scoring_category == "GOOD":
#         explanation = (
#             f"Score is {normalized_score:.1f} => GOOD. "
#             f"Top 3 drivers: " + "; ".join(explanations)
#         )
#     elif scoring_category == "MIDDLE":
#         explanation = (
#             f"Score is {normalized_score:.1f} => MIDDLE. "
#             f"2 top + 1 bottom: " + "; ".join(explanations)
#         )
#     else:
#         explanation = (
#             f"Score is {normalized_score:.1f} => BAD. "
#             f"Lowest 3 drivers: " + "; ".join(explanations)
#         )
        
#     return explanation


def update_score_explanation(row):
    """
    Updates the scoring explanation based on normalized score thresholds:
    - GOOD: Above 75th percentile
    - MIDDLE: Between 75th percentile and 10 points below
    - BAD: Below 10 points from 75th percentile
    """
    normalized_score = row.get('Normalized Score', 0)
    
    # Define excluded keys for reasoning
    excluded_keys_for_reasoning = {
        'Business Model Score', 'Location Score', 'Founded Year Score',
        'Total Employees Score', 'Kill Metrics Score',
        'Total Score', 'Normalized Score', 'Total Possible Points', 
        'Total Raised', 'Last Funding Amount', 'Location Kill Score', 'Offshore Score'
    }
    
    # Add all raw score metrics to excluded keys
    raw_score_metrics = [
        'Location Raw Score', 'Founded Year Raw Score', 'Last Fundraise Date Raw Score',
        'Total Employees Raw Score', 'Last Funding Amount Raw Score', 'Total Raised Raw Score',
        '1 Year Headcount Growth Raw Score', '180d Headcount Growth Raw Score', '2 Year Headcount Growth Raw Score',
        '180d Web Traffic Growth Raw Score', '1 Year Web Traffic Growth Raw Score', 'Last Round Valuation Raw Score',
        '1 Year Sales Growth Raw Score', '180d Sales Growth Raw Score', 'Business Model Raw Score',
        '1 Year Engineering Headcount Growth Raw Score',    # ADD THIS
        '1 Year Operations Headcount Growth Raw Score',     # ADD THIS
        '1 Year Linkedin Followers Growth Raw Score',       # ADD THIS
        'Small Company Penalty Raw Score',                  # ADD THIS
        'Total Raise Penalty Raw Score',
        'Kill Metrics Raw Score', 'Location Kill Raw Score', 'Offshore Raw Score'
    ]
    # Add all raw score metrics to the excluded set
    for raw_metric in raw_score_metrics:
        excluded_keys_for_reasoning.add(raw_metric)
    
    # Filter subscores to only include metrics we want for reasoning
    metric_subscores = {}
    for k, v in row.items():
        if k.endswith('Score') and k not in excluded_keys_for_reasoning and isinstance(v, (int, float)):
            metric_subscores[k] = v
    
    # Sort for top and bottom drivers
    desc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1], reverse=True)
    asc_sorted = sorted(metric_subscores.items(), key=lambda x: x[1])
    
    # Determine scoring category based on 75th percentile thresholds
    if normalized_score >= normalized_score_75th:
        scoring_category = "GOOD"
        drivers_to_explain = desc_sorted[:3]  # Top 3 drivers for GOOD
    elif threshold_middle <= normalized_score < normalized_score_75th:
        scoring_category = "MIDDLE"
        drivers_to_explain = desc_sorted[:2] + asc_sorted[:1]  # 2 top + 1 bottom for MIDDLE
    else:
        scoring_category = "BAD"
        drivers_to_explain = asc_sorted[:3]  # Bottom 3 drivers for BAD
    
    # Format explanations
    explanations = []
    for (m_name, val) in drivers_to_explain:
        sign = "+" if val >= 0 else ""
        explanations.append(f"{m_name} => {sign}{val:.2f} pts")
        
    if scoring_category == "GOOD":
        explanation = (
            f"Score is {normalized_score:.1f} => GOOD. "
            f"Top 3 drivers: " + "; ".join(explanations)
        )
    elif scoring_category == "MIDDLE":
        explanation = (
            f"Score is {normalized_score:.1f} => MIDDLE. "
            f"2 top + 1 bottom: " + "; ".join(explanations)
        )
    else:
        explanation = (
            f"Score is {normalized_score:.1f} => BAD. "
            f"Lowest 3 drivers: " + "; ".join(explanations)
        )
        
    return explanation

# Apply the updated scoring logic
print(f"Using 75th percentile value for Normalized Score: {normalized_score_75th:.2f}")
print(f"Middle threshold: {threshold_middle:.2f}")
df['Scoring Explanation'] = df.apply(update_score_explanation, axis=1)

# STEP 4: Find investor connections from priority VC list
print("Step 4: Finding investor connections...")
priority_vcs = vc_list[vc_list['Status'].isin(['Low Priority','Medium Priority', 'High Priority'])]

def find_investor_connections(row, priority_vcs_df):
    """
    Returns a string of investors from priority_vcs_df that match with the 
    Current Investors in the row, handling variations in naming conventions.
    """
    if 'Current Investors' in row and pd.notna(row['Current Investors']):
        investors = [investor.strip() for investor in str(row['Current Investors']).split(';')]
        
        if 'Name' in priority_vcs_df.columns:
            vc_names = priority_vcs_df['Name'].tolist()
            connections = []
            
            # Map common suffixes and abbreviations
            common_suffixes = [
                'Capital', 'Ventures', 'Venture Partners', 'Partners', 'VC', 
                'Venture Capital', 'Fund', 'Equity', 'Management', 'Group'
            ]
            
            # Abbreviation mappings
            abbreviations = {
                'NEA': 'New Enterprise Associates',
                'a16z': 'Andreessen Horowitz',
                'USV': 'Union Square Ventures'
            }
            
            # For each VC in our priority list
            for vc in vc_names:
                # Check if any investor matches this VC
                matched = False
                
                for investor in investors:
                    # First check exact match
                    if vc.lower() == investor.lower():
                        connections.append(vc)
                        matched = True
                        break
                    
                    # Check normalized forms (without common suffixes)
                    vc_base_name = vc
                    investor_base_name = investor
                    
                    # Remove common suffixes
                    for suffix in common_suffixes:
                        # Should be corrected to:
                        vc_base_name = re.sub(r'\s+' + suffix + '$', '', vc_base_name, flags=re.IGNORECASE)
                        investor_base_name = re.sub(r'\s+' + suffix + '$', '', investor_base_name, flags=re.IGNORECASE)

                    vc_base_name = vc_base_name.strip().lower()
                    investor_base_name = investor_base_name.strip().lower()
                    
                    if vc_base_name == investor_base_name:
                        connections.append(vc)
                        matched = True
                        break
                    
                    # Check for abbreviations
                    for abbr, full_name in abbreviations.items():
                        if ((investor.lower() == abbr.lower() and full_name.lower() in vc.lower()) or
                            (vc.lower() == abbr.lower() and full_name.lower() in investor.lower())):
                            connections.append(vc)
                            matched = True
                            break
                    
                    if matched:
                        break
                    
                    # Check if one contains the other (but prevent very short name matches)
                    if ((len(investor) > 3 and investor.lower() in vc.lower()) or 
                        (len(vc) > 3 and vc.lower() in investor.lower())):
                        connections.append(vc)
                        matched = True
                        break
            
            return '; '.join(connections) if connections else None
        else:
            print("Warning: 'Name' column not found in priority_vcs_df")
            return None
    return None

# Apply the function to create "Our Investor Connections" in df
df['Our Investor Connections'] = df.apply(lambda row: find_investor_connections(row, priority_vcs), axis=1)

# STEP 5: Save final outputs
print("Step 5: Saving final outputs...")

# Define columns for subscores.csv including raw score columns
subscore_columns = [
    'Affinity Row ID', 'Organization Id', 'Name', 'Website', 'Business Model',
    'Business Model Score', 'Business Model Raw Score', 
    'OffShore Data', 'Offshore Score', 'Offshore Raw Score',
    'Total Employees','Total Employees Score', 'Total Employees Raw Score',
    '1 Year Web Traffic △%', '1 Year Web Traffic Growth Score', '1 Year Web Traffic Growth Raw Score',
    'Last Round Valuation', 'Last Round Valuation Score', 'Last Round Valuation Raw Score',
    '1 Year Sales △%', '1 Year Sales Growth Score', '1 Year Sales Growth Raw Score',
    '180d Sales Growth △%', '180d Sales Growth Score', '180d Sales Growth Raw Score',
    '2 Year Headcount Growth', '2 Year Headcount Growth Score', '2 Year Headcount Growth Raw Score',
    'Most Recent Funding', 'Last Fundraise Date Score', 'Last Fundraise Date Raw Score',
    'Total Raised', 'Total Raised Score', 'Total Raised Raw Score',
    'Founded Date','Founded Year Score', 'Days Since Founded Raw Score',
    '1 Year Headcount △%','1 Year Headcount Growth Score', '1 Year Headcount Growth Raw Score',
    '180d Headcount △%', '180d Headcount Growth Score', '180d Headcount Growth Raw Score',
    '180d Web Traffic △%', '180d Web Traffic Growth Score', '180d Web Traffic Growth Raw Score',
    'HQ Location (Country)', 'Location Score', 'Location Raw Score',
    'Last Funding Amount', 'Last Funding Amount Score', 'Last Funding Amount Raw Score',
    # ADD THESE NEW COLUMNS:
    '1 Year Engineering Headcount △%', '1 Year Engineering Headcount Growth Score', '1 Year Engineering Headcount Growth Raw Score',
    '1 Year Operations Headcount △%', '1 Year Operations Headcount Growth Score', '1 Year Operations Headcount Growth Raw Score',
    '1 Year Linkedin Followers △%', '1 Year Linkedin Followers Growth Score', '1 Year Linkedin Followers Growth Raw Score',
    'Small Company Penalty', 'Small Company Penalty Raw Score', 'Total Raise Penalty Raw Score',
    # END NEW COLUMNS
    'Kill Metrics Score', 'Kill Metrics Raw Score',
    'Location Kill Score', 'Location Kill Raw Score',
    'Normalized Score', 'Total Score', 'Total Possible Points', 'Last Round', 'End Market'
]

# Filter to only columns that exist in the dataframe
final_subscore_columns = [col for col in subscore_columns if col in df.columns]
#df[final_subscore_columns].to_csv('subscores.csv', index=False)
write_csv_to_s3(df[final_subscore_columns], 'company-scoring', "subscores.csv")



# Define columns for raw scores only CSV
raw_score_columns = [
    'Affinity Row ID', 'Organization Id', 'Name', 'Website',
    'Total Raised Raw Score', 'Last Round Valuation Raw Score',
    'Total Employees Raw Score', 'Last Funding Amount Raw Score',
    '1 Year Headcount Growth Raw Score', '180d Headcount Growth Raw Score',
    '2 Year Headcount Growth Raw Score', '180d Web Traffic Growth Raw Score',
    '1 Year Web Traffic Growth Raw Score', '1 Year Sales Growth Raw Score',
    '180d Sales Growth Raw Score', 'Last Fundraise Date Raw Score',
    'Location Raw Score', 'Days Since Founded Raw Score', 'Business Model Raw Score',
    # ADD THESE NEW COLUMNS:
    '1 Year Engineering Headcount Growth Raw Score',
    '1 Year Operations Headcount Growth Raw Score',
    '1 Year Linkedin Followers Growth Raw Score',
    'Small Company Penalty Raw Score',
    'Total Raise Penalty Raw Score',
    # END NEW COLUMNS
    'Kill Metrics Raw Score', 'Location Kill Raw Score', 'Offshore Raw Score'
]

# Filter to only raw score columns that exist in the dataframe
final_raw_score_columns = [col for col in raw_score_columns if col in df.columns]
write_csv_to_s3(df[final_raw_score_columns], 'company-scoring', "raw_scores.csv")


# Define columns for official scores
official_columns = [
    "Affinity Row ID", "Organization Id", "Name", "Website",
    "Normalized Score", "Scoring Explanation", "Company Status", "Current Investors",
    "Our Investor Connections"
]

# Filter to only columns that exist in the dataframe
final_official_columns = [col for col in official_columns if col in df.columns]
official_df = df[final_official_columns].copy()

# Round the Normalized Score for the official output
if "Normalized Score" in official_df.columns:
    official_df.loc[:, "Normalized Score"] = official_df["Normalized Score"].round(2)

# Save the official company scores
#official_df.to_csv('official_company_scores_final.csv', index=False)
write_csv_to_s3(official_df, 'company-scoring', "official_company_scores_final.csv")

# STEP 6: Generate city and state distribution for top 200 companies using Affinity API
print("Step 6: Generating city and state distribution for top 200 companies from Affinity...")

# Affinity API Credentials
API_BASE_URL = "https://api.affinity.co"
API_VERSION_PATH = "/v2"
API_KEY = "_1EA5jw1tabci0t38DXPz-VcIW0Pz_CmTCbbMsPQsOg" # Replace if needed

# Initialize headers
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Sort companies by Normalized Score and take top 200
if 'Normalized Score' not in df.columns or not pd.api.types.is_numeric_dtype(df['Normalized Score']):
    print("Error: 'Normalized Score' column is missing or not numeric. Cannot sort.")
    exit()
else:
    df_sorted = df.dropna(subset=['Normalized Score'])
    top_200_companies = df_sorted.sort_values(by='Normalized Score', ascending=False).head(200)


# --- MODIFIED FUNCTION TO RETURN DICT {city, state} ---
def get_location_info(org_id):
    """Fetches company data including specific fields and extracts city and state."""
    if pd.isna(org_id):
        print(f"Warning: Received NaN org_id.")
        return {'city': None, 'state': None, 'error': 'NaN ID'} # Return None for both

    try:
        org_id_int = int(org_id)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert org_id '{org_id}' to integer. Skipping.")
        return {'city': None, 'state': None, 'error': 'Invalid ID Format'}

    location_field_id_param = "affinity-data-location"
    params = {"fieldIds": location_field_id_param}
    entity_url = f"{API_BASE_URL}{API_VERSION_PATH}/companies/{org_id_int}"

    try:
        response = requests.get(entity_url, headers=headers, params=params)

        if response.status_code == 404:
            print(f"Warning: Org ID {org_id_int} not found (404).")
            return {'city': None, 'state': None, 'error': 'Not Found (404)'}
        elif response.status_code == 403:
             print(f"Warning: Permission denied (403) for Org ID {org_id_int}.")
             return {'city': None, 'state': None, 'error': 'Permission Denied (403)'}
        elif response.status_code != 200:
            print(f"Warning: API Error for Org ID {org_id_int}. Status: {response.status_code}.")
            return {'city': None, 'state': None, 'error': f'API Error ({response.status_code})'}

        # Process successful response
        entity_data = response.json()
        entity_fields = entity_data.get("fields", [])

        if not entity_fields:
             print(f"Warning: 'fields' array missing/empty for {org_id_int}.")
             return {'city': None, 'state': None, 'error': 'Fields Missing in Response'}

        for field in entity_fields:
            if field.get("id") == location_field_id_param:
                location_value = field.get("value", {})
                if isinstance(location_value, dict):
                    # Extract data dict safely
                    data_dict = location_value.get("data", {})
                    if isinstance(data_dict, dict):
                        city = data_dict.get("city")
                        state = data_dict.get("state")
                        # Return None if value is null or empty string, otherwise return value
                        return {
                            'city': city if city else None,
                            'state': state if state else None
                        }
                    else:
                        #print(f"Warning: Unexpected location value.data format for {org_id_int}: {data_dict}")
                        return {'city': None, 'state': None, 'error': 'Unexpected Data Format'}

                elif location_value is None:
                    # Field exists but value is null
                    return {'city': None, 'state': None, 'error': 'Location Value is Null'}
                else:
                    print(f"Warning: Unexpected location value format for {org_id_int}: {location_value}")
                    return {'city': None, 'state': None, 'error': 'Unexpected Value Format'}

        # If the loop finishes without finding the field
        return {'city': None, 'state': None, 'error': 'Location Field Missing'}

    except requests.exceptions.RequestException as e:
        print(f"Error during API request for {org_id_int}: {e}")
        return {'city': None, 'state': None, 'error': 'Request Exception'}
    except Exception as e: # Catch JSON errors or other issues
        print(f"An unexpected error occurred processing {org_id_int}: {e}")
        return {'city': None, 'state': None, 'error': 'Unexpected Processing Error'}


# --- MODIFIED LOOP TO STORE CITY AND STATE ---
location_data_list = []
#print(f"Fetching location data for top {len(top_200_companies)} companies...")

for index, row in top_200_companies.iterrows():
    if 'Organization Id' not in row:
        #print(f"Skipping row {index} - 'Organization Id' column missing.")
        continue

    org_id = row["Organization Id"]
    company_name = row.get("Name", "Unknown Name")

    # Skip invalid IDs
    if pd.isna(org_id) or str(org_id).strip() == "" or str(org_id).lower() == "unknown":
        #print(f"Skipping {company_name} - Invalid or missing Organization ID: {org_id}")
        location_info = {'city': None, 'state': None} # Assign None if ID is invalid
    else:
        #print(f"Processing {company_name} (ID: {org_id})...")
        location_info = get_location_info(org_id) # Fetch the dict {city, state}

    # Append data including separate city and state
    location_data_list.append({
        "Organization Id": org_id,
        "Name": company_name,
        "City": location_info.get('city'),    # Get city from dict
        "State": location_info.get('state')   # Get state from dict
        # Optional: add location_info.get('error') here if you want to track errors in the df
    })

# Convert list of dicts to DataFrame
location_df = pd.DataFrame(location_data_list)

# --- GENERATE AND SAVE CITY DISTRIBUTION ---
# Use dropna=True (default) to exclude entries where city is None/NaN
city_distribution = location_df['City'].value_counts(dropna=True).reset_index()
city_distribution.columns = ['City', 'Count']
city_filename = 'top_200_companies_city_distribution.csv'
write_csv_to_s3(city_distribution, 'company-scoring', city_filename)


# --- GENERATE AND SAVE STATE DISTRIBUTION ---
# Use dropna=True (default) to exclude entries where state is None/NaN
state_distribution = location_df['State'].value_counts(dropna=True).reset_index()
state_distribution.columns = ['State', 'Count']
state_filename = 'top_200_companies_state_distribution.csv'
write_csv_to_s3(state_distribution, 'company-scoring', state_filename)



# STEP 7: Generate investor distribution for top 200 companies
print("Step 7: Generating investor distribution for top 200 companies...")

# Function to process investor strings and extract individual investors
def extract_investors(investor_string):
    """
    Extract individual investors from a semicolon-separated string
    Returns a list of cleaned investor names
    """
    if pd.isna(investor_string) or not isinstance(investor_string, str):
        return []
    
    # Split by semicolon and clean each investor name
    investors = [inv.strip() for inv in investor_string.split(';') if inv.strip()]
    
    # Filter out "N/A", "Unknown" or empty values
    investors = [inv for inv in investors if inv.lower() not in ["n/a", "unknown", ""]]
    
    return investors

# Create empty list to store all investors from top 200 companies
all_investors = []

# Extract investors from each company in the top 200
for _, row in top_200_companies.iterrows():
    if 'Current Investors' in row and pd.notna(row['Current Investors']):
        company_investors = extract_investors(row['Current Investors'])
        all_investors.extend(company_investors)

# Create a distribution (count) of all investors
investor_counts = pd.Series(all_investors).value_counts().reset_index()
investor_counts.columns = ['Investor', 'Count']

# Filter to only include investors that appear in at least 2 companies
investor_counts = investor_counts[investor_counts['Count'] >= 2]


# Sort by count (descending)
investor_counts = investor_counts.sort_values(by='Count', ascending=False)

# Save the investor distribution to CSV
investor_filename = 'top_200_companies_investor_distribution.csv'
write_csv_to_s3(investor_counts, 'company-scoring', investor_filename)



# --- FINAL OUTPUT MESSAGE ---
print("\nSuccessfully updated scoring and saved final outputs:")
print("1. subscores.csv")
print("2. subscores_distribution.csv")
print("3. raw_scores.csv")
print("4. official_company_scores_final.csv")
print(f"5. {city_filename}")
print(f"6. {state_filename}")
print(f"7. {investor_filename}")


# <hr style="border: 3px solid black;">
# 
# # JSON

# In[16]:


# import json
# from datetime import datetime
# import pytz

# # For saving general JSON data (not just dataframes)
# def write_dict_to_s3(data_dict, bucket, key):
#     json_str = json.dumps(data_dict, indent=4)
#     s3.put_object(Bucket=bucket, Key=key, Body=json_str)
    

# # STEP 6: Calculate and save additional metrics to JSON
# print("Step 6: Saving key metrics to JSON...")

# # Get maximum normalized score
# max_normalized_score = df['Normalized Score'].max() if 'Normalized Score' in df.columns else 0

# # Count companies above 75th percentile
# companies_above_75th = len(df[df['Normalized Score'] >= normalized_score_75th]) if 'Normalized Score' in df.columns else 0

# # Create metrics dictionary - only with the requested fields
# metrics = {
#     "normalized_score_75th_percentile": float(normalized_score_75th),
#     "max_normalized_score": float(max_normalized_score),
#     "companies_above_75th_percentile": int(companies_above_75th)
# }

# # Save metrics to JSON file
# # with open('scoring_metrics.json', 'w') as f:
# #     json.dump(metrics, f, indent=4)

# write_dict_to_s3(metrics, 'company-scoring', 'scoring_metrics.json')

# print(f"Saved key metrics to scoring_metrics.json:")
# print(f"- 75th percentile: {normalized_score_75th:.2f}")
# print(f"- Max score: {max_normalized_score:.2f}")
# print(f"- Companies above 75th percentile: {companies_above_75th}")

import json
from datetime import datetime
import pytz

# For saving general JSON data (not just dataframes)
def write_dict_to_s3(data_dict, bucket, key):
    json_str = json.dumps(data_dict, indent=4)
    s3.put_object(Bucket=bucket, Key=key, Body=json_str)
    
# STEP 6: Calculate and save additional metrics to JSON
print("Step 6: Saving key metrics to JSON...")
# Get maximum normalized score
max_normalized_score = df['Normalized Score'].max() if 'Normalized Score' in df.columns else 0
# Count companies above 75th percentile
companies_above_75th = len(df[df['Normalized Score'] >= normalized_score_75th]) if 'Normalized Score' in df.columns else 0

# Get current date in LA timezone with the specified format
la_timezone = pytz.timezone('America/Los_Angeles')
current_la_date = datetime.now(la_timezone)
# Format date like "11th April 2025"
day = current_la_date.day
# Add appropriate suffix to day
if 4 <= day <= 20 or 24 <= day <= 30:
    suffix = "th"
else:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
formatted_date = current_la_date.strftime(f"%B %-d{suffix} %Y")


# Create metrics dictionary with the new "Last Updated" field
metrics = {
    "normalized_score_75th_percentile": float(normalized_score_75th),
    "max_normalized_score": float(max_normalized_score),
    "companies_above_75th_percentile": int(companies_above_75th),
    "last_updated": formatted_date
}

# Save metrics to S3
write_dict_to_s3(metrics, 'company-scoring', 'scoring_metrics.json')

print(f"Saved key metrics to scoring_metrics.json:")
print(f"- 75th percentile: {normalized_score_75th:.2f}")
print(f"- Max score: {max_normalized_score:.2f}")
print(f"- Companies above 75th percentile: {companies_above_75th}")
print(f"- Last Updated: {formatted_date}")


# <hr style="border: 3px solid black;">
# 

# # Saving Last Round for Each Company in S3

# In[17]:


import json
from datetime import datetime
import pytz
# For saving general JSON data (not just dataframes)
def write_dict_to_s3(data_dict, bucket, key):
    json_str = json.dumps(data_dict, indent=4)
    s3.put_object(Bucket=bucket, Key=key, Body=json_str)
    
# STEP 6: Calculate and save additional metrics to JSON
print("Step 6: Saving key metrics to JSON...")
# Get maximum normalized score
max_normalized_score = df['Normalized Score'].max() if 'Normalized Score' in df.columns else 0
# Count companies above or equal to 75th percentile overall
companies_above_or_equal_75th = len(df[df['Normalized Score'] >= normalized_score_75th]) if 'Normalized Score' in df.columns else 0

# Count companies above or equal to 75th percentile by Last Round
companies_above_75th_by_round = {}
if 'Normalized Score' in df.columns and 'Last Round' in df.columns:
    # Filter companies above or equal to 75th percentile
    high_scoring_companies = df[df['Normalized Score'] >= normalized_score_75th]
    # Count by Last Round
    round_counts = high_scoring_companies['Last Round'].value_counts()
    
    # Convert to dictionary and handle NaN/missing values and VENTURE_UNKNOWN
    for round_type, count in round_counts.items():
        if pd.isna(round_type) or round_type == "N/A" or round_type == "":
            companies_above_75th_by_round["Unknown/Not Specified"] = int(count)
        elif str(round_type) == "VENTURE_UNKNOWN":
            # Add VENTURE_UNKNOWN to Series A
            if "SERIES_A" in companies_above_75th_by_round:
                companies_above_75th_by_round["SERIES_A"] += int(count)
            else:
                companies_above_75th_by_round["SERIES_A"] = int(count)
        else:
            companies_above_75th_by_round[str(round_type)] = int(count)
# Get current date in LA timezone with the specified format
la_timezone = pytz.timezone('America/Los_Angeles')
current_la_date = datetime.now(la_timezone)
# Format date like "11th April 2025"
day = current_la_date.day
# Add appropriate suffix to day
if 4 <= day <= 20 or 24 <= day <= 30:
    suffix = "th"
else:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
formatted_date = current_la_date.strftime(f"%B %-d{suffix} %Y")
# Create metrics dictionary with the new "Last Updated" field
metrics = {
    "normalized_score_75th_percentile": float(normalized_score_75th),
    "max_normalized_score": float(max_normalized_score),
    "companies_above_or_equal_75th_percentile": int(companies_above_or_equal_75th),
    "last_updated": formatted_date
}
# Save metrics to S3
write_dict_to_s3(metrics, 'company-scoring', 'scoring_metrics.json')

# Save high-scoring companies by round to separate JSON
high_scoring_by_round_metrics = {
    "companies_above_75th_by_last_round": companies_above_75th_by_round,
    "total_high_scoring_companies": sum(companies_above_75th_by_round.values()),
    "normalized_score_75th_percentile_threshold": float(normalized_score_75th),
    "last_updated": formatted_date
}
write_dict_to_s3(high_scoring_by_round_metrics, 'company-scoring', 'last_round_metrics.json')

print(f"Saved key metrics to scoring_metrics.json:")
print(f"- 75th percentile: {normalized_score_75th:.2f}")
print(f"- Max score: {max_normalized_score:.2f}")
print(f"- Companies above or equal to 75th percentile: {companies_above_or_equal_75th}")
print(f"- Last Updated: {formatted_date}")

print(f"\nSaved high-scoring companies by round to last_round_metrics.json:")
print(f"- Total high-scoring companies: {sum(companies_above_75th_by_round.values())}")
print("- Breakdown by funding round:")
for round_type, count in companies_above_75th_by_round.items():
    print(f"  - {round_type}: {count} companies")


# <hr style="border: 3px solid black;">
# 
# # Writing Back to Affinity

# In[9]:


import pandas as pd
import requests
import base64
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # <-- Progress bar

# ========= Affinity Configuration =========
API_BASE_URL = "https://api.affinity.co"

# Multiple API keys for load balancing
API_KEYS = [
    "_1EA5jw1tabci0t38DXPz-VcIW0Pz_CmTCbbMsPQsOg",  # Garv
    "SogEa2N5IGN8R9HuLQVjmE-iaDSfgKQr1OPIHEih40w",  # Koby
    "cS4CrWG3wJpcZiG5V5V1vQBI9c4kwBJZFI-LRqvtPJw",  # Michael
    "_1EA5jw1tabci0t38DXPz-VcIW0Pz_CmTCbbMsPQsOg",  # Nikhil
    "R1dgkjCc4U99xd3bAUKF6kJWwHvvscH4JvypwRpT26c",  # Stu
    "pSKLDwjX7sO6CJUE6F_DQ_mnJp6fGOB2fRWUc_cvSWk",  # Xiaolei
    "VQDCunt_0YzaIyE4jQ3UOR4SY1otaS9_FmUkeyogxAc",  # Justin
    "vULDrXTdXh1DAFwFl1mAtJdCHiyD2DVDPbNP3oIvqtc",  # Marci
    "4IqSCRl9pkJf6HS4b533TXFzaDok5rNGu5c1lTHGGyI",  # Akash
    "-yR7ZTXnRjiPnetN9RWnkfJ4YSuuHIwcIDpK-teGqEo",  # Ethan
]

# Track API key usage and rate limits
API_KEY_STATUS = {key: {"remaining": 1000, "reset": 0, "in_use": False} for key in API_KEYS}

# **Use actual field mappings instead of test fields**
FIELD_MAPPING = {
    "Company Score": 4834562,
    "Score Reasoning": 4957189,
    "Innovius Investor Connections": 4834690
}

# ========= API Key Management =========
def get_available_api_key():
    """Returns an available API key with the highest remaining requests."""
    current_time = time.time()
    available_keys = []
    
    for key, status in API_KEY_STATUS.items():
        # Reset the "in_use" flag if enough time has passed since the reset time
        if status["in_use"] and current_time > status["reset"]:
            status["in_use"] = False
            status["remaining"] = 1000  # Assume full quota after reset
        
        # Add keys that have remaining quota and aren't currently in use
        if status["remaining"] > 0 and not status["in_use"]:
            available_keys.append((key, status["remaining"]))
    
    if not available_keys:
        # If no keys available, find the one that will reset soonest
        next_reset = min(API_KEY_STATUS.items(), key=lambda x: x[1]["reset"])
        wait_time = max(0, next_reset[1]["reset"] - current_time)
        print(f"⏳ All API keys at limit. Waiting {wait_time:.1f} seconds for reset...")
        time.sleep(wait_time)
        # After waiting, mark that key as available again
        API_KEY_STATUS[next_reset[0]]["in_use"] = False
        API_KEY_STATUS[next_reset[0]]["remaining"] = 1000
        return next_reset[0]
    
    # Sort by remaining quota (highest first) and return the best key
    available_keys.sort(key=lambda x: x[1], reverse=True)
    return available_keys[0][0]

def create_session(api_key):
    """Creates a new session with the given API key."""
    auth_header = base64.b64encode(f":{api_key}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/json"
    }
    
    session = requests.Session()
    session.headers.update(headers)
    return session

# --- Rate Limit Settings ---
LOW_REMAINING_THRESHOLD = 10
TRANSIENT_ERRORS = {429, 500, 502, 503}

def request_with_retries(api_key, method, url, max_attempts=5, **kwargs):
    """Handles API requests with retries for transient errors and rate limits."""
    session = create_session(api_key)
    delay = 2
    
    for attempt in range(max_attempts):
        response = session.request(method, url, **kwargs)
        
        # Update API key status based on headers
        user_remaining = response.headers.get("X-Ratelimit-Limit-User-Remaining")
        user_reset = response.headers.get("X-Ratelimit-Limit-User-Reset")
        
        if user_remaining is not None:
            API_KEY_STATUS[api_key]["remaining"] = int(user_remaining)
        
        if user_reset is not None:
            API_KEY_STATUS[api_key]["reset"] = time.time() + int(user_reset)
        
        # Handle transient errors
        if response.status_code in TRANSIENT_ERRORS:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                delay = int(retry_after)
            else:
                delay = 2 * (attempt + 1)
            print(f"⚠️ Transient error ({response.status_code}) with key {api_key[-5:]}. Retrying in {delay} seconds...")
            time.sleep(delay)
            
            # Try a different API key if this one is having issues
            if attempt >= 2:
                new_key = get_available_api_key()
                print(f"Switching from key {api_key[-5:]} to {new_key[-5:]}")
                api_key = new_key
                session = create_session(api_key)
            
            continue
        
        # Check if we're getting low on remaining calls
        if user_remaining is not None and int(user_remaining) < LOW_REMAINING_THRESHOLD:
            # Mark this key as potentially problematic
            API_KEY_STATUS[api_key]["in_use"] = True
        
        return response, api_key
    
    return response, api_key  # Return last response if max_attempts exceeded

# ========== 1) HELPER: FETCH & DELETE FIELD-VALUES ==========
def get_org_field_values(org_id):
    """Fetches all field values for a given organization."""
    api_key = get_available_api_key()
    API_KEY_STATUS[api_key]["in_use"] = True
    
    url = f"{API_BASE_URL}/field-values?organization_id={org_id}"
    resp, used_key = request_with_retries(api_key, "GET", url)
    
    # Mark the key as available again
    API_KEY_STATUS[used_key]["in_use"] = False
    
    if resp.status_code != 200:
        raise ValueError(f"❌ Error fetching field-values for org {org_id}: {resp.status_code}, {resp.text}")
    return resp.json()

def delete_field_value(field_value_id):
    """Deletes a field value from Affinity."""
    api_key = get_available_api_key()
    API_KEY_STATUS[api_key]["in_use"] = True
    
    url = f"{API_BASE_URL}/field-values/{field_value_id}"
    r, used_key = request_with_retries(api_key, "DELETE", url)
    
    # Mark the key as available again
    API_KEY_STATUS[used_key]["in_use"] = False
    
    if r.status_code not in (200, 204):
        raise ValueError(f"Error deleting field value {field_value_id}: {r.status_code}, {r.text}")
    print(f"✅ Deleted field value {field_value_id} using key {used_key[-5:]}")

# ========== 2) UPDATE FIELD VALUES ==========
def update_single_field_value(field_id, org_id, new_value):
    """Updates a single-value field by deleting existing values and posting the new value."""
    all_values = get_org_field_values(org_id)
    relevant = [fv for fv in all_values if fv["field_id"] == field_id]
    
    # Delete existing values
    for fv in relevant:
        delete_field_value(fv["id"])

    # Get a fresh API key for the POST operation
    api_key = get_available_api_key()
    API_KEY_STATUS[api_key]["in_use"] = True
    
    payload = {
        "field_id": field_id,
        "entity_id": org_id,
        "value": new_value
    }
    url_post = f"{API_BASE_URL}/field-values"
    r, used_key = request_with_retries(api_key, "POST", url_post, json=payload)
    
    # Mark the key as available again
    API_KEY_STATUS[used_key]["in_use"] = False
    
    if r.status_code not in (200, 201):
        raise ValueError(f"Error setting field value: {r.status_code}, {r.text}")
    print(f"✅ Updated field {field_id} for org {org_id} to value: {new_value} using key {used_key[-5:]}")

def update_multi_select_field_values(field_id, org_id, new_values):
    """Updates a multi-select field by deleting existing values and adding new ones."""
    all_values = get_org_field_values(org_id)
    relevant = [fv for fv in all_values if fv["field_id"] == field_id]

    # Delete existing values
    for fv in relevant:
        delete_field_value(fv["id"])

    # Add new values
    for value in new_values:
        if not value:
            continue
            
        # Get a fresh API key for each value addition
        api_key = get_available_api_key()
        API_KEY_STATUS[api_key]["in_use"] = True
        
        payload = {
            "field_id": field_id,
            "entity_id": org_id,
            "value": value
        }
        url_post = f"{API_BASE_URL}/field-values"
        r, used_key = request_with_retries(api_key, "POST", url_post, json=payload)
        
        # Mark the key as available again
        API_KEY_STATUS[used_key]["in_use"] = False
        
        if r.status_code not in (200, 201):
            raise ValueError(f"Error adding multi-select field value: {r.status_code}, {r.text}")
        print(f"✅ Added '{value}' to multi-select field {field_id} for org {org_id} using key {used_key[-5:]}")

# ========== 3) PROCESS AN ORGANIZATION ==========
def process_org(row):
    """Processes a single organization by updating field values."""
    try:
        if pd.isna(row["Organization Id"]):
            return
        org_id = int(row["Organization Id"])

        if pd.notnull(row["Normalized Score"]):
            numeric_score = float(row["Normalized Score"])
            update_single_field_value(FIELD_MAPPING["Company Score"], org_id, numeric_score)

        if pd.notnull(row["Scoring Explanation"]):
            explanation = str(row["Scoring Explanation"])
            update_single_field_value(FIELD_MAPPING["Score Reasoning"], org_id, explanation)

        if pd.notnull(row["Our Investor Connections"]):
            investors = [x.strip() for x in row["Our Investor Connections"].split(";") if x.strip()]
            update_multi_select_field_values(FIELD_MAPPING["Innovius Investor Connections"], org_id, investors)

    except Exception as e:
        print(f"❌ Error processing org {row.get('Organization Id')}: {e}")

# ========== 4) MAIN SCRIPT ==========
def main():
    df = read_csv_from_s3('company-scoring', 'official_company_scores_final.csv')
    
    # Shuffle the dataframe to distribute load more evenly across API keys
    df = df.sample(frac=1).reset_index(drop=True)

    # Use ThreadPoolExecutor for parallel processing
    # Increase max_workers based on number of API keys available
    max_workers = min(10, len(API_KEYS))  # Use at most 10 workers, or fewer if we have fewer keys
    
    print(f"Starting processing with {max_workers} parallel workers using {len(API_KEYS)} API keys")
    
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            futures.append(executor.submit(process_org, row))

        # Display a progress bar for the processing
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing organizations"):
            try:
                future.result()
            except Exception as ex:
                print(f"Error in thread: {ex}")

    print("✅ Done! All data has been updated in Affinity.")
    

    def notify_success(message):
        sns = boto3.client('sns')
        sns.publish(
            TopicArn="arn:aws:sns:us-east-2:149536460887:company-scoring-updates",
            Subject="Company Scoring Completed",
            Message=message
        )

    # Build SNS summary for scoring
    summary_lines = [
        "Company Scoring Completed Successfully!"
    ]

    # Convert the list to a string before passing to the function
    summary = "\n".join(summary_lines)

    # Now pass the string to the function
    notify_success(summary)

if __name__ == "__main__":
    main()


# In[ ]:




