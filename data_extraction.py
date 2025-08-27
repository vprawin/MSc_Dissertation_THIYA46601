#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_extraction.py
------------------
pyodbc to pull TWO datasets:
  1) Defect dataset from the SAME SOURCE DATABASE used by SAP BO/WebI (via ODBC)
  2) Active user list from an operational SQL database (via ODBC)

Outputs:
  - Defect Dataset.xlsx
  - 250524_ACTIVE_USERS_v1.xlsx


Usage:
  pip install pyodbc pandas openpyxl
  python "data_extraction.py"
"""

import os
import sys
import getpass
import pandas as pd



# ===================================================================
# CONFIG A — SAP BO "DEFECT" DATASET (via pyodbc to the source DB)
# ===================================================================

BO_DSN         = os.getenv("BO_DSN", "_REMOVED_DUE_TO_CONFIDENTIALITY_")  
BO_DRIVER      = os.getenv("BO_DRIVER", "{ODBC Driver 17 for SQL Server}")
BO_SERVER      = os.getenv("BO_SERVER", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
BO_DATABASE    = os.getenv("BO_DATABASE", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
BO_USERNAME    = os.getenv("BO_USERNAME", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
BO_PASSWORD    = os.getenv("BO_PASSWORD", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
BO_TRUST_CERT  = os.getenv("BO_TRUST_CERT", "Yes")

DEFECTS_XLSX   = "Defect Dataset.xlsx"

# Example SQL query (replace with your own!)
DEFECTS_SQL = r"""
SELECT
    *
FROM dbo.Defects AS d;
"""

# ===================================================================
# CONFIG B — ACTIVE USERS DATASET (via pyodbc to SQL DB)
# ===================================================================
AU_DSN         = os.getenv("AU_DSN", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
AU_DRIVER      = os.getenv("AU_DRIVER", "{ODBC Driver 17 for SQL Server}")
AU_SERVER      = os.getenv("AU_SERVER", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
AU_DATABASE    = os.getenv("AU_DATABASE", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
AU_USERNAME    = os.getenv("AU_USERNAME", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
AU_PASSWORD    = os.getenv("AU_PASSWORD", "_REMOVED_DUE_TO_CONFIDENTIALITY_")
AU_TRUST_CERT  = os.getenv("AU_TRUST_CERT", "Yes")

ACTIVE_USERS_XLSX = "250524_ACTIVE_USERS_v1.xlsx"

ACTIVE_USERS_SQL = r"""
SELECT
    *
FROM dbo.Users AS u;
"""

# ===================================================================
# Helpers
# ===================================================================

def build_conn_str(dsn, driver, server, database, username, password, trust_cert="Yes"):
    """Build ODBC connection string."""
    if dsn:
        return f"DSN={dsn};UID={username};PWD={password};"
    return (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate={trust_cert};"
    )

def fetch_dataframe(conn_str, sql, label):
    """Connect, run SQL, return DataFrame."""
    import pyodbc
    print(f"[*] Connecting to {label} ...")
    conn = pyodbc.connect(conn_str, timeout=30)
    df = pd.read_sql(sql, conn)
    conn.close()
    print(f"    Retrieved {len(df)} rows, {len(df.columns)} cols from {label}")
    return df

def save_excel(df, path, label):
    """Save DataFrame to Excel."""
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="ignore")
    df.to_excel(path, index=False)
    print(f"    Saved {label} to {path}")

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=== Data Extraction (pyodbc) ===")

    # Defects dataset
    bo_conn = build_conn_str(BO_DSN, BO_DRIVER, BO_SERVER, BO_DATABASE, BO_USERNAME, BO_PASSWORD, BO_TRUST_CERT)
    df_defects = fetch_dataframe(bo_conn, DEFECTS_SQL, "DEFECTS")
    save_excel(df_defects, DEFECTS_XLSX, "DEFECTS")

    # Active users dataset
    au_conn = build_conn_str(AU_DSN, AU_DRIVER, AU_SERVER, AU_DATABASE, AU_USERNAME, AU_PASSWORD, AU_TRUST_CERT)
    dataset_sampled = fetch_dataframe(au_conn, ACTIVE_USERS_SQL, "ACTIVE USERS")
    save_excel(dataset_sampled, ACTIVE_USERS_XLSX, "ACTIVE USERS")

    print("=== Done. Active users DataFrame available as `dataset_sampled`. ===")
