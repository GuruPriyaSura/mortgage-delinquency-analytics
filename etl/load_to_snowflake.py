# Loads Freddie Mac CSV files into Snowflake RAW schema.
# Usage: python etl/load_to_snowflake.py

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import os
import glob
from dotenv import load_dotenv

load_dotenv()

ORIGINATION_COLS = [
    'CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG',
    'MATURITY_DATE', 'MSA', 'MI_PERCENTAGE', 'NUMBER_OF_UNITS',
    'OCCUPANCY_STATUS', 'ORIGINAL_CLTV', 'ORIGINAL_DTI',
    'ORIGINAL_UPB', 'ORIGINAL_LTV', 'ORIGINAL_INTEREST_RATE',
    'CHANNEL', 'PREPAYMENT_PENALTY_FLAG', 'PRODUCT_TYPE',
    'PROPERTY_STATE', 'PROPERTY_TYPE', 'POSTAL_CODE',
    'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE', 'ORIGINAL_LOAN_TERM',
    'NUMBER_OF_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME',
    'SUPER_CONFORMING_FLAG'
]

PERFORMANCE_COLS = [
    'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD',
    'CURRENT_ACTUAL_UPB', 'CURRENT_LOAN_DELINQUENCY_STATUS',
    'LOAN_AGE', 'REMAINING_MONTHS_TO_LEGAL_MATURITY',
    'REPURCHASE_FLAG', 'MODIFICATION_FLAG', 'ZERO_BALANCE_CODE',
    'ZERO_BALANCE_EFFECTIVE_DATE', 'CURRENT_INTEREST_RATE',
    'CURRENT_DEFERRED_UPB', 'DUE_DATE_OF_LAST_PAID_INSTALLMENT',
    'MI_RECOVERIES', 'NET_SALES_PROCEEDS', 'NON_MI_RECOVERIES',
    'EXPENSES', 'LEGAL_COSTS', 'MAINTENANCE_AND_PRESERVATION_COSTS',
    'TAXES_AND_INSURANCE', 'MISCELLANEOUS_EXPENSES',
    'ACTUAL_LOSS_CALCULATION', 'MODIFICATION_COST'
]

def get_connection():
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )
    print("Connected!")
    return conn

def load_origination_files(conn):
    files = glob.glob('data/raw/historical_data_[0-9]*.txt')
    files = [f for f in files if 'time' not in f]

    if not files:
        print("No origination files found in data/raw/")
        return

    print(f"\nFound {len(files)} origination file(s)...")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        print(f"\nLoading: {filename}")

        df = pd.read_csv(filepath, sep='|', names=ORIGINATION_COLS,
                         low_memory=False)
        print(f"  Rows: {len(df):,}")

        df['CREDIT_SCORE'] = df['CREDIT_SCORE'].replace([9999, 999], pd.NA)
        df['ORIGINAL_DTI'] = df['ORIGINAL_DTI'].replace(999, pd.NA)

        success, nchunks, nrows, _ = write_pandas(
            conn, df, 'ORIGINATION',
            database='MORTGAGE_DB', schema='RAW',
            auto_create_table=True, overwrite=False
        )

        if success:
            print(f"  Uploaded {nrows:,} rows")
        else:
            print(f"  Upload failed for {filename}")

def load_performance_files(conn):
    files = glob.glob('data/raw/historical_data_time_*.txt')

    if not files:
        print("No performance files found in data/raw/")
        return

    print(f"\nFound {len(files)} performance file(s)...")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        print(f"\nLoading: {filename}")

        chunk_size = 500_000
        total_rows = 0

        for chunk_num, chunk in enumerate(
            pd.read_csv(filepath, sep='|', names=PERFORMANCE_COLS,
                        low_memory=False, chunksize=chunk_size)
        ):
            cols_to_keep = [
                'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD',
                'CURRENT_ACTUAL_UPB', 'CURRENT_LOAN_DELINQUENCY_STATUS',
                'LOAN_AGE', 'CURRENT_INTEREST_RATE'
            ]
            chunk = chunk[cols_to_keep]

            success, _, nrows, _ = write_pandas(
                conn, chunk, 'PERFORMANCE',
                database='MORTGAGE_DB', schema='RAW',
                auto_create_table=True,
                overwrite=(chunk_num == 0)
            )
            total_rows += nrows
            print(f"  Chunk {chunk_num+1}: {nrows:,} rows", end='\r')

        print(f"\n  Total: {total_rows:,} rows")

def run_staging_transform(conn):
    print("\nRunning staging transformation...")

    staging_sql = """
    INSERT INTO MORTGAGE_DB.ANALYTICS.FACT_LOAN_MONTHLY
    SELECT
        p.LOAN_SEQUENCE_NUMBER,
        TRY_TO_DATE(p.MONTHLY_REPORTING_PERIOD, 'MM/YYYY'),
        o.PROPERTY_STATE,
        o.CREDIT_SCORE,
        o.ORIGINAL_LTV,
        o.ORIGINAL_DTI,
        o.ORIGINAL_INTEREST_RATE,
        o.ORIGINAL_UPB,
        p.LOAN_AGE,
        p.CURRENT_ACTUAL_UPB,
        p.CURRENT_LOAN_DELINQUENCY_STATUS,
        CASE
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS = '0' THEN 'Current'
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS = '1' THEN '30-day'
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS = '2' THEN '60-day'
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS IN ('3','4','5','6') THEN '90+ day'
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS IN ('RA','F') THEN 'Foreclosure'
            ELSE 'Other'
        END,
        CASE
            WHEN o.CREDIT_SCORE < 620 THEN 'High Risk'
            WHEN o.CREDIT_SCORE < 680 THEN 'Medium Risk'
            WHEN o.CREDIT_SCORE < 740 THEN 'Low Risk'
            ELSE 'Very Low Risk'
        END,
        CASE
            WHEN p.CURRENT_LOAN_DELINQUENCY_STATUS IN ('3','4','5','6','RA','F') THEN 1
            ELSE 0
        END,
        o.LOAN_PURPOSE,
        o.SELLER_NAME
    FROM MORTGAGE_DB.RAW.PERFORMANCE p
    JOIN MORTGAGE_DB.RAW.ORIGINATION o
        ON p.LOAN_SEQUENCE_NUMBER = o.LOAN_SEQUENCE_NUMBER
    WHERE o.CREDIT_SCORE IS NOT NULL
      AND o.ORIGINAL_LTV IS NOT NULL
      AND o.ORIGINAL_DTI IS NOT NULL
    """

    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE MORTGAGE_DB.ANALYTICS.FACT_LOAN_MONTHLY")
    cursor.execute(staging_sql)
    print("Staging transform complete!")
    cursor.close()

if __name__ == "__main__":
    conn = get_connection()
    load_origination_files(conn)
    load_performance_files(conn)
    run_staging_transform(conn)
    conn.close()
    print("\nAll done! Data is in Snowflake and ready for Power BI.")