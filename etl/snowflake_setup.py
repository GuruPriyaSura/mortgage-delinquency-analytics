# Run this ONCE to create all tables in Snowflake.
# Usage: python etl/snowflake_setup.py

import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return snowflake.connector.connect(
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
    )

def run_setup():
    print("Connecting to Snowflake...")
    conn = get_connection()
    cursor = conn.cursor()

    with open('sql/snowflake_setup.sql', 'r') as f:
        sql_content = f.read()

    statements = [s.strip() for s in sql_content.split(';')
                  if s.strip() and not s.strip().startswith('--')]

    for i, statement in enumerate(statements):
        if statement:
            try:
                cursor.execute(statement)
                print(f"  [{i+1}/{len(statements)}] OK")
            except Exception as e:
                print(f"  [{i+1}/{len(statements)}] ERROR: {e}")

    cursor.close()
    conn.close()
    print("\nSnowflake setup complete!")

if __name__ == "__main__":
    run_setup()