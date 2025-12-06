"""
Configuration file for Trading Journal Database Project

Easily switch between databases:
- Database_project.db (Midterm - synthetic data)
- Trading_Journal_ML.db (Masters - real market data + ML models)

Auto-downloads database from Google Drive if not present locally.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database selection
USE_ML_DATABASE = True  # Set to False to use midterm database

# Database paths
MIDTERM_DB = os.path.join(BASE_DIR, "Database_project.db")
ML_DB = os.path.join(BASE_DIR, "Trading_Journal_ML.db")

# Google Drive file ID for auto-download
GDRIVE_FILE_ID = "1R7zBpf_6pEJ7PQEucGjkbH8CkaB4JBGj"

def ensure_database_exists():
    """Download database from Google Drive if it doesn't exist locally"""
    if USE_ML_DATABASE and not os.path.exists(ML_DB):
        print("=" * 60)
        print("DATABASE NOT FOUND - AUTO-DOWNLOADING FROM GOOGLE DRIVE")
        print("=" * 60)
        print(f"\nDatabase file: {ML_DB}")
        print("This is a one-time download (~881 MB)")
        print("Please wait...\n")

        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, ML_DB, quiet=False)
            print("\n" + "=" * 60)
            print("DATABASE DOWNLOADED SUCCESSFULLY!")
            print("=" * 60 + "\n")
        except ImportError:
            print("\nERROR: gdown package not installed.")
            print("Please run: pip install gdown")
            print("Then restart the application.\n")
            raise SystemExit(1)
        except Exception as e:
            print(f"\nERROR: Failed to download database: {e}")
            print("Please download manually from:")
            print(f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
            print(f"And save it as: {ML_DB}\n")
            raise SystemExit(1)

# Auto-download database on import
ensure_database_exists()

# Active database
DB_PATH = ML_DB if USE_ML_DATABASE else MIDTERM_DB

# Display which database is active
def get_active_db_info():
    """Return info about active database"""
    if USE_ML_DATABASE:
        return {
            'name': 'Trading_Journal_ML.db',
            'description': 'Masters-level database with real market data',
            'path': DB_PATH
        }
    else:
        return {
            'name': 'Database_project.db',
            'description': 'Midterm database with synthetic data',
            'path': DB_PATH
        }

if __name__ == "__main__":
    info = get_active_db_info()
    print(f"Active Database: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Path: {info['path']}")
