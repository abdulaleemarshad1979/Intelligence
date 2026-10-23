import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests must never write to the real record databases.
import tempfile
os.environ.setdefault("CCTV_DB_DIR", tempfile.mkdtemp(prefix="cctv_test_db_"))
