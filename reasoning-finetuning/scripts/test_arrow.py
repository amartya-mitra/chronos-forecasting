print("Importing pyarrow...")
try:
    import pyarrow
    print(f"PyArrow version: {pyarrow.__version__}")
    print("PyArrow imported successfully.")
except Exception as e:
    print(f"Failed to import pyarrow: {e}")
