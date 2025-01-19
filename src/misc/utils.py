import os


def load_dotenv(dotenv_path=".env"):
    """Load environment variables from a .env file."""
    try:
        with open(dotenv_path) as f:
            for line in f:
                # Remove comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split the line into key and value
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip(' "')
    except FileNotFoundError:
        print(f"{dotenv_path} file not found.")
