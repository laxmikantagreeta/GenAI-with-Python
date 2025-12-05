"""Configuration helpers for the Gemini-powered documentation agent."""

import os

from dotenv import load_dotenv

def load_api_key(env_var: str = "GEMINI_API_KEY") -> str:
    """Load the Gemini API key from environment variables or a .env file.

    Args:
        env_var: Environment variable name that stores the API key.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the API key is missing.
    """

    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        raise EnvironmentError(
            f"Set the {env_var} environment variable or add it to a .env file."
        )
    return api_key
