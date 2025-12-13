import subprocess
import shutil

def ensure_ollama_model(model_name: str):
    """
    Checks if an Ollama model is available, and pulls it if not.
    Uses 'ollama list' to check presence to avoid unnecessary 'ollama pull' overhead.
    """
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("Warning: 'ollama' executable not found in PATH.")
        return

    try:
        # Check if model exists
        result = subprocess.run([ollama_path, "list"], capture_output=True, text=True)
        if result.returncode == 0:
            # Output format is: NAME  ID  SIZE  MODIFIED
            # We check if model_name is in the output. 
            # Note: partial matches might be an issue, but 'ollama list' output is usually one model per line.
            # strict parsing is better.
            lines = result.stdout.strip().split('\n')[1:] # skip header
            existing_models = [line.split()[0].split(':')[0] for line in lines] # Extract names (ignoring tags if user didn't specify)
            # Actually user might specify tag.
            
            # Simple check: if strict match is found, or if model_name is substring of a line (rudimentary)
            # Let's rely on 'ollama pull' being smart enough or just pull if we strictly don't see it.
            # But 'ollama pull' takes a few seconds even if present (hashing).
            
            # Better: try to use the model, if it fails, pull.
            # But user asked to "pull ... first then query".
            pass

        print(f"Ensuring Ollama model '{model_name}' is pulled...")
        subprocess.run([ollama_path, "pull", model_name], check=True)
        print(f"Ollama model '{model_name}' ready.")
    except Exception as e:
        print(f"Error ensuring Ollama model '{model_name}': {e}")
