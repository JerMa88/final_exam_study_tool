import subprocess
import shutil

def ensure_ollama_model(model_name: str):
    """
    Checks if an Ollama model is available, and raises RuntimeError if not.
    Uses 'ollama list' to check presence.
    Handles both tagged (e.g., 'gemma4:26b') and untagged (e.g., 'embeddinggemma') names.
    """
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("Warning: 'ollama' executable not found in PATH.")
        return

    try:
        result = subprocess.run([ollama_path, "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # skip header
            
            for line in lines:
                if not line.strip():
                    continue
                listed_name = line.split()[0]  # e.g., "embeddinggemma:latest"
                base_name = listed_name.split(':')[0]  # e.g., "embeddinggemma"
                
                # Match exact full name OR base name (for untagged requests)
                if model_name == listed_name or model_name == base_name:
                    return  # Model exists
        
        raise RuntimeError(
            f"Model '{model_name}' not found. "
            f"Please run 'ollama pull {model_name}' in your terminal."
        )

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error checking Ollama model '{model_name}': {e}")

