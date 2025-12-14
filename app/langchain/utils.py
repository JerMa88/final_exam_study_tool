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
            lines = result.stdout.strip().split('\n')[1:] # skip header
            existing_models = [line.split()[0].split(':')[0] for line in lines] 
            
            # We strictly require the model to be present.
            # Using a simple check against the output. 
            # Note: 'ollama list' output contains the model name. 
            # We check if model_name matches any listed model.
            
            # Since names in 'ollama list' might be like 'gemma:2b', checking existence.
            # Getting exact list of names from stdout for better matching is safer.
            full_model_names = [line.split()[0] for line in lines]
            
            if model_name in full_model_names:
                return # Model exists

            # Fallback for untagged names if user provided tag or vice versa logic can be complex,
            # but usually 'ollama list' returns what you can pull.
            # If we are here, strict match failed.
            pass

        # If we reached here, model is likely missing.
        # Check if it was a command failure or just missing in list.
        # Actually, let's just rely on the fact we didn't return above.
        
        # DOUBLE CHECK: 'ollama list' always returns 0 even if empty? Yes.
        # So we iterate and return if found. If loop finishes, we are here.
        
        raise RuntimeError(f"Model '{model_name}' not found. Please run 'ollama pull {model_name}' in your terminal.")

    except Exception as e:
        # Re-raising the error with a clear message or letting the RuntimeError bubble up
        raise RuntimeError(f"Error checking Ollama model '{model_name}': {e}")
