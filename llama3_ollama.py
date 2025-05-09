import requests
import json
import time

def generate_response(prompt, model="llama2:70b", temperature=0.7):
    """
    Generate a response using Ollama's API
    
    Args:
        prompt (str): The input prompt
        model (str): The model to use (default: llama2:70b)
        temperature (float): Controls randomness (0.0 to 1.0)
    
    Returns:
        str: The generated response
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def main():
    # Example usage
    prompt = "Explain quantum computing in simple terms."
    print("Generating response...")
    response = generate_response(prompt)
    
    if response:
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main() 