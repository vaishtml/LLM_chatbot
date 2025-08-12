# LLM_Playground/test_inference.py

import os
from huggingface_hub import InferenceClient
import argparse

def test_model_inference(prompt, model_name):
    """
    Tests model inference using an environment variable for authentication.
    """
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_api_token:
        print("Error: HUGGINGFACEHUB_API_TOKEN environment variable not set.")
        return

    print(f"--- Testing model: {model_name} ---")
    try:
        client = InferenceClient(model=model_name, token=hf_api_token)
        response = client.text_generation(prompt=prompt, max_new_tokens=100)

        print("\n[Prompt]")
        print(prompt)
        print("\n[Model Response]")
        print(response)
        print("\n--- Test complete ---")

    except Exception as e:
        print(f"An error occurred during the inference test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Hugging Face Model Inference.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to the model.")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="The model ID to test."
    )
    
    args = parser.parse_args()
    test_model_inference(args.prompt, args.model)