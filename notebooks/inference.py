import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os

print(os.getcwd() + "\n")

# Device and dtype setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

# Load the model and tokenizer from your checkpoint
CHECKPOINT_PATH = "checkpoints/moondream-ducci-dataset-with-vision-encoder-ft-2024-12-30-12-03-01"
MD_REVISION = "2024-07-23"

# Initialize tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

# Load your fine-tuned model
moondream = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE,
    device_map={"": DEVICE}
)

def process_image(image_path):
    """
    Process an image for inference
    """
    image = Image.open(image_path)
    return moondream.encode_image(image)

def generate_response(image_embedding, prompt):
    """
    Generate a response based on the image and prompt
    """
    response = moondream.answer_question(image_embedding, prompt, tokenizer=tokenizer, num_beams=4, 
                                         no_repeat_ngram_size=5, early_stopping=True )
    return response

# Example usage
def main():
    # Example image path - replace with your actual image path
    image_path = "hololens_sample_img.jpg"
    
    # Process the image
    image_embedding = process_image(image_path)
    
    # Example prompt - replace with your actual prompt
    prompt = "What can you see in this image?"
    
    # Generate response
    response = generate_response(image_embedding, prompt)
    print(f"\n\nResponse: {response}")

if __name__ == "__main__":
    main()