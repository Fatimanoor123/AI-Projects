from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

#Load Pre-trained Model from Hugging face
# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = AutoFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Check if a GPU is available and use it, otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device (GPU if available, else CPU)
model.to(device)

#Upload and Preprocess Images
from google.colab import files
uploaded = files.upload()

image_path=next(iter(uploaded))

#Preprocess method for  uploaded imaged for model
 
def preprocess_image(image_path):
    image=Image.open(image_path)
    #Pre_process the image such as resizin etc
    pixel_values=feature_extractor(images=image,return_tensors="pt", ).pixel_values
    pixel_values=pixel_values.to(device)
    return pixel_values

# Preprocess the uploaded image and move it to the selected device (GPU or CPU)
pixel_values = preprocess_image(image_path).to(device)

def generate_caption(pixel_values):
    max_length = 16  # Set the maximum length of the generated caption
    num_beams = 4    # Set the number of beams for beam search (higher value = better results but slower)
    # Use the model to generate caption IDs for the input image
    caption_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)

    # Decode the generated caption IDs into a readable string
    caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
     
     # Return the decoded caption 
    return caption 

# Generate and display the caption for the image
caption = generate_caption(pixel_values)
print(f"Generated Caption: {caption}")  # Print the generated caption
