from PIL import Image
import pytesseract

# Path to the Tesseract executable (you need to specify the correct path on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load an image containing characters for recognition
input_image_path = r'C:\Users\ABHI\Downloads\pattern regogneetion\img.jpg'
output_text = ""

# Open the image using Pillow
image = Image.open(input_image_path)

# Perform OCR on the image
try:
    output_text = pytesseract.image_to_string(image)
except Exception as e:
    print(f"Error during OCR: {str(e)}")

# Print the recognized text
print("Recognized Text:")
print(output_text)
