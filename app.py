from flask import Flask, render_template, request
from transformers import pipeline
from PIL import Image
import pytesseract
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # dossier pour stocker les images temporairement
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 👉 Indique ici le chemin vers tesseract.exe si nécessaire (si Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Charger le modèle
classifier = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1")

def clean_text(text):
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    extracted_text = ''
    
    if request.method == 'POST':
        user_text = request.form.get('user_text', '')
        image_file = request.files.get('image_file')

        # 📝 Si un fichier image est fourni
        if image_file and image_file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)
            
            # Utilisation de Tesseract pour extraire le texte
            extracted_text = pytesseract.image_to_string(Image.open(image_path))
            user_text = extracted_text  # On remplace user_text par le texte extrait

        if user_text.strip():
            prediction = classifier(clean_text(user_text))[0]
            result = {
                'label': prediction['label'],
                'score': f"{prediction['score']:.4f}",
                'extracted_text': extracted_text
            }

    return render_template('index.html', result=result)
    
if __name__ == '__main__':
    app.run(debug=True)
