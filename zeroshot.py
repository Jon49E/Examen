#pip install transformers, openai
from transformers import pipeline
from flask import Flask, request, jsonify #framework de desarrollo para un servicio web, datos request, enviar respuestas desde el lado del servidor json
#pip install flask_cors
from flask_cors import CORS #restringir las solicitudes http


app = Flask(__name__)
CORS(app)

@app.route('/clasificar', methods=['POST'])

def classify_text():
    texto = request.form.get('texto', '')
    candidate_labels = ["Religion", "Cine", "Politica", 'Deporte']
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    
    resultado_clasificacion = classifier(texto, candidate_labels)
    max_score = resultado_clasificacion ['scores'].index(max(resultado_clasificacion['scores']))
    label_score = resultado_clasificacion['labels'][max_score]
    
    return jsonify ({'label': label_score})

if __name__ == '__main__':
    app.run(debug=False, host='localhost', port=5011)
