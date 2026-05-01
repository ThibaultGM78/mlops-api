import pandas as pd
from flask import Flask, request, jsonify

from src.model import Model

app = Flask(__name__)
model = Model()


@app.route("/", methods=["GET"])
def home():
    # At beginning, we load model from MLflow
    return ("OK !", 200)

"""
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    df = pd.read_json(body)
    results = [int(x) for x in model.predict(df).flatten()]
    return (jsonify(results), 200)
"""

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Récupère les données (c'est une liste Python)
        body = request.get_json()
        
        # 2. Convertis la liste directement en DataFrame
        # On n'utilise PAS read_json ici
        df = pd.DataFrame(body)
        
        # 3. Appel de la méthode predict de ta classe Model
        # Elle va gérer le pipeline et les colonnes à supprimer
        predictions = model.predict(df)
        
        # 4. Retourne les résultats
        results = [int(x) for x in predictions.flatten()]
        return jsonify({"predictions": results}), 200
        
    except Exception as e:
        # En cas d'erreur, on renvoie le message pour débugger plus vite
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000)