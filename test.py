import pandas as pd
import requests

# 1. Charger les données
df = pd.read_csv("../purchase-predict/data/03_primary/primary.csv")
df = df.drop(["user_session", "user_id", "purchased"], axis=1, errors='ignore')

# 2. Remplacer les NaN par une valeur par défaut (ex: 0 ou "unknown")
# C'est crucial car le JSON ne peut pas transporter de NaN
# df = df.fillna(0) 
# Au lieu de df.fillna(0), on traite les types séparément
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('unknown') # Pour le texte
    else:
        df[col] = df[col].fillna(0)         # Pour les nombres

# 3. Prendre 5 lignes au format records
sample_data = df.sample(n=5).to_dict(orient='records')

# 4. Envoyer la requête
url = "http://35.205.192.172/predict"
print(f"Envoi de {len(sample_data)} lignes à {url}...")

try:
    response = requests.post(url, json=sample_data)
    if response.status_code == 200:
        print("Succès ! Prédictions :")
        print(response.json())
    else:
        print(f"Erreur {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Erreur lors de la requête : {e}")