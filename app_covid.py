import streamlit as st
import joblib
import numpy as np

# URL directa al archivo .pkl en GitHub (usa el enlace RAW)
MODEL_URL = "https://github.com/JoseSC2025/predictor_covid_mortalidad/blob/main/best_model.pkl" 

# Cargar modelo
with open('best_model.pkl', 'rb') as f:
    model = joblib.load(f)

st.set_page_config(page_title="Predicción de Mortalidad por COVID-19")

st.title("🦠 Predicción de Mortalidad por COVID-19")
st.markdown("Ingrese los datos del paciente para predecir el riesgo de mortalidad.")

# Variables del modelo (modifica según tu modelo real)
# Ejemplo de variables binarias:
dicotomicas = {
    "fiebre_Si": None,
    "malestar_gen_Si": None,
    "tos_Si": None,
    "dolor_garganta_Si": None,
    "congestion_nasal_Si": None,
    "dificultad_resp_Si": None,
    "diarrea_Si": None,
    "vomitos_Si": None,	
}

# Ingreso de datos
input_data = []

# Edad
edad = st.slider("Edad", 0, 120, 20)
input_data.append(edad)

# Sexo
sexo = st.radio("Sexo", ["Hombre", "Mujer"])
input_data.append(1 if sexo == "Hombre" else 0)  # Asumiendo Hombre=1, Mujer=0

# Dicotómicas
st.subheader("Síntomas / Comorbilidades")
for var in dicotomicas:
    respuesta = st.radio(f"{var}:", ["No", "Sí"], horizontal=True)
    input_data.append(1 if respuesta == "Sí" else 0)

# Botón de predicción
if st.button("🔍 Predecir"):
    X = np.array(input_data).reshape(1, -1)
    pred = model.predict(X)[0]
    st.subheader("Resultado de la predicción:")
    st.success("✅ No muerte") if pred == 0 else st.error("⚠️ Sí muerte")

