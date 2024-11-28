import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Este analiza la imagen y detecta la emoción de la foto
messages = [
    {
        "role": "system",
        "content": "Eres un asistente diseñado para detectar emociones en textos. Identifica las dos emociones principales presentes y, con base en ellas, proporciona recomendaciones personalizadas, como contenido, palabras de apoyo, libros, canciones, actividades o cualquier recurso relevante que pueda ayudar al usuario. No respondas a textos que no transmitan emociones o que no estén relacionados con el análisis emocional. Evita proporcionar información irrelevante o fuera del contexto emocional del texto"
    }
]

def recibir_mensaje(mensaje):
    messages.append({
        "role": "user",
        "content": mensaje
    })

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Corrige el modelo si era un error tipográfico.
        messages=messages
    )

    messages.append({
        "role": "system",
        "content": response.choices[0].message.content
    })

    return response.choices[0].message.content

# Capturar texto del usuario
texto_usuario = input("Ingresa un texto para analizar: ")

# Analizar el texto usando la función definida
resultado = recibir_mensaje(texto_usuario)

# Mostrar el resultado
print("Resultado del análisis emocional:")
print(resultado)
