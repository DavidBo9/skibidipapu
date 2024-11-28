import openai
import os
from flask import Flask, request, jsonify
#from dotenv import load_dotenv

# Cargar las variables de entorno
#load_dotenv()

# Configurar la API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar la aplicación Flask
app = Flask(__name__)

# Mensaje del sistema para el modelo
messages = [
    {
        "role": "system",
        "content": "Eres un asistente diseñado para combinar resultados de análisis de imágenes y textos. Responde de manera coherente con base en ambos prompts y proporciona recomendaciones o interpretaciones relevantes basadas en los análisis. No generes contenido irrelevante o fuera de contexto."
    }
]

# Función para recibir los prompts y generar la respuesta
def procesar_prompts(prompt_imagen, prompt_texto):
    # Agregar los prompts del usuario al historial de mensajes
    messages.append({
        "role": "user",
        "content": f"Analisis de imagen: {prompt_imagen}"
    })
    messages.append({
        "role": "user",
        "content": f"Analisis de texto: {prompt_texto}"
    })

    # Llamar a la API de OpenAI con los mensajes
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    # Agregar la respuesta del modelo al historial
    messages.append({
        "role": "system",
        "content": response.choices[0].message.content
    })

    # Retornar el contenido de la respuesta
    return response.choices[0].message.content

# Ruta de la API para procesar los prompts
@app.route('/procesar-prompts', methods=['POST'])
def analizar_prompts():
    # Verificar que el cuerpo de la solicitud contenga los campos necesarios
    data = request.get_json()
    if 'prompt_imagen' not in data or 'prompt_texto' not in data:
        return jsonify({"error": "Los campos 'prompt_imagen' y 'prompt_texto' son requeridos"}), 400

    # Obtener los prompts de la solicitud
    prompt_imagen = data['prompt_imagen']
    prompt_texto = data['prompt_texto']

    # Generar la respuesta combinada
    resultado = procesar_prompts(prompt_imagen, prompt_texto)

    # Retornar la respuesta como JSON
    return jsonify({"resultado": resultado})

# Ejecutar la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)
