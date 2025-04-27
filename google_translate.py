import google.generativeai as genai

genai.configure(api_key="")

model_name = 'models/gemini-1.5-flash-latest'
model = genai.GenerativeModel(model_name)

phrases = [
    ("ñuqa aycha-ta-m miku-ni", "yo como carne"),
    ("Pitaq kanki?", "¿quién eres?"),
    ("Yachay wasinchikpi", "En nuestra casa de estudios"),
    ("Yachachiq yachachisqakunapas yachay wasi ukupi kachkanku.", "El profesor y sus alumnos están en el aula."),
    ("Allinllam, yachachiqniy, qamrí?", "Estamos bien, mi profesor. ¿Y tú?"),
    ("Ñuqataq San Isidropi tiyachkani.", "Yo, por mi parte, vivo en San Isidro."),
    ("Ñuqaqa mamaypaq yanuqmi kani, qamrí, yaw Ricardo?", "Yo suelo cocinar para mi mamá, ¿y tú, oye, Ricardo?"),
    ("Arí, ñuqaqa futbolpi pukllaqmi kani.", "Sí, yo suelo jugar fútbol."),
    ("Haykaptaq qawaytarí tukunki?", "¿Cuándo vas a terminar la revista?"),
    ("Imaynataq kachkan?", "¿Cómo está?")
]

for idx, (source_text, expected_translation) in enumerate(phrases, start=1):
    prompt = f"Translate the following phrase from Quechua to Spanish without any additional commentary, only the direct translation please:\n\n'{source_text}'"
    
    try:
        response = model.generate_content([prompt])
        gemini_translation = response.text.strip()
        
        print(f"Phrase {idx}:")
        print(f"Quechua: {source_text}")
        print(f"Gemini Translation: {gemini_translation}")
        print(f"Expected Spanish: {expected_translation}")
        print("-" * 10)
        
    except Exception as e:
        print(f"An error occurred for phrase {idx}: {e}")
