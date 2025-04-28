import sacrebleu

expected_translations = [
    "yo como carne",
    "¿quién eres?",
    "En nuestra casa de estudios",
    "El profesor y sus alumnos están en el aula.",
    "Estamos bien, mi profesor. ¿Y tú?",
    "Yo, por mi parte, vivo en San Isidro.",
    "Yo suelo cocinar para mi mamá, ¿y tú, oye, Ricardo?",
    "Sí, yo suelo jugar fútbol.",
    "¿Cuándo vas a terminar la revista?",
    "¿Cómo está?"
]

model_translations = [
    "yo como carne",
    "quien eres?",
    "en nuestra casa de estudios",
    "el profesor y sus alumnos están en el aula",
    "estamos bien, mi profesor. y tú?",
    "yo vivo en san isidro",
    "yo cocino para mi mama, y tú, ricardo?",
    "sí, juego fútbol",
    "cuándo vas a terminar la revista?",
    "cómo estás?"
]

for ref, model_output in zip(expected_translations, model_translations):
    score = sacrebleu.sentence_chrf(model_output, [ref])
    print(f"Reference: {ref}")
    print(f"Model Output: {model_output}")
    print(f"BLEU score:" ,sentence_bleu(ref.split(), model_output.split()))
    print(f"ChrF++ score: {score.score:.2f}")
    print("-" * 10)
