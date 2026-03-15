import os
from google import genai
from dotenv import load_dotenv

from typing import Optional
load_dotenv()

# Inicializamos el nuevo cliente
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def obtain_ai_analysis(ai_data: dict) -> Optional[str]:
    lang_instruction = "Respond ALWAYS in English." if ai_data.get("language") == "english" else "Responde SIEMPRE en Español."
    title = "Elite Analysis" if ai_data.get("language") == "english" else "Análisis de Élite"
    prompt = f"""
        SYSTEM ROLE:
        You are a Senior Data Analyst and Padel Expert (World Padel Tour/Premier Padel specialist). Your goal is to explain a Machine Learning match prediction to a professional audience.

        INPUT DATA (JSON format):
        {ai_data}

        TECHNICAL CONTEXT FOR YOUR ANALYSIS:
        - Court Speed & Environment: High altitude, high temperature, and low humidity significantly increase ball bounce and speed (favoring aggressive smashers/power players). Low temperature and high humidity make the ball "heavy" and slow (favoring defensive players/tactical lobs).
        - Key Metrics: 
            * 'diff_avg_height': Advantage in reach and leverage for smashes.
            * 'diff_comeback_rate': Mental toughness and resilience when losing.
            * 'diff_tie_break_win_pct': Performance under high-pressure "clutch" moments.
            * 'match_quality_sum': Overall level of the match based on combined rankings.

        INSTRUCTIONS:
        1. Direct Analysis: Explain WHY the winner is predicted based on the most significant 'features' provided.
        2. Environmental Impact: Explicitly mention how the environmental conditions (Altitude: {ai_data['altitude']}m, Temp: {ai_data['temperature']}°C, Humidity: {ai_data['humidity']}%) affect this specific match-up.
        3. Tone: Professional, authoritative, and engaging (like an ESPN or Sky Sports commentator). Avoid saying "the model says" or "the data suggests"; state it as a professional insight.
        4. Constraints: Maximum 3-4 sentences.
        5. Language: {lang_instruction}

        OUTPUT STRUCTURE:
        "{title}: [Your insight here in the stated language]"
    """
    try:
        # La nueva sintaxis de llamada
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error en IA: {e}")
        return "Analysis unavailable." if ai_data.get("language") == "english" else "Análisis no disponible."