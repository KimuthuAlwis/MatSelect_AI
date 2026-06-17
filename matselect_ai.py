import json
from google import genai
from google.genai import types

# Initialize the Gemini Client
# It will automatically look for an environment variable named GEMINI_API_KEY,
# or you can pass it directly like: client = genai.Client(api_key="YOUR_KEY")
client = genai.Client()

def parse_engineering_prompt(user_prompt: str) -> dict:
    """
    Uses Gemini 2.5 Flash to convert natural language engineering requests
    into structured numerical filters and search constraints.
    """
    
    system_instruction = """
    You are an expert Materials Selection Assistant. Your job is to translate a user's informal engineering requirement 
    into a structured JSON block containing specific constraints and weights for a materials database query.
    
    You must output ONLY valid JSON matching this exact structure:
    {
        "max_density": float or null,
        "min_tensile": float or null,
        "min_youngs": float or null,
        "corrosion_priority": "High" or "Normal",
        "explanation": "A brief 1-sentence engineering justification for these choices."
    }
    
    Rules for interpretation:
    - "Lightweight" or "low weight" -> set max_density around 2.5 to 3.0 g/cm³
    - "Strong" or "high stress" -> set min_tensile around 300 to 500 MPa
    - "Stiff" or "rigid" -> set min_youngs around 70 to 200 GPa
    - "Won't rust" or "marine environment" or "corrosion resistant" -> set corrosion_priority to "High"
    """

    try:
        # We use gemini-2.5-flash as it is lightning fast and perfect for structured JSON extraction
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Analyze this engineering requirement: '{user_prompt}'",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # Force the model to output strictly valid JSON
                response_mime_type="application/json",
                temperature=0.1, # Low temperature makes the output highly predictable and stable
            ),
        )
        
        # Parse the string output straight into a Python dictionary
        constraints = json.loads(response.text)
        return constraints

    except Exception as e:
        # Fallback dictionary if something goes wrong or connection fails
        return {
            "max_density": None,
            "min_tensile": None,
            "min_youngs": None,
            "corrosion_priority": "Normal",
            "explanation": f"Failed to call Gemini API: {str(e)}"
        }