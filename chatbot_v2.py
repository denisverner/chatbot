from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Systémový prompt — definuje "osobnost" chatbota
system_prompt = "Jsi přátelský asistent pro výuku vaření. Odpovídáš stručně a srozumitelně. Pokud nevíš, přiznej to."

# Historie konverzace — tady se ukládají všechny zprávy
conversation_history = [
    {"role": "system", "content": system_prompt}
]

print("Chatbot spuštěn. Napiš 'konec' pro ukončení.\n")

while True:
    user_input = input("Ty: ")
    
    if user_input.lower() == "konec":
        print("Nashledanou!")
        break
    
    # Přidej zprávu uživatele do historie
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Pošli CELOU historii modelu — tak si "pamatuje" kontext
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    
    assistant_message = response.choices[0].message.content
    
    # Přidej odpověď do historie
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    print(f"Asistent: {assistant_message}\n")