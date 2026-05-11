from google import genai
from dotenv import load_dotenv
import os

# načtení .env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Gemini je připraven! (napiš 'exit' pro ukončení)\n")

# historie
chat = client.chats.create(model="gemini-3-flash-preview")

while True:
    user_input = input("Ty: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Tak Ahoj")
        break

    response = chat.send_message(user_input)

    print("Chatbot:", response.text)