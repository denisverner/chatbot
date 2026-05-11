from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

system_prompt = "Jsi přátelský asistent pro výuku programování. Odpovídáš stručně a srozumitelně."

conversation_history = [
    {"role": "system", "content": system_prompt}
]

def ask(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

def summarize(text):
    result = ask([
        {"role": "system", "content": "Shrň následující text do 3 vět. Pouze shrnutí, nic jiného."},
        {"role": "user", "content": text}
    ])
    print(f"\n Shrnutí:\n{result}\n")

def translate(text):
    result = ask([
        {"role": "system", "content": "Přelož následující text do angličtiny. Pouze překlad, nic jiného."},
        {"role": "user", "content": text}
    ])
    print(f"\n Překlad:\n{result}\n")

def extract_keywords(text):
    result = ask([
        {"role": "system", "content": "Extrahuj klíčová slova z textu. Vrať pouze seznam slov oddělených čárkou, nic jiného."},
        {"role": "user", "content": text}
    ])
    print(f"\n Klíčová slova:\n{result}\n")

print("Chatbot spuštěn.")
print("Příkazy: /shrnout <text> | /prelozit <text> | /klicovaslova <text> | konec\n")

while True:
    user_input = input("Ty: ").strip()

    if user_input.lower() == "konec":
        print("Nashledanou!")
        break
    elif user_input.startswith("/shrnout "):
        summarize(user_input[9:])
    elif user_input.startswith("/prelozit "):
        translate(user_input[10:])
    elif user_input.startswith("/klicovaslova "):
        extract_keywords(user_input[14:])
    else:
        # Normální konverzace s historií
        conversation_history.append({"role": "user", "content": user_input})
        reply = ask(conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Asistent: {reply}\n")