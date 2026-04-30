from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Model
model = ChatOpenAI(model="gpt-4o-mini")

# 2. Prompt template — {topic} je proměnná kterou dosadíš
prompt = ChatPromptTemplate.from_messages([
    ("system", "Odpovídej stručně, maximálně 2 věty."),
    ("user", "Vysvětli mi co je {topic}.")
])

# 3. Chain — spojení promptu a modelu pomocí |
chain = prompt | model

# 4. Spuštění
response = chain.invoke({"topic": "vektorové databáze"})

print(response.content)