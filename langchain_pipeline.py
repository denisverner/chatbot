from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()  # převede response objekt na čistý string

# --- KROK 1: Extrakce tématu ---
prompt_topic = ChatPromptTemplate.from_messages([
    ("system", "Extrahuj hlavní téma z textu uživatele. Vrať pouze téma, 2-4 slova, nic jiného."),
    ("human", "{user_input}")
])

# --- KROK 2: Generování odpovědi na základě tématu ---
prompt_answer = ChatPromptTemplate.from_messages([
    ("system", "Jsi technický mentor. Vysvětli téma srozumitelně, maximálně 3 věty."),
    ("human", "Vysvětli mi toto téma: {topic}")
])

# --- PIPELINE: krok1 → krok2 ---
# StrOutputParser převede výstup kroku 1 na string → ten jde jako {topic} do kroku 2
pipeline = (
    prompt_topic | model | parser  # výstup = string s tématem
    | (lambda topic: {"topic": topic})  # zabalí string do slovníku pro krok 2
    | prompt_answer | model | parser    # výstup = finální odpověď
)

# --- Spuštění ---
test_inputs = [
    "Nevím jak funguje paměť v LangChainu a proč se používá session_id",
    "Zajímalo by mě co jsou to vektorové databáze a k čemu slouží",
    "Chci pochopit rozdíl mezi RAG a fine-tuningem"
]

for user_input in test_inputs:
    print(f"Vstup:    {user_input}")
    result = pipeline.invoke({"user_input": user_input})
    print(f"Výstup:   {result}")
    print("-" * 60)