from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# Prompt s místem pro historii konverzace
prompt = ChatPromptTemplate.from_messages([
    ("system", "Jsi technický asistent. Odpovídáš stručně a česky."),
    MessagesPlaceholder(variable_name="history"),  # sem se vloží historie
    ("human", "{input}")
])

chain = prompt | model

# Úložiště historií — každá session_id má vlastní historii
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chain obalený pamětí
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Konfigurace session — stejné session_id = stejná paměť
config = {"configurable": {"session_id": "denis_session"}}

print("Konverzační chatbot s pamětí. Napiš 'konec' pro ukončení.\n")

while True:
    user_input = input("Ty: ").strip()
    
    if user_input.lower() == "konec":
        break
    
    response = chain_with_memory.invoke(
        {"input": user_input},
        config=config
    )
    
    print(f"Asistent: {response.content}\n")