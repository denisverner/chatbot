from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# --- Načtení indexu z disku (bez generování embeddings) ---
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Index načten z disku — {vectorstore.load_local .index.ntotal} vektorů")

# --- Retriever — hledá relevantní chunky ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # vrátí 2 nejrelevantnější chunky

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """Odpovídej pouze na základě poskytnutého kontextu.
Pokud odpověď v kontextu není, řekni 'Tuto informaci v dokumentu nemám.'

Kontext:
{context}"""),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# --- RAG pipeline ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,  # otázka → retriever → chunky → string
        "question": RunnablePassthrough()    # otázka projde beze změny
    }
    | prompt
    | model
    | parser
)

# --- Interaktivní smyčka ---
print("RAG chatbot — ptej se na obsah dokumentu. Napiš 'konec' pro ukončení.\n")

while True:
    question = input("Otázka: ").strip()

    if question.lower() == "konec":
        break

    answer = rag_chain.invoke(question)
    print(f"\nOdpověď: {answer}\n")
    print("-" * 60)