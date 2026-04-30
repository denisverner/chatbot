from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
import sys

load_dotenv()

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

def build_vectorstore(filepath):
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    print(f"📄 Načteno {len(chunks)} chunků z {filepath}")
    return FAISS.from_documents(chunks, embeddings)

def load_vectorstore():
    vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Index načten z disku — {vs.index.ntotal} vektorů")
    return vs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Odpovídej pouze na základě poskytnutého kontextu.
Pokud odpověď v kontextu není, řekni 'Tuto informaci v dokumentu nemám.'

Kontext:
{context}"""),
        MessagesPlaceholder(variable_name="history"),  # paměť konverzace
        ("human", "{question}")
    ])

    # Retriever potřebuje čistý string — vytáhneme ho ze slovníku
    retriever_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
    )

    chain = retriever_chain | prompt | model | parser

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

# --- Hlavní logika ---
if len(sys.argv) > 1:
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"❌ Soubor {filepath} nenalezen.")
        sys.exit(1)
    vectorstore = build_vectorstore(filepath)
else:
    vectorstore = load_vectorstore()

rag_chain = get_rag_chain(vectorstore)
config = {"configurable": {"session_id": "rag_session"}}

print("\nRAG chatbot s pamětí. Napiš 'konec' pro ukončení.\n")

while True:
    question = input("Otázka: ").strip()
    if not question:
        continue
    if question.lower() == "konec":
        break

    answer = rag_chain.invoke({"question": question}, config=config)
    print(f"\nOdpověď: {answer}\n")
    print("-" * 60)