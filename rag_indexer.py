from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# --- KROK 1: Načtení dokumentu ---
loader = TextLoader("dokument.txt", encoding="utf-8")
documents = loader.load()

print(f"Načteno dokumentů: {len(documents)}")
print(f"Délka textu: {len(documents[0].page_content)} znaků\n")

# --- KROK 2: Rozdělení na chunky ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # max znaků na chunk
    chunk_overlap=40      # překryv — kontext se nepřeruší na hranici chunku
)

chunks = splitter.split_documents(documents)

print(f"Počet chunků: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk.page_content}")

# --- KROK 3: Embeddings + uložení do FAISS ---
print("\nVytvářím embeddings a ukládám do FAISS...")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Ulož na disk
vectorstore.save_local("faiss_index")

print("Hotovo — index uložen do složky faiss_index/")