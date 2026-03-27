import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# --- Configuration ---
load_dotenv()  # Securely load from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_DIR = "faiss_index"

# Initialize singletons for LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=API_KEY
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY,
)


def load_vectorstore() -> FAISS:
    """Loads the FAISS index from disk. Requires running ingest.py first."""
    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"Index directory '{FAISS_INDEX_DIR}/' not found. "
            "Please run 'python ingest.py' first to build the database."
        )

    print("📂 Loading FAISS index from disk...")
    store = FAISS.load_local(
        folder_path=FAISS_INDEX_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ FAISS index successfully loaded ({store.index.ntotal} vectors)")
    return store


# Load the vector store at startup
vectorstore = load_vectorstore()


def retrieve_relevant_documents(question: str, top_k: int = 6) -> list:
    """
    Retrieves the most relevant and diverse documents for a given question using 
    Maximal Marginal Relevance (MMR) and score-based filtering.
    """
    candidate_docs = vectorstore.max_marginal_relevance_search(
        question,
        k=top_k,
        fetch_k=20,
        lambda_mult=0.7,
    )

    scored_docs = vectorstore.similarity_search_with_score(question, k=top_k)
    score_threshold = 1.5
    highly_relevant_ids = {
        doc.page_content for doc, score in scored_docs if score < score_threshold
    }

    final_docs = [doc for doc in candidate_docs if doc.page_content in highly_relevant_ids]
    return final_docs if final_docs else candidate_docs[:4]


def format_context_for_prompt(docs: list) -> str:
    """Formats retrieved document chunks into a clean, readable context string."""
    formatted_parts = []
    
    for doc in docs:
        meta = doc.metadata
        doc_type = meta.get("type", "text").upper()
        page = meta.get("page", "?")
        section = meta.get("section", "")
        subsection = meta.get("subsection", "")
        topic = meta.get("topic", "")
        
        header = f"[{doc_type} — Page {page}]"
        if section:
            header += f" — Section: {section}"
            if subsection:
                header += f" > {subsection}"
        if topic:
            header += f" (Topic: {topic})"
            
        # Prioritize pure string over embedding string (which contains prepended headers)
        content = meta.get("original_content", doc.page_content)
        formatted_parts.append(f"{header}\n{content}")
        
    return "\n\n---\n\n".join(formatted_parts)


def build_prompt(context: str, question: str) -> str:
    """Constructs the exact instruction prompt for the LLM."""
    return f"""You are an intelligent, and youre name is CHatway a conversational assistant. Answer the user's question accurately using ONLY the information in the provided Context below.

CRITICAL TONE INSTRUCTIONS:
1. Be entirely natural and direct. NEVER use robotic phrases like "Selon le contexte fourni...", "D'après les documents...", or "Le document ne mentionne pas...".
2. Speak with authority as if you inherently possess this knowledge. 
3. If the answer is NOT in the Context, do NOT state that the document is lacking info. Instead, respond naturally and politely (e.g., "Je n'ai pas les détails exacts à ce sujet, mais...") and offer related known facts if applicable.
4. Answer in the same language as the user's question (e.g. French).
5. Cleanly integrate metrics or page numbers naturally within your sentences without sounding rigid.

--- Context ---
{context}

--- User Question ---
{question}

Answer cleanly and naturally:"""


def generate_answer(question: str) -> str:
    """End-to-end function to retrieve context and generate an answer."""
    relevant_docs = retrieve_relevant_documents(question)
    context = format_context_for_prompt(relevant_docs)
    prompt = build_prompt(context, question)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
