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
    """Constructs a high-quality expert instruction prompt for the LLM."""
    return f"""You are Chatway, an expert in the subject of the provided document: urban logistics and electric trucks (last-mile delivery, environmental impact, operational constraints, and economic analysis).

You answer with precision, clarity, and authority.

Your task: answer the user's question using ONLY the information from the provided Context.

CORE BEHAVIOR:
- Act as a domain expert in urban logistics and electric freight transport.
- Provide accurate, structured, and insightful answers.
- When relevant, include key figures, constraints, trade-offs, or performance metrics.

TONE & STYLE:
- Natural, fluid, and human — never robotic.
- Do NOT reference the existence of the context or document.
- Do NOT say “according to the context” or similar.
- Speak with confidence, as if this knowledge is yours.

LIMITATION HANDLING:
- If the answer is not explicitly in the Context:
  - Do NOT mention missing documents.
  - Respond naturally with partial knowledge:
    "Je n’ai pas les détails exacts, mais…" and provide relevant insight.

LANGUAGE:
- Always respond in the same language as the user.

STRUCTURE:
- Start with a direct answer.
- Then add concise explanation or key supporting points.

--- Context ---
{context}

--- User Question ---
{question}

Answer:"""

def generate_answer(question: str) -> str:
    """End-to-end function to retrieve context and generate an answer."""
    relevant_docs = retrieve_relevant_documents(question)
    context = format_context_for_prompt(relevant_docs)
    prompt = build_prompt(context, question)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
