import os
import json
import base64
import io
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from pdf2image import convert_from_path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# --- Configuration ---
PDF_FILE = "data.pdf"
FAISS_INDEX_DIR = "faiss_index"
DPI = 200

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Initialize Models ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY
)


# --- Helper Functions ---

def convert_image_to_base64(img) -> str:
    """Converts a PIL Image to a base64 encoded JPEG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json_from_llm_response(raw_text: str) -> List[Dict[str, Any]]:
    """Cleans up markdown code blocks from the response and parses it as JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove the first line (e.g., ```json) and the closing ```
        lines = text.split("\n")
        if len(lines) > 2 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
            if text.rfind("```") != -1:
                text = text[:text.rfind("```")]
    return json.loads(text.strip())


def extract_chunks_from_image(img, page_num: int, retries: int = 2) -> List[Dict[str, Any]]:
    """Uses Gemini Vision to extract structured semantic chunks from a page image."""
    base64_image = convert_image_to_base64(img)
    
    prompt = """You are a document analysis expert. Analyze this PDF page image and extract its content as **structured semantic chunks**.

Rules:
1. Return a JSON array of chunks. Each chunk is one semantic unit (a paragraph, a chart, a table, a list, a key finding).
2. DO NOT return raw OCR text. Clean and restructure the content.
3. For charts/graphs: extract ALL data points, labels, values, percentages, and relationships. Describe what the chart shows.
4. For tables: convert to clean structured text with all values preserved.
5. For text: clean up OCR artifacts, fix encoding issues, and maintain the original meaning in proper French.
6. **CRITICAL**: Each chunk MUST accurately identify its `section` and `subsection` to preserve hierarchical context. If the page continues a previous section, infer the section name.

Return ONLY valid JSON in this exact format (no markdown, no commentary):
[
  {
    "content": "the clean, readable text content of this chunk",
    "type": "text | chart | table | list",
    "section": "the main section heading this belongs to",
    "subsection": "the subsection heading, or empty string",
    "topic": "a 1-3 word summary of the chunk's topic"
  }
]

If the page is a title page, copyright page, or mostly empty, return a single chunk with the relevant info.
If you cannot extract meaningful content, return an empty array: []"""

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
    ])

    for attempt in range(retries + 1):
        try:
            response = llm.invoke([message])
            chunks = parse_json_from_llm_response(response.content)
            
            # Tag each chunk with its source page number
            for chunk in chunks:
                chunk["page"] = page_num
                
            return chunks

        except Exception as e:
            if attempt < retries:
                print(f"    ⚠ Retry {attempt + 1} for page {page_num} (error: {e})")
                time.sleep(2)
            else:
                print(f"    ❌ Failed after {retries + 1} attempts on page {page_num}: {e}")
                return []


def clean_and_filter_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters out empty or garbage chunks, and ensures content is stringified."""
    cleaned = []
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # Ensure content is a string
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
            chunk["content"] = content
            
        # Keep only chunks with meaningful text length
        if content.strip() and len(content.strip()) > 20:
            cleaned.append(chunk)
            
    return cleaned


def save_chunks_backup(chunks: List[Dict[str, Any]], filename: str = "chunks.json"):
    """Saves the extracted chunks to a JSON file for backup/debugging."""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def print_extraction_summary(chunks: List[Dict[str, Any]]):
    """Prints a summary of the extracted chunk types."""
    types_count = {}
    for chunk in chunks:
        chunk_type = chunk.get("type", "text")
        types_count[chunk_type] = types_count.get(chunk_type, 0) + 1

    print(f"\n✅ Extraction Complete!")
    print(f"   {len(chunks)} total chunks embedded into {FAISS_INDEX_DIR}/")
    print(f"   Breakdown by type: {types_count}")
    print(f"   Backup saved to chunks.json")


# --- Main Pipeline ---

def build_vector_index():
    """Main pipeline execution: PDF to Images -> Gemini Extraction -> FAISS Index."""
    print(f"📄 Converting '{PDF_FILE}' to images (DPI={DPI})...")
    images = convert_from_path(PDF_FILE, dpi=DPI)
    total_pages = len(images)
    print(f"   → Found {total_pages} pages.\n")

    all_extracted_chunks = []

    # Phase 1: Structured Information Extraction
    print("🔍 Phase 1: Extracting content with Gemini Vision...")
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"  Processing Page {page_num}/{total_pages}...", end=" ", flush=True)

        page_chunks = extract_chunks_from_image(img, page_num)
        all_extracted_chunks.extend(page_chunks)
        print(f"→ Extracted {len(page_chunks)} chunks.")

        # Respect API rate limits
        if page_num < total_pages:
            time.sleep(1)

    print(f"\n   Total raw chunks extracted: {len(all_extracted_chunks)}")

    # Phase 2: Data Cleaning
    valid_chunks = clean_and_filter_chunks(all_extracted_chunks)
    print(f"   Chunks after filtering: {len(valid_chunks)}")

    # Phase 3: Vector Store Creation
    print(f"\n🔨 Phase 3: Building FAISS index...")
    
    # Prepend contextual metadata to the embedding representation to dramatically improve retrieval preciseness
    texts = []
    for chunk in valid_chunks:
        meta_prefix = f"Topic: {chunk.get('topic', '')}\n"
        if chunk.get("section"):
            meta_prefix += f"Section: {chunk['section']}\n"
        if chunk.get("subsection"):
            meta_prefix += f"Subsection: {chunk['subsection']}\n"
        texts.append(meta_prefix + chunk["content"])
        
    metadatas = [
        {
            "type": chunk.get("type", "text"),
            "section": chunk.get("section", ""),
            "subsection": chunk.get("subsection", ""),
            "topic": chunk.get("topic", ""),
            "page": chunk.get("page", 0),
            "original_content": chunk["content"]  # Keep raw content for LLM generation
        }
        for chunk in valid_chunks
    ]

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(FAISS_INDEX_DIR)

    # Phase 4: Backup & Summary
    save_chunks_backup(valid_chunks)
    print_extraction_summary(valid_chunks)


if __name__ == "__main__":
    build_vector_index()
