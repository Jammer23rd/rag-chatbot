from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredEPubLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import warnings

# Configure logging and warnings first
logging.basicConfig(level=logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("filetype").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Set environment variables for underlying libraries
os.environ["UNSTRUCTURED_HIDE_LOAD_PROGRESS"] = "true"
os.environ["UNSTRUCTURED_LOGGING_MODE"] = "silent"

load_dotenv()

DOCS_DIR = "documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

FILE_HANDLERS = [
    # Text formats
    (TextLoader, ["**/*.txt", "**/*.log"]),
    (UnstructuredMarkdownLoader, ["**/*.md"]),

    # Code files
    (TextLoader, [
        "**/*.py", "**/*.js", "**/*.java",
        "**/*.c", "**/*.cpp", "**/*.h"
    ]),

    # Office documents
    (PyPDFLoader, ["**/*.pdf"]),
    (UnstructuredWordDocumentLoader, ["**/*.docx"]),
    (UnstructuredPowerPointLoader, ["**/*.pptx"]),
    (UnstructuredExcelLoader, ["**/*.xlsx"]),

    # Web formats - modified to suppress HTML output
    (UnstructuredHTMLLoader, ["**/*.html", "**/*.htm"], {"mode": "single", "silent_errors": True}),

    # Data formats
    (JSONLoader, ["**/*.json"], {"jq_schema": ".content"}),
    (CSVLoader, ["**/*.csv"]),

    # Ebooks
    (UnstructuredEPubLoader, ["**/*.epub"]),

    # Other formats
    (UnstructuredFileLoader, ["**/*.rtf", "**/*.odt"])
]

def load_documents():
    """Load documents with silent error handling"""
    docs = []
    print("\n🔍 Scanning documents directory...")

    for handler in FILE_HANDLERS:
        loader_cls = handler[0]
        patterns = handler[1]
        loader_args = handler[2] if len(handler) > 2 else {}

        for pattern in patterns:
            file_ext = pattern.split("*")[-1]
            try:
                with tqdm(
                    desc=f"📂 {file_ext.upper()} files",
                    unit="file",
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
                ) as pbar:
                    loader = DirectoryLoader(
                        DOCS_DIR,
                        glob=pattern,
                        loader_cls=loader_cls,
                        loader_kwargs=loader_args,
                        show_progress=False,
                        use_multithreading=True,
                        silent_errors=True
                    )
                    loaded = loader.load()

                    if loaded:
                        print(f"✅ Loaded {len(loaded)} {file_ext} files")
                        docs.extend(loaded)
                        pbar.total = len(loaded)
                        pbar.update(len(loaded))
                    else:
                        pbar.update(0)

            except Exception as e:
                continue  # Silent error handling

    return docs

def split_documents(documents):
    """Split documents with silent processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    print("\n🔪 Splitting documents...")
    chunks = []

    with tqdm(
        total=len(documents),
        desc="📄 Processing documents",
        unit="doc",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        disable=True  # Silence progress bars
    ) as pbar:
        for doc in documents:
            try:
                chunks.extend(text_splitter.split_documents([doc]))
            except Exception as e:
                continue
            pbar.update(1)

    return chunks

logging.getLogger("unstructured").setLevel(logging.CRITICAL)

if __name__ == "__main__":
    print("🚀 Starting document processing pipeline")
    print("🌈 Supported formats:", ", ".join(sorted({p.split(".")[-1] for h in FILE_HANDLERS for p in h[1]})))

    # Load documents
    documents = load_documents()
    print(f"\n📦 Total documents loaded: {len(documents)}")

    if not documents:
        print("❌ No documents found - exiting")
        exit(1)

    # Split documents
    chunks = split_documents(documents)
    print(f"\n📄 Total chunks created: {len(chunks)}")

    # Create embeddings
    print("\n🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # Create and persist vector store
    print("\n💾 Building vector database...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "db")
        )
        print("✅ Vector database saved successfully")

        if chunks:
            print("\n🔍 Sample chunk preview:")
            print("-" * 50)
            print(chunks[0].page_content[:200] + "...")
            print("-" * 50)
            print(f"📝 Source: {chunks[0].metadata.get('source', 'unknown')}")
            print(f"🔢 Chunk size: {len(chunks[0].page_content)} characters")

    except Exception as e:
        print(f"❌ Database creation failed: {str(e)}")
        exit(1)

    print("\n🎉 Processing complete! You can now run talk.py")
