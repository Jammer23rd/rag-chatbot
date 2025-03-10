from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
import logging
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ChatLogger:
    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log = self._new_log_file()

    def _new_log_file(self) -> str:
        return os.path.join(
            self.log_dir,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    def log_interaction(self, user_input: str, response: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response
        }
        try:
            with open(self.current_log, "a") as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Logging failed: {str(e)}")

class DocumentAnalyzer:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.domain_summary = self._analyze_documents()
        self.capabilities = self._generate_capabilities()

    def _get_document_texts(self) -> List[str]:
        """Get text content from documents"""
        results = self.vector_db.get()
        return results["documents"] if results else []

    def _extract_key_phrases(self, texts: List[str]) -> List[str]:
        """Extract important phrases using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Get top 20 phrases
            sums = tfidf_matrix.sum(axis=0)
            return [feature_names[col] for col in sums.argsort()[0, -20:][::-1].flat]
        except Exception as e:
            logger.error(f"TF-IDF failed: {str(e)}")
            return []

    def _analyze_documents(self) -> str:
        """Generate domain expertise summary locally"""
        try:
            texts = self._get_document_texts()
            if not texts:
                return "various technical topics"

            # Extract key phrases
            phrases = self._extract_key_phrases(texts)

            # Count noun phrases
            counter = Counter(phrases)
            top_terms = [term for term, _ in counter.most_common(7)]

            return f"{', '.join(top_terms[:-1])}, and {top_terms[-1]}"

        except Exception as e:
            logger.error(f"Local analysis failed: {str(e)}")
            return "technical documentation and resources"

    def _generate_capabilities(self) -> str:
        """Create capabilities list from terms"""
        terms = self.domain_summary.split(', ')
        bullets = [
            f"• Questions about {term}"
            for term in terms[:5]  # Use first 5 terms
        ]
        bullets.append("• General technical assistance")
        return "\n".join(bullets)

class ChatBot:
    def __init__(self, force_cpu=False):
        print("🔧 Initializing RAG system...")
        self.force_cpu = force_cpu
        self.logger = ChatLogger()
        self.history: List[Dict] = []

        try:
            # Initialize components
            self.embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                model_kwargs={'device': 'cpu'} if force_cpu else {}
            )

            self.vector_db = Chroma(
                persist_directory=os.getenv("CHROMA_PERSIST_DIR", "db"),
                embedding_function=self.embeddings
            )

            # Verify documents exist
            if self.vector_db._collection.count() == 0:
                raise ValueError("No documents found. Run load_and_split_docs.py first!")

            # Initialize LLM only for responses
            self.llm = ChatOpenAI(
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("MODEL_NAME", "deepseek-chat"),
                temperature=0.3
            )

            self.analyzer = DocumentAnalyzer(self.vector_db)
            self._build_chains()

            print(f"✅ Ready! I can help with:\n{self.analyzer.capabilities}")
            print("\nType '/exit' to quit\n")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _build_chains(self):
        """Create processing pipelines"""
        self.main_chain = (
            ChatPromptTemplate.from_template(
                """Context:
                {context}

                Conversation History:
                {history}

                Question: {question}
                Answer concisely based on the documents:"""
            )
            | self.llm
            | StrOutputParser()
        )

        self.greeting_chain = (
            ChatPromptTemplate.from_template(
                """Create a friendly greeting mentioning these topics: {topics}"""
            )
            | self.llm
            | StrOutputParser()
        )

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context from database"""
        try:
            docs = self.vector_db.similarity_search(query, k=3)
            return "\n".join(
                f"📑 {doc.metadata.get('source', 'Document')}:\n"
                f"{doc.page_content[:300].strip()}{'...' if len(doc.page_content) > 300 else ''}"
                for doc in docs
            )
        except Exception as e:
            logger.error(f"Context error: {str(e)}")
            return ""

    def _format_history(self) -> str:
        """Format conversation history"""
        return "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.history[-4:]
        )

    def chat(self):
        """Main chat loop"""
        try:
            while True:
                user_input = input("\n🧑💻 You: ").strip()

                if not user_input:
                    continue
                if user_input.lower() == "/exit":
                    print("\n👋 Goodbye!")
                    break

                # Handle capabilities question
                if any(keyword in user_input.lower() for keyword in ["help with", "what can you do", "capabilities"]):
                    response = f"I can help with:\n{self.analyzer.capabilities}"
                elif any(g in user_input.lower() for g in ["hi", "hello", "hey"]):
                    response = self.greeting_chain.invoke({
                        "topics": self.analyzer.domain_summary
                    })
                else:
                    context = self._get_context(user_input)
                    response = self.main_chain.invoke({
                        "context": context,
                        "history": self._format_history(),
                        "question": user_input
                    })

                # Update state
                self.history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                self.logger.log_interaction(user_input, response)
                print(f"\n🤖 {response}")

        except KeyboardInterrupt:
            print("\n🛑 Session interrupted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only mode')
    args = parser.parse_args()

    try:
        ChatBot(force_cpu=args.cpu).chat()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print("💥 System failed to initialize. Check:")
        print("1. API credentials in .env")
        print("2. Document database exists")
        print("3. Internet connection")
