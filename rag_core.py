import os, json
import traceback
from typing import List, Literal, Dict, Optional
from dotenv import load_dotenv
from langdetect import detect
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# DOCUMENT PROCESSING

class DocumentProcessor:
    def __init__(self, folder: str = "folder", chunk_size=1000, chunk_overlap=80):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.folder = folder

    def load_documents(self) -> List[Document]:
        """
        Load all PDFs in the folder and return a list of LangChain Documents.
        Adds source and page metadata.
        """
        if not os.path.exists(self.folder):
            print(f"⚠️ Folder not found: {self.folder}")
            return []

        pdf_files = [f for f in os.listdir(self.folder) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"⚠️ No PDF files found in folder: {self.folder}")
            return []

        all_docs = []

        for pdf_file in pdf_files:
            full_path = os.path.join(self.folder, pdf_file)
            print(f"   Loading: {pdf_file} ... ", end="")

            try:
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                page_count = len(docs)
                print(f"Success! ({page_count} pages)")

                # Add source metadata for each page
                for i, doc in enumerate(docs):
                    doc.metadata["source"] = pdf_file
                    doc.metadata["page"] = i + 1

                all_docs.extend(docs)
            except Exception as e:
                print(f"Failed! Error: {str(e)}")

        total_pages = len(all_docs)
        print(f"\n✅ Total pages loaded: {total_pages} across {len(pdf_files)} documents")

        if total_pages == 0:
            print("⚠️ No content extracted. Make sure PDFs are text-based (not scanned images).")

        return all_docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        Preserves source metadata for each chunk.
        """
        if not docs:
            print("⚠️ No documents to chunk.")
            return []

        all_chunks = []
        total_pages = len(docs)

        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            chunks = self.splitter.split_documents([doc])
            chunk_count = len(chunks)
            print(f"   Document {i}/{total_pages}: {source} → {chunk_count} chunks created")
            all_chunks.extend(chunks)

        total_chunks = len(all_chunks)
        print(f"\n✅ Total chunks created: {total_chunks}")

        return all_chunks

# VECTOR STORE

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory

    def create_vectorstore(self, chunks: List[Document]) -> Optional[Chroma]:
        """
        Create a Chroma vector store from document chunks and save it.

        Args:
            chunks: List of document chunks to embed

        Returns:
            Chroma vector store instance
        """
        try:
            total_chunks = len(chunks)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            print("\n✅ SUCCESS! Vector database built and saved to", self.persist_directory)
            print(f"Ready with {total_chunks} searchable chunks from your Tax Reform Bills documents!")
            print("RAG system is now ready for accurate answers from the documents!\n")

            # 🔍 Run sanity check
            self._sanity_check()
            
            return self.vectorstore

        except Exception as e:
            print("\n❌ Critical error during vector store build:")
            print(f"   {str(e)}")
            traceback.print_exc()
            print("\nSuggested fixes:")
            print("   • Ensure PDFs were loaded correctly and are text-based")
            print("   • Check that .env has a valid OPENAI_API_KEY")
            print("   • Try with 1-2 small PDFs first")
            print("   • Make sure you have an internet connection (needed for embeddings)")
            print("   • Try running again — sometimes it’s a temporary connection issue")
            return None
    
    def load_vectorstore(self):
        """
        Load an existing vector store from disk.
        """
        try:
            
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("✅ Vector store loaded!")
            
            # 🔍 Run sanity check
            self._sanity_check()

            return self.vectorstore

        except Exception as e:
            print("❌ Failed to load vector store:", str(e))
            traceback.print_exc()
            return None

    def _sanity_check(self):
        """
        Sanity check to verify vector store retrieval works.
        """
        if not self.vectorstore:
            print("⚠️ Sanity check skipped: vector store not initialized.")
            return
        
        print("\n🔎 Running vector store sanity check...")

        test_query = "What is the VAT rate according to the tax reform bill?"

        results = self.vectorstore.similarity_search(test_query, k=3)

        if not results:
            print("⚠️ Sanity check FAILED: No documents retrieved.")
            return

        print(f"✅ Sanity check PASSED: Retrieved {len(results)} documents\n")

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")

            print(f"Result {i}:")
            print(f"   Source: {source}")
            print(f"   Page: {page}")
            print(f"   Preview: {doc.page_content[:200]}...\n")

# RETRIEVAL TOOL (RETURNS DOCUMENTS)

def build_retrieval_tool(vectorstore: Chroma):
    @tool
    def retrieve_documents(query: str) -> str:
        """
        Search for relevant documents in the knowledge base.
        
        Use this tool when you need information from the document collection
        to answer the user's question. Do NOT use this for:
        - General knowledge questions
        - Greetings or small talk
        - Simple calculations
        
        Args:
            query: The search query describing what information is needed
            
        Returns:
            Relevant document excerpts that can help answer the question
        """
        # Use MMR (Maximum Marginal Relevance) for diverse results
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        
        # Retrieve documents
        results = retriever.invoke(query)
        if not results:
            return "No relevant documents found."
        
        # Format results
        formatted = "\n\n---\n\n".join(
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(results)
        )
        return formatted
    return retrieve_documents

# MAIN AGENT

def build_agent(vectorstore: Chroma):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # ---------- Language ----------
    def detect_language(text: str) -> str:
        try:
            return detect(text)
        except:
            return "en"

    def translate(text: str, target_lang: str) -> str:
        return llm.invoke(
            f"Translate the following text to {target_lang}:\n{text}"
        ).content

    #  Retrieval Gate 
    def needs_retrieval(text: str) -> bool:
        trivial = [
            "hello", "hi", "thanks", "thank you",
            "bye", "goodbye", "who are you", "what can you do"
        ]
        if text.lower().strip() in trivial:
            return False

        keywords = [
            "tax", "bill", "act", "law", "reform",
            "vat", "cit", "penalty", "rate",
            "exemption", "section", "amendment"
        ]
        return any(k in text.lower() for k in keywords)

    # ---------- Query Rewrite ----------
    def rewrite_query(query: str) -> str:
        return llm.invoke(
            f"Rewrite this query for document retrieval:\n{query}"
        ).content

    # ---------- Faithfulness ----------
    def check_faithfulness(answer: str, docs: List[Document]) -> Dict:
        context = "\n\n".join(d.page_content for d in docs)
        prompt = ChatPromptTemplate.from_template("""
    Answer:
    {answer}

    Context:
    {context}

    Respond strictly in JSON:
    {{"faithful": true | false, "reason": "short explanation"}}
    """)
        
        resp = llm.invoke(
            prompt.format(answer=answer, context=context)
        )
        return json.loads(resp.content)

    # Confidence 
    def confidence_score(answer: str, docs: List[Document]) -> Dict:
        prompt = f"""
    Answer:
    {answer}

    Context:
    {" ".join(d.page_content for d in docs)}

    Respond in JSON:
    {{"score": 0.0-1.0, "reason": "why"}}
    """
        return json.loads(llm.invoke(prompt).content)

    # Citations 
    def extract_sources(docs: List[Document]) -> List[Dict]:
        sources = []
        for d in docs:
            sources.append({
                "source": os.path.basename(
                    d.metadata.get("source", "Unknown")
                ),
                "page": d.metadata.get("page", "N/A")
            })
        return sources

    # Retrieval Tools 
    retrieval_tool = build_retrieval_tool(vectorstore)
    tools = [retrieval_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
    You are a Nigerian Tax Reform Assistant.

    Rules:
    - Use document retrieval for factual questions
    - Cite sources
    - Do NOT hallucinate
    - If documents are insufficient, say so
    """)

   
    # GRAPH NODES

    def assistant(state: MessagesState):
        user_input = state["messages"][-1].content

        user_lang = detect_language(user_input)
        query = (
            translate(user_input, "English")
            if user_lang != "en"
            else user_input
        )

        state["user_lang"] = user_lang

        messages = [system_prompt, HumanMessage(content=query)]

        if needs_retrieval(query):
            refined = rewrite_query(query)
            messages.append(
                SystemMessage(
                    content=f"Use retrieval for:\n{refined}"
                )
            )

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "assistant")

    agent = builder.compile(checkpointer=MemorySaver())
   
    # AGENT QUERY FUNCTION
   
    def query_agent(user_input: str, thread_id: str = "default") -> Dict:
        user_lang = detect_language(user_input)
        query = translate(user_input, "English") if user_lang != "en" else user_input
        
        messages = [system_prompt,
                    HumanMessage(content=query)]

        if needs_retrieval(query):
            refined = rewrite_query(query)
            messages.append(SystemMessage(content=f"Use retrieval for:\n{refined}"))

        result = llm_with_tools.invoke(messages)
        
        # Assume result is AIMessage or ToolMessage
        final_answer = result.content if hasattr(result, "content") else None
        retrieved_docs = getattr(result, "content", [])

        if retrieved_docs:
            faith = check_faithfulness(final_answer, retrieved_docs)
            conf = confidence_score(final_answer, retrieved_docs)
            if not faith["faithful"]:
                final_answer = "I couldn't find sufficient info in the documents."
            sources = extract_sources(retrieved_docs)
        else:
            faith = {"faithful": True, "reason": "No retrieval needed"}
            conf = {"score": 1.0, "reason": "Direct answer"}
            sources = []

        final_answer = final_answer if user_lang == "en" else translate(final_answer, user_lang)

        return {
            "answer": final_answer,
            "language": user_lang,
            "faithful": faith["faithful"],
            "faithfulness_reason": faith["reason"],
            "confidence": conf["score"],
            "sources": sources
        }

    # Return everything for modular testing 
    return {
        "query_agent": query_agent,
        "detect_language": detect_language,
        "translate": translate,
        "needs_retrieval": needs_retrieval,
        "rewrite_query": rewrite_query,
        "check_faithfulness": check_faithfulness,
        "confidence_score": confidence_score,
        "extract_sources": extract_sources,
        "retrieval_tool": retrieval_tool
    }
        # result = agent.invoke(
        #     {"messages": [HumanMessage(content=user_input)]},
        #     config={"configurable": {"thread_id": thread_id}}
        # )

        
        # final_answer = None
        # retrieved_docs = []

        # for msg in result["messages"]:
        #     if isinstance(msg, AIMessage) and msg.content:
        #         final_answer = msg.content
        #     if isinstance(msg, ToolMessage):
        #         retrieved_docs.extend(msg.content)

    #     if retrieved_docs:
    #         faith = check_faithfulness(final_answer, retrieved_docs)
    #         conf = confidence_score(final_answer, retrieved_docs)

    #         if not faith["faithful"]:
    #             final_answer = (
    #                 "I couldn't find sufficient information "
    #                 "in the documents to confidently answer this."
    #             )

    #         sources = extract_sources(retrieved_docs)
    #     else:
    #         faith = {"faithful": True, "reason": "No retrieval needed"}
    #         conf = {"score": 1.0, "reason": "Direct answer"}
    #         sources = []

    #     final_answer = (
    #         final_answer if user_lang == "en"
    #         else translate(final_answer, user_lang)
    #     )

    #     return {
    #         "answer": final_answer,
    #         "language": user_lang,
    #         "faithful": faith["faithful"],
    #         "faithfulness_reason": faith["reason"],
    #         "confidence": conf["score"],
    #         "sources": sources
    #     }

    # return query_agent



# # SYSTEM BUILDER

# def build_rag_system(doc_path: str):
#     """
#     Build a complete RAG system from documents.
#     """
#     print(f"\n{'='*60}")
#     print("BUILDING RAG SYSTEM")
#     print(f"{'='*60}\n")

#     processor = DocumentProcessor()
#     docs = processor.load_documents(doc_path)
#     chunks = processor.chunk_documents(docs)

#     store = VectorStoreManager().create(chunks)
#     agent = build_agent(store)

#     return agent
