from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.

Your task is to answer questions **only using the information in the provided context**.  
You must not use any external knowledge. If the context does not contain the answer, respond exactly:  
"I'm sorry, but I couldn't find relevant information in the provided documents."

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer Instructions**:
1. Use **only the context** to answer. Do not guess or provide general medical knowledge.  
2. Quote or paraphrase text from the context directly.  
3. If the context does not contain the answer, respond exactly:  
   "I'm sorry, but I couldn't find relevant information in the provided documents."  
4. Do **not** provide medical advice, diagnoses, or recommendations.  
5. Keep your tone **calm, factual, and respectful**.  
6. Provide a concise answer; do not include information not in the context.

---

**Example**:  
**Context**: "In most cases, surgery will not be needed for an ischemic stroke. If serious brain swelling occurs, a decompressive craniectomy may be considered."  
**Question**: "Will surgery be needed for ischemic stroke?"  
**Answer**: "In most cases, surgery will not be needed. If serious brain swelling occurs, a decompressive craniectomy may be considered."
 

"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
