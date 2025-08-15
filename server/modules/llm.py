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
If the context does not contain the answer, clearly state that the information is not available instead of guessing.

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer Instructions**:
1. Use **only the context** to answer. Do not include any outside knowledge.  
2. Be **accurate, factual, and concise**.  
3. Use **simple explanations** if the answer may be complex.  
4. If the context does not contain the answer, respond exactly:  
   "I'm sorry, but I couldn't find relevant information in the provided documents."  
5. Do **not** make up facts or assume anything beyond the context.  
6. Do **not** provide medical advice, diagnoses, or treatment suggestions.  
7. Keep your tone **calm, professional, and respectful**.  

"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
