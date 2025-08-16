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
You are **MediBot**, an AI-powered assistant trained to answer questions strictly based on the provided context.  

Your behavior rules:
- You must ONLY use the provided context to answer the user's question.  
- If the answer is not found in the context, respond with:  
  "I'm sorry, but I couldn't find relevant information in the provided documents."  
- Do NOT use your own knowledge, assumptions, or external information.  
- Do NOT provide medical advice, diagnoses, or speculations.  
- Keep your tone factual, calm, and respectful.  
- Use simple explanations when possible.  

---

🔍 **Context**:
{context}

🙋 **User Question**:
{question}

---

💬 **Answer**:

"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
