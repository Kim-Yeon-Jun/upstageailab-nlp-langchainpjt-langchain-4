from typing import Dict, List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

def create_category_vectorstores(category_docs: Dict[str, List[Document]]) -> Dict[str, FAISS]:
    """카테고리별 문서를 기반으로 FAISS 벡터스토어를 생성합니다."""
    vectorstores = {}
    embeddings = UpstageEmbeddings(model="embedding-query")

    for category, docs in category_docs.items():
        if not docs:
            continue
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstores[category] = vectorstore
        print(f"[INFO] {category}: {len(docs)}개 문서로 벡터스토어 생성")

    return vectorstores