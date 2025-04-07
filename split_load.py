import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, Language


# 카테고리별 분할기 설정
def get_splitter_for_category(category: str):
    """카테고리에 따라 적절한 텍스트 분할기를 반환합니다."""
    # 카테고리명에 기반한 분할기 선택
    if category in ['markdown', '마크다운', 'docs', 'documentation']:
        return MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    elif category.lower() in ['파이썬', '파이썬_다익스트라_코드']:
        return RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=1000, chunk_overlap=100)
    else:
        # 기본 분할기
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
# 여러 출처의 데이터를 불러와 처리하는 함수
def load_all_data(data_dir: str ="data"):
    """
    모든 카테고리의 데이터를 불러와 처리합니다.
    
    Args:
        data_dir: 데이터가 저장된 디렉토리
        
    Returns:
        카테고리별 분할된 문서와 통합 문서
    """
    if not os.path.exists(data_dir):
        print(f"오류: {data_dir} 디렉토리가 존재하지 않습니다.")
        return {}, []
    
    all_docs = []
    category_docs = {}
    
    # 모든 카테고리 파일 처리
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            category = filename.split('.')[0]
            file_path = os.path.join(data_dir, filename)
            
            with open(file_path, "rb") as f:
                docs = pickle.load(f)
            
            # 카테고리별 splitter 설정
            text_splitter = get_splitter_for_category(category)
            splits = text_splitter.split_documents(docs)
            
            category_docs[category] = splits
            all_docs.extend(splits)
            
            print(f"{category} 카테고리에서 {len(splits)}개 청크 로드")
    
    return category_docs, all_docs
