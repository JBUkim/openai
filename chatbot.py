import os
import logging
import sys
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

# 환경 변수 설정
api_key = os.getenv('OPENAI_API_KEY')

# Flask 애플리케이션 초기화
app = Flask(__name__)
app.secret_key = 'secret_key'
CORS(app)

# 로깅 설정
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)

# 문서 로드(data 폴더에 문서를 넣어 두세요)
documents = SimpleDirectoryReader("data").load_data() # /openai/llama_index/data/chatbot.txt 저장

from llama_index.llms.openai import OpenAI

llm_predictor = OpenAI( 
    model="gpt-3.5-turbo",
    temperature=0)

# ServiceContext 준비
service_context = ServiceContext.from_defaults(
    llm_predictor = llm_predictor
)

# 인덱스 생성
index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context
)

# 인덱스 저장, 기본 폴더: storage
index.storage_context.persist()

# # 인덱스 로드
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context)
# query_engine = index.as_query_engine()

# # 라우트: 스크롤 테스트 페이지
# @app.get("/chatbot_scroll")
# def chatbot_scroll():
#     return render_template("chatbot_scroll.html")

# # 라우트: 챗봇 입력 폼
# @app.get("/chatbot")
# def chatbot_form():
#     return render_template("chatbot.html")

# # 라우트: 챗봇 처리
# @app.post("/chatbot")
# def chatbot_proc():
#     data = request.json
#     question = data['question']
#     logging.info(f"Received question: {question}")

#     response = query_engine.query(question)
#     return jsonify({"res": response.response})

# # Flask 서버 시작
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)



