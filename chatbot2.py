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

# 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# 라우트: 챗봇 입력 폼
@app.get("/chatbot")
def chatbot_form():
    return render_template("chatbot.html")

# 라우트: 챗봇 처리
@app.post("/chatbot")
def chatbot_proc():
    data = request.json
    question = data['question']
    logging.info(f"Received question: {question}")

    response = query_engine.query(question)
    return jsonify({"res": response.response})

# Flask 서버 시작
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)



