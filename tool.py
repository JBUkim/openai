import os
import time
import json

# import openai  # 0.28.0
from openai import OpenAI

# openai.api_key='키를 직접 지정하는 경우(권장 아님)' 
# os.environ['OPENAI_API_KEY'] = '키를 직접 지정하는 경우(권장 아님)'

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

# OpenAI API 사용함수
# role: GPT 역활  예) 너는 문화 해설가야
# prompt: 질문 메시지
# format='': 출력 세부 형식, 파라미터 전달이 안되면 아무 값도 사용하지 않는다는 선언
# llm='gpt-3.5-turbo': 사용할 인공지능 모델
# output='json': 출력 형식
def answer(role, prompt, format='json', llm='gpt-3.5-turbo', output='json'):
    
    if output.lower() == 'json':
      # gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-4-turbo
      response = client.chat.completions.create(
          model=llm,
          messages=[
              {
                  'role': 'system',
                  'content': role
              },
              {
                  'role': 'user',
                  'content': prompt + '\n\n출력 형식(json): ' + format
              }
          ],
          n=1,             # 응답수, 다양한 응답 생성 가능
          max_tokens=1000, # 응답 생성시 최대 1000개의 단어 사용
          temperature=0,   # 창의적인 응답여부, 값이 클수록 확률에 기반한 창의적인 응답이 생성됨
          response_format= { "type":"json_object" }
      )
    else:
      response = client.chat.completions.create(
          model=llm,
          messages=[
              {
                  'role': 'system',
                  'content': role
              },
              {
                  'role': 'user',
                  'content': prompt
              }
          ],
          n=1,             # 응답수, 다양한 응답 생성 가능
          max_tokens=1000, # 응답 생성시 최대 1000개의 단어 사용
          temperature=0    # 창의적인 응답여부, 값이 클수록 확률에 기반한 창의적인 응답이 생성됨
      )
   
    return json.loads(response.choices[0].message.content) # str -> json
  
# 문자열을 줄바꿈 기준으로 분리하여, 빈 라인을 제거하고, 문장들로 이루어진 리스트 생성
def remove_empty_lines(text):
    lines = [line for line in text.splitlines() if line.strip()]
    # print('-> lines:', lines)
    # print('-' * 80)
    # 문장들을 다시 합쳐서 하나의 문자열로 반환
    result = '\n'.join(lines)
    return result
  
# DBMS connection 생성
# def connection():
#     conn = cx_Oracle.connect('gpt/69017000@3.34.236.207:1521/XE')
#     cursor = conn.cursor() # SQL 실행 객체 생성
   
#     return conn, cursor
