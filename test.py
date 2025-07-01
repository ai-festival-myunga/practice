import requests
import json
import yfinance as yf
import pandas as pd
import re
from datetime import datetime
import os
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Tool: yfinance 주가 조회 함수
# --------------------------------------------------------------------------
def get_stock_price(ticker, date):
    """
    지정된 티커와 날짜의 주식 종가를 가져옵니다.
    :param ticker: 주식 티커 (예: '005930.KS' for 삼성전자)
    :param date: 날짜 (YYYY-MM-DD 형식)
    :return: 해당 날짜의 종가 또는 None
    """
    try:
        stock = yf.Ticker(ticker)
        # 날짜가 유효한지 확인하기 위해 해당 날짜의 데이터를 직접 요청합니다.
        hist = stock.history(start=date, end=(pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        
        if not hist.empty:
            return hist['Close'].iloc[0]
        else:
            # 해당 날짜에 데이터가 없으면(주말/공휴일), 이전 가장 가까운 거래일의 데이터를 찾습니다.
            hist_before = stock.history(end=date, period="5d")
            if not hist_before.empty:
                latest_price = hist_before['Close'].iloc[-1]
                latest_date = hist_before.index[-1].strftime('%Y-%m-%d')
                print(f"Info: {date}은 거래일이 아닙니다. 가장 가까운 거래일인 {latest_date}의 데이터를 사용합니다.")
                return latest_price
            return None
    except Exception as e:
        print(f"yfinance 오류 발생: {e}")
        return None

# --------------------------------------------------------------------------
# Agent Core Logic
# --------------------------------------------------------------------------
def call_clova_api(prompt, system_message="당신은 금융 정보를 쉽고 친절하게 설명해주는 AI 비서입니다."):
    """Naver ClovaX LLM API를 호출하는 함수"""
    # 실제 키는 외부에 노출되지 않도록 환경 변수 등으로 관리하는 것이 안전합니다.
    # CLOVA_API_KEY = "여기에 api"
    api_url = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
    
    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "maxTokens": 1000,
        "temperature": 0.1, # 일관된 답변을 위해 온도를 낮춤
        "topP": 0.8,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result.get('result', {}).get('message', {}).get('content', 'No content found')
    except requests.exceptions.RequestException as e:
        return f"API 요청 중 오류 발생: {e}"
    except json.JSONDecodeError:
        return f"API 응답을 파싱하는 중 오류 발생: {response.text}"

def run_stock_agent(user_query: str):
    """
    사용자 질문을 처리하고, 필요시 yfinance 도구를 사용한 후, LLM을 호출하여 최종 답변을 생성합니다.
    """
    # 1. LLM을 사용하여 사용자 질문에서 회사명, 티커, 날짜 추출
    extraction_prompt = f"""
    사용자 질문에서 주식 종가 조회에 필요한 정보를 추출해줘.
    - 회사명 (예: 삼성전자)
    그런데 사람들은 회사명을 종종 줄여 말하곤 해 니가 그건 감안해줘.
    예를 들어 삼성전자 -> 삼전, 현대자동차 -> 현차 or 현대
    - 주식 티커 (예: 005930.KS)
    - 날짜 (YYYY-MM-DD 형식)

    
    만약 정보가 없다면 "없음"이라고 출력해줘.
    질문: "{user_query}"

    추출 결과 (JSON 형식):
    """
    system_message_extraction = "당신은 주어진 텍스트에서 특정 정보를 정확하게 추출하는 AI입니다. 한국 주식 티커는 회사명 뒤에 '.KS'를 붙여야 합니다(예: 삼성전자 -> 005930.KS). 없는 정보는 '없음'으로 표시하세요."
    
    print("1단계: LLM을 통해 정보 추출 중...")
    extracted_info_str = call_clova_api(extraction_prompt, system_message_extraction)
    print(f"추출된 정보 (문자열): {extracted_info_str}")

    try:
        # LLM의 답변(문자열)에서 JSON 부분만 추출
        json_match = re.search(r'\{.*\}', extracted_info_str, re.DOTALL)
        if json_match:
            extracted_info = json.loads(json_match.group())
            company_name = extracted_info.get("회사명")
            ticker = extracted_info.get("주식 티커")
            date_to_check = extracted_info.get("날짜")
        else:
            raise json.JSONDecodeError("No JSON object found", extracted_info_str, 0)

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"정보 추출 실패: {e}. 일반적인 답변으로 전환합니다.")
        # 정보 추출에 실패하면, LLM이 직접 답변하도록 함
        return call_clova_api(user_query)

    # 2. 도구(yfinance) 사용 결정 및 실행
    if ticker and ticker != "없음" and date_to_check and date_to_check != "없음":
        print(f"2단계: 도구 사용 - get_stock_price(ticker='{ticker}', date='{date_to_check}')")
        price = get_stock_price(ticker, date_to_check)
        
        if price is not None:
            # 3. 최종 답변 생성을 위해 LLM 재호출
            fact = f"{company_name}의 {date_to_check} 기준 종가는 {price:,.0f}원입니다."
            final_prompt = f"다음 정보를 바탕으로 사용자의 질문에 대해 친절하고 상세하게 답변해 주세요.\n\n[정보]: {fact}\n\n[사용자 질문]: {user_query}"
            print("3단계: 최종 답변 생성 중...")
            return call_clova_api(final_prompt)
        else:
            # 주가 조회 실패 시
            fail_prompt = f"'{company_name}'({ticker})의 {date_to_check} 주가 정보를 조회하는 데 실패했습니다. 아마도 해당 날짜에 거래가 없었거나 티커가 잘못되었을 수 있습니다. 이 상황을 사용자에게 친절하게 설명해주세요."
            print("3단계: 주가 조회 실패에 대한 답변 생성 중...")
            return call_clova_api(fail_prompt)
    else:
        # 필요한 정보가 부족하여 도구를 사용할 수 없는 경우
        print("2단계: 도구 미사용 - 일반적인 질문으로 처리합니다.")
        return call_clova_api(user_query)

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    CLOVA_API_KEY = os.getenv('CLOVA_API_KEY')
    # --- 테스트 케이스 ---
    # 1. 주가 조회 (성공 케이스)
    # user_question = "현대차의 2025년 6월 30일 종가는 얼마인가요?"
    
    # 2. 주가 조회 (날짜가 주말인 케이스)
    # user_question = "SK하이닉스의 2024년 5월 11일 주가가 궁금해."

    # 3. 일반적인 질문
    # user_question = "오늘 날씨 어때그리고 현대차의 최신 뉴스는 뭐야?"
    
    # 4. 사용자가 직접 질문 입력
    user_question = input("무엇이 궁금하신가요? ")

    print(f"사용자 질문: {user_question}")
    print("-" * 20)
    
    # 에이전트 실행
    final_answer = run_stock_agent(user_query=user_question)
    
    print("-" * 20)
    print(f"AI 최종 답변:\n{final_answer}")
