#!/bin/bash

# 모든 스크립트를 실행하는 Bash 파일

# Python 환경 확인
if ! command -v python &> /dev/null
then
    echo "Python이 설치되어 있지 않습니다. 설치 후 다시 시도해주세요."
    exit 1
fi

# 스크립트 실행
echo "[1/4] get_aaai_papers.py 실행 중..."
python scripts/get_aaai_papers.py || { echo "get_aaai_papers.py 실행 실패"; exit 1; }

echo "[2/4] get_cvf_papers.py 실행 중..."
python scripts/get_cvf_papers.py || { echo "get_cvf_papers.py 실행 실패"; exit 1; }

echo "[3/4] get_eccv_papers.py 실행 중..."
python scripts/get_eccv_papers.py || { echo "get_eccv_papers.py 실행 실패"; exit 1; }

echo "[4/4] get_openreview_papers.py 실행 중..."
python scripts/get_openreview_papers.py || { echo "get_openreview_papers.py 실행 실패"; exit 1; }

# 완료 메시지
echo "모든 스크립트가 성공적으로 실행되었습니다."