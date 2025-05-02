# Paper Filter

## 프로젝트 소개
Paper Collector는 학술 논문 데이터를 수집하고 정리하는 도구입니다. 다양한 학술 컨퍼런스의 논문 데이터를 자동으로 수집하여 CSV 파일로 저장하고, 미리 정의한 논문들과의 유사도를 기반으로 관련 논문을 필터링합니다.

## 설치 및 실행 방법
1. 프로젝트를 클론:
   ```bash
   git clone <repository-url>
   ```
2. 필요한 라이브러리를 설치합니다:
   ```bash
   uv sync or pip install -r requirements.txt
   ```
3. 스크립트를 실행하여 데이터를 수집합니다:
   ```bash
   sh scripts/download_resources.sh
   ```

## 파일 구조

```
.
├── pyproject.toml
├── README.md
├── data/ # 논문 목록 저장
├── network/
│   └── vpngate.py # 스크래핑 시 VPN 연결
├── scripts/
│   ├── download_resources.sh 
│   ├── execute_all_scripts.sh
│   └── process_related_papers.sh
└── src/
    ├── fetch_aaai_papers.py
    ├── fetch_cvf_papers.py
    ├── fetch_eccv_papers.py
    ├── fetch_openreview_papers.py
    └── process_related_papers.py
```