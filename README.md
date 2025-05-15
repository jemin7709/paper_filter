# Paper Filter

## 프로젝트 소개
Paper Collector는 논문 제목과 초록을 수집하고 필터링하는 툴입니다. 다양한 학회의 논문 데이터를 자동으로 수집하여 CSV 파일로 저장하고, 미리 정의한 논문들과의 유사도를 기반으로 관련 논문을 필터링해 사용자의 불필요한 노동을 줄여줍니다.

## 설치 및 실행 방법
1. 프로젝트를 클론:
   ```bash
   git clone <repository-url>
   ```
2. 필요한 라이브러리를 설치합니다:
   ```bash
   uv sync or pip install -r requirements.txt (uv 사용 권장)
   ```
3. download_papers 스크립트를 실행하여 논문 데이터를 스크래핑합니다:
   ```bash
   sh scripts/download_papers.sh
   ```
4. process_related_papers 스크립트를 실행해 논문 유사도를 계산합니다:
   ```bash
   sh scripts/process_related_papers.sh
   ```

## 파일 구조

```
.
├── pyproject.toml
├── README.md
├── data/ # 논문 목록 저장
├── network/
│   └── vpngate.py # [선택사항] 스크래핑 시 VPN 연결 (vpn대신 tor 사용 권장, tor 사용 시 proxy 옵션 사용)
├── scripts/
│   ├── download_papers.sh 
│   └── process_related_papers.sh
└── src/
    ├── fetch_aaai_papers.py
    ├── fetch_cvf_papers.py
    ├── fetch_eccv_papers.py
    ├── fetch_openreview_papers.py
    └── process_related_papers.py
```
