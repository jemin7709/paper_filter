python ../src/fetch_cvf_papers.py --conference_url "https://openaccess.thecvf.com/CVPR2024?day=all" --csv_file "../data/cvpr2024.csv"
python ../src/fetch_cvf_papers.py --conference_url "https://openaccess.thecvf.com/WACV2024" --csv_file "../data/wacv2024.csv"
python ../src/fetch_eccv_papers.py
python ../src/fetch_aaai_papers.py
python ../src/fetch_openreview_papers.py --venue_id "ICLR.cc/2024/Conference" --output_file "../data/iclr2024.csv"
python ../src/fetch_openreview_papers.py --venue_id "ICML.cc/2024/Conference" --output_file "../data/icml2024.csv"