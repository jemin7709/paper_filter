#!/usr/bin/zsh

source ~/.zshrc
conda activate paper

python related_papers.py --file ./data/iclr2025.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/aaai2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/cvpr2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/eccv2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/iclr2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/icml2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
python related_papers.py --file ./data/wacv2024.csv --similarity_threshold 0.61 --individual_sim_threshold 0.7 --min_top_papers 1 --top_k_similar 3
