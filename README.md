# m6-l6a-hashemalqurashi805-svg
# Cross-Lingual Semantic Search with BERT (Lab 6A Stretch Task)

This repository demonstrates the ability of **Multilingual BERT** (`bert-base-multilingual-cased`) to align English and Arabic climate-related texts into a shared semantic space.

## 🚀 Project Overview
In this stretch task, we explored how a single transformer model can understand the contextual meaning of sentences regardless of the language. We specifically compared 10 English articles and 10 Arabic articles related to climate change.

## 🛠️ Tech Stack
- **Model:** `bert-base-multilingual-cased` (Hugging Face)
- **Environment:** WSL2 (Ubuntu) / VS Code / Git Bash
- **Libraries:** `transformers`, `torch`, `pandas`, `seaborn`, `scikit-learn`

## 📊 Results: Cosine Similarity Heatmap
The project generates a similarity matrix visualized as a heatmap. 
- High similarity scores (lighter colors) were observed between English sentences and their corresponding Arabic translations.
- This confirms that BERT maps "Climate Change" and "تغير المناخ" to very similar vector embeddings.

## 📁 Repository Structure
- `stretch_cross_lingual.py`: Main script for embedding extraction and visualization.
- `stretch_analysis.md`: Detailed two-paragraph analysis of the results.
- `data/climate_articles.csv`: The dataset containing mixed-language climate texts.
- `similarity_heatmap.png`: The final output visualization.

## 🔧 How to Run
1. Activate the virtual environment:
   ```bash
   source venv/Scripts/activate