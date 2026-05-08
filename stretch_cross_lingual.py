import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 1. إعداد النموذج
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. تحميل البيانات
df = pd.read_csv('data/climate_articles.csv')

# اختيار 10 نصوص إنجليزية و 10 عربية (يفضل أن تكون متقابلة أو تتحدث عن نفس الموضوع)
en_articles = df[df['language'] == 'en'].iloc[:10]
ar_articles = df[df['language'] == 'ar'].iloc[:10]

texts = en_articles['text'].tolist() + ar_articles['text'].tolist()
labels = [f"EN_{i}" for i in range(10)] + [f"AR_{i}" for i in range(10)]

# 3. دالة لاستخراج التضمينات
def get_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # استخدام [CLS] token كممثل للنص
    return outputs.last_hidden_state[:, 0, :].numpy()

print("جاري استخراج التضمينات...")
embeddings = get_embeddings(texts)

# 4. حساب مصفوفة التشابه (Cosine Similarity)
similarity_matrix = cosine_similarity(embeddings)

# 5. رسم خريطة الحرارة (Heatmap)
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, annot=False, cmap='YlGnBu')
plt.title("Cross-lingual Similarity: English vs Arabic")
plt.savefig('similarity_heatmap.png')
plt.show()