import pandas as pd
import jieba
import re
import tomotopy as tp
import pyLDAvis
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import pyLDAvis.gensim_models as gensimvis
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)
# 数据导入
data = pd.read_csv(r"E:\pythonProject\python练习\data\merged_file.csv")
# 数据预处理和分词
def chinese_word_cut(mytext, stopwords):
    new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
    new_data = " ".join(new_data)
    seg_list_exact = jieba.cut(new_data)
    result_list = [word for word in seg_list_exact if word not in stopwords and len(word) > 1]
    return " ".join(result_list)
# 加载停用词列表
stopwords = load_stopwords(r'E:\\pythonProject\\python练习\\data\\cn_stopwords.txt')
# 应用分词函数
data['content_cutted'] = data['comments'].apply(lambda x: chinese_word_cut(x, stopwords))
# 过滤掉空评论
data = data[data['content_cutted'].str.len() > 0]
docs = [doc for doc in data['content_cutted']]
# 手肘法确定主题数K
def find_k(docs, min_k=1, max_k=20, min_df=3):
    perplexity_scores = []
    coherence_scores = []
    for k in range(min_k, max_k + 1):
        mdl = tp.LDAModel(min_df=min_df, k=k, seed=555)
        for doc in docs:
            mdl.add_doc(doc)  
        mdl.burn_in = 100
        mdl.train(400, workers=1)
        
        # 计算困惑度
        perplexity = mdl.perplexity  
        perplexity_scores.append(perplexity)
        
        # 计算一致性得分
        coh = tp.coherence.Coherence(mdl)
        coherence_scores.append(coh.get_score())
    
    # 绘制困惑度曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(min_k, max_k + 1), perplexity_scores, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Scores for Different Number of Topics")
    
    # 绘制一致性得分曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(min_k, max_k + 1), coherence_scores, marker='o', color='orange')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores for Different Number of Topics")
    plt.tight_layout()
    plt.show()
    
    return perplexity_scores, coherence_scores

# 计算一致性得分和困惑度
perplexity_scores, coherence_scores = find_k(docs, min_k=1, max_k=10, min_df=30)
k = 4
mdl = tp.LDAModel(k=k, min_df=30, seed=123)
for words in data['content_cutted']:
    if words:
        mdl.add_doc(words.split())
# 训练模型
mdl.train(2000,workers=1)  
# 查看每个主题的特征词
for k in range(mdl.k):
    print(f'Top 5 words of topic #{k}')
    print(mdl.get_topic_words(k, top_n=5))
    print('\n')
# 文档的主题分布
docs = [doc for doc in mdl.docs]
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists=np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)]), 
    doc_topic_dists=doc_topic_dists, 
    doc_lengths=doc_lengths, 
    vocab=vocab, 
    term_frequency=term_frequency,
    start_index=0,  
    sort_topics=False 
)
pyLDAvis.save_html(prepared_data, 'ldavis.html')
pyLDAvis.display(prepared_data)
#--proxy http://127.0.0.1:7890 
