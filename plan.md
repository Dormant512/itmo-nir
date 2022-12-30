# Literature review

## Topic analysis 

The main goal of this part of literature review is the choice of topic analysis methods to be used in this research work for processing news headlines. Thus, the most acknowledged methods ranging from classic LDA to modern BERTopic are looked upon in this section.

Topic analysis is one of the tasks of natural language processing (NLP). It consists of embedding human-legible texts into machine-readable embeddigs, dimension reduction and further clustering of the embeddings. Such clusters can be interpreted as topics found in the texts, hence the name "topic analysis". One of the classical methodologies of topic analysis is the "bag-of-words", an approach for viewing documents as unstructured vocabularies of words with different frequencies. It was the foundation for a plethora of commonly used topic analysis tools such as LDA, NMF, Top2Vec and CTM.

Authors of the publication [onan2019] extracted topics from scientific literature via custom two-stage pipelines. The first step in every experiment was to embed the text into machine-interpretable numeric vectors. Then in the second step different algorithms were used to cluster the vectors, validate the extracted clusters and output an array of topics.

For embedding scientific texts several approaches including word2vec, POS2vec, word-position2vec and LDA2vec were used individually as well as combined into the improved word embedding based scheme. The latter model takes scientific abstract sentences as input, which is processed by word2vec, a simple embedder that transforms all words within the sentence into vectors without taking the word position into account. Then the authors computed averages of word vectors within each sentence so that the resulting vectors could be used as sentence representations. Before averaging, stop-words have been removed due to this tactic yielding better results. After that, POS2vec (part-of-speech2vec) was used on each word in the sentence to generate POS tags. Then, word-position2vec was employed to generate position vectors for the words and the whole sentence. The latter was concatenated to the vectors obtained earlier. Lastly, the sentences were further processed by LDA2vec. Its output vector was concatenated to yield the final ensemble embedding.

In the publication K-means, K-modes, K-means++, SOM and DIANA were chosen as the clustering algorithms. Classical K-means is a simple partition-based clustering method, while K-modes is its extension better suited for categorical data. As K-means' performance highly depends on the initial random seed for the coordinates of cluster centers, K-means++ uses a heuristic function to determine the optimal initial cluster centers. Self-organizing maps (SOM) method is based on clustering and dimension reduction. Divisive analysis clustering (DIANA) is a divisive hierarchical clustering method. Finally, the authors proposed iterative voting consensus (IVC) - a consensus function which seeks cohesion between different clustering methods on the same data and thus lets those algorithms compensate for each others' weaknesses.

The authors have compared 43 different configurations of topic analysis pipeline on the same dataset of scientific abstracts. The models were compared by values of the following metrics: Jaccard Coefficient (JC), FM and F1.
$$
JC = \frac{a}{a+b+c} \\
FM = \left( \frac{a}{a+b} \cdot \frac{a}{a+c} \right)^{0.5} \\
F1 = \frac{2a^2}{2a^2 + ac + ab}
$$
Here $a$ denotes the number of pairs of instances which are grouped in the same cluster and fall in the same category in the initial dataset. The value $b$ stands for pairs from the same category which were not grouped in the same cluster. Lastly, $c$ denotes pairs from different categories which were grouped together. The higher the metrics described above, the better results models produce.

According to the results presented in the article and in figure [FIGURE XXX] ([onan2019]), among single method clustering strategies, LDA (latent Dirichlet allocation) outperformed other algorithms, with DIANA taking the second place. Results of clustering seemed to have improved significantly when integrating LDA2vec embedding scheme with clustering algorithms.

![onan2019](file:///home/dormant/itmo-nir/sci-res/visuals/src/onan2019.png)

*Figure XXX: Jaccard Coefficient metric for different combinations of embedding and clustering methods.* 

Even better results were shown by the proposed clustering ensemble framework which utilizes IVC for combining the mentioned methods. Thus, such proposed ensemble approach can lead to better topic analysis models.

As opposed to the example presented above, topic analysis can solve not only the problem of extracting broad semantic themes from texts but also some more nuanced information. Authors of [huang2020] propose their two-step approach for aspect-based sentiment analysis. Their procedure lets evaluate reviews in terms of how positive or negative they are and also define aspects of the review object which are evaluated. In the publication, restaurant and laptop reviews were used as input data. The proposed model outputs a tuple in form of (sentiment, aspect). For example, a review of a restaurant could output (good, ambiance) or (bad, food).

For this finesse task a two-step procedure called JASen was developed. It consists of sentence embedding and classification via convolutional neural networks (CNNs). The former takes labeled data for embedding training and unlabeled data for further classification.

The results shown by the JASen model were compared with those of previously developed approaches. Values of aspect identification and sentiment polarity classification are presented in the [TABLES XXX AND YYY] below ([huang2020]).

*Table XXX: Quantitative evaluation on aspect identification (%).*

[TABLE]

*Table YYY: Quantitative evaluation on sentiment polarity classification (%).*

[TABLE]

In conclusion, the authors have proposed a model fine-tuned for joint aspect-sentiment extraction which outperformed earlier methods for this task.

However, more often (including this research work) the extraction of topics is enough. In contrast to the ensemble approach proposed in the publication [onan2019], authors of the article [grootendorst2022] leverage different modified algorithms that compose BERTopic - a modern and efficient topic analysis tool.

Its defining feature is the ability to extract topics dynamically and to build topic popularity time series. This is allowed by using c-TF-IDF (class term frequency inverse document frequency) representations of topics:
$$
\mathrm{c\_TF\_IDF} = \underbrace{\frac{n_t}{\sum\limits_{k \in \mathrm{class}} n_k}}_{\mathrm{c\_TF}} \cdot \underbrace{\log \left[ \frac{|D|}{|\{ d_i \in D | t \in d_i \}|} \right]}_{\mathrm{IDF}}
$$
Here c-TF is defined by the number of instances of word $t$ divided by the number of words in the class, $|D|$ - number of documents in the dataset, $|\{ d_i \in D | t \in d_i \}|$ - number of documents in dataset, containing the word $t$. First, BERTopic is fitted on the entire dataset to create a global view of topics. Using c-TF-IDF is efficient, as IDF is computed globally and only c-TF needs to be computed at each timestep.

In the publication, newly proposed BERTopic is compared to its predecessors: LDA, NMF, top2vec and CTM by means of topic coherence (TC) and topic diversity (TD) metrics on three datasets of news and twitter posts. According to the article, BERTopic has high coherence scores across all datasets. In terms of topic diversity, it is outperformed by CTM.

One of the major strengths of BERTopic is that it has the capability to operate in the multilingual mode. The default embedding model "all-MiniLM-L6-v2" shows both moderately high topic coherence and diversity across all supported languages in addition to it being lightweight.

More detailed comparison of BERTopic with other methods of topic analysis is presented in the publication [egger2022]. The authors have chosen latent Dirichlet allocation (LDA), non-negative matrix factorization (NMF), top2vec and BERTopic in the task of topic analysis on the dataset of Twitter posts.

LDA is a generative probabilistic model for discrete data, which could be viewed as three-level hierarchical Bayesian clustering algorithm. Each document is represented as a mixture of topics with corresponding probabilities and each topic is a mixture over the collection of topic probabilities.

NMF is a decompositional method which works of TF-IDF transformed data by breaking down the input term-document matrix ($A$) into a pair of lower-ranking matrices:

- terms-topics ($W$) matrix containing basis vectors;
- topics-documents ($H$) matrix containing weights.

In NMF all elements in those matrices are non-negative so as to be interpretable.

Top2vec algorithm uses word embeddings via pretrained embedding models so that semantically close words have spatially close embedding vectors. Due to the sparsity of the vector space, a dimension reduction is performed before clustering. Commonly, hierarchical density-based spatial clustering of applications with noise (HDBSCAN) is used to identify dense regions in the reduced vector space of documents. As words that appear in several documents cannot be assigned to one document, they are recognized as noise. Thus, no lemmatization is needed beforehand.

BERTopic uses BERT pretrained embedders alongside with sentence-transformers for turning documents into vectors across more than 50 languages. Similarly to top2vec, BERTopic uses uniform manifold approximation and projection (UMAP) for dimension reduction and HDBSCAN for clustering. The principal difference between BERTopic and top2vec is that the latter utilizes c-TF-IDF metric instead of normal TF-IDF. Thus, importance of words in clusters and not in documents is taken into account. The usage of HDBSCAN eliminates the need for lemmatization.

Human interpretation of the extracted topics has shown that BERTopic and NMF perform the best among four tested methods. Despite that top2vec and BERTopic both use pretrained embedders for the representation of documents in vector space, many topics extracted by top2vec contained several semantic concepts or intersected each other. LDA gave the least legible results of all methods.

The authors mentioned several drawbacks of BERTopic models such as them yielding large numbers of clusters which leads to the need in manual topic processing. Another mentioned disadvantage is that one document could potentially be assigned to just one topic, which often does not reflect reality. In spite of this, BERTopic still can be characterized as one of the most successful and universally applicable methods of topic analysis due to high model quality and efficiency. Thus, BERTopic was the topic analysis method of choice in the current research work.

## Time series prediction

Прогнозирование временных рядов является одной из важнейших задач, относящихся к временным рядам. Существует множество методов прогнозирования, и наиболее значимые были перечислены в обзорной статье [30]. В ней названы следующие параметрические модели: 

- Модели скользящего среднего MA (moving average), представляющие прогнозируемые значения в виде среднего по некому интервалу до прогноза. 
- Модели простого экспоненциального сглаживания SES (simple exponential smoothing) – модели, рассматривающие прогнозируемое значение как линейную комбинацию всех значений временного ряда с коэффициентами, экспоненциально растущими от удалённых членов ряда к наиболее недавним. 
- Модели экспоненциального сглаживания Гольта и Гольта-Винтерса (Holt, Holt-Winters exponential smoothing) – расширение модели SES, учитывающее наличие тренда и высокочастотного шума во временном ряду, соответственно. 
- ARIMA/SARIMA модели (autoregressive integrated moving average) – авторегрессионные модели, которые могут учитывать или не учитывать сезонные циклические процессы. 

Среди непараметрических методов были названы искусственные нейронные сети, методы опорных векторов и k ближайших соседей. Стоит отметить, что эти методы реже применяются к временным рядам. 

 Наиболее популярным методом прогнозирования временных рядов стали модели ARIMA. Подробный пример использования данной авторегрессионной модели приведён в статье [31], где авторы прогнозируют статистические показатели для COVID-19. Было показано, что с помощью ARIMA можно добиться реалистичных и адекватных предсказаний временных рядов. 

## Prediction with exogenous variables

Наличие внешних (экзогенных) переменных, скоррелированных с прогнозируемой, может улучшить качество предсказания. Для этого используются модели ARIMAX/SARIMAX, являющиеся расширением ARIMA/SARIMA, учитывающих экзогенные параметры. В публикации [33] модели ARIMA и ARIMAX сравнивались в задаче прогнозирования урожая сахарного тростника в Харьяне, Индии. Было показано, что привлечение погодных условий в качестве экзогенных переменных, позволило уменьшить ошибки прогнозирования.

## Impact of context information on consumption

В рамках данной работы была выдвинута гипотеза, что временные ряды могут быть зависимыми от других переменных, измеряемых во времени. В частности, потребительская активность может быть в какой-то мере объяснена контекстными переменными, такими как экономическими параметрами или популярностью тем в новостных источниках. Несмотря на то, что автор публикации [32] опровергает теорию Дэвида Юма, гласящую о наблюдаемых примерах причинности как частном случае общей системы мира, знание о контексте процесса может быть использовано для улучшения качества прогнозов. Это связано с возможностью непрямой взаимосвязи процесса и контекста, когда речь не идёт о простых причинно-следственных отношениях. В математических терминах можно назвать такую связь корреляцией. 
