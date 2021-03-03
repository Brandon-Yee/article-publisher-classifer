# Article Publisher Classification

The goal of this project is to classify article publishers based on the article text and title using statisical learning. 

Our classifier uses a neural network to perform the article classification.

We used a dataset from Components of 2.7 million news articles and essays from 27 different publications. The dataset can be found at the following link: https://components.one/datasets/all-the-news-2-news-articles-dataset/


Publication classes:
1. Reuters        840,094
2. TechCrunch     52,095
3. Economist      26,227
4. CNN            127,602
5. CNBC           238,096
6. Fox News       20,144
7. Politico       46,377
8. The New York Times 252,259
9. Washington Post    40,882
10. Business Insider  57,953
11. Other


Notes:
- The date ranges by class are very different, so we are not using it as a feature.
- 'section' feature seems unique to class
