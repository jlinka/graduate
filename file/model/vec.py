from gensim.models import Word2Vec
 
en_wiki_word2vec_model = Word2Vec.load('wordvec.model')
testwords = ['按照', '远洋', '百度']
for i in testwords:
    res = en_wiki_word2vec_model.most_similar(i)
    print(i)
    print('----------------')
    for j in res:
        print(j)
    
    #print(res)


