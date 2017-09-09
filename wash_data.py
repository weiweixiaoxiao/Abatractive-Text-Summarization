#coding:utf-8
import re
import jieba
from gensim.models import word2vec
import sys
from idlelib.IOBinding import encoding
import numpy as np
import random

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
'''
读入txt文件{"summarization": "知情人透露章子怡怀孕后，父母很高兴。章母已开始悉心照料。据悉，预产期大概是12月底", "article": "当晚9时，华西都市报记者为了求证章子怡，但电话通了，一直没有人<Paragraph>。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打，相信9月26日的演唱会应该还会有惊喜大白天下吧。"}
对txt文件处理,分成article、summary两个文件
'''
def readTXT(inFile,articleFile,summaryFile):
    reader = open(inFile, 'r',encoding = 'UTF-8')
    contentlines = reader.readlines()
    articleWriter = open(articleFile, 'w',encoding = 'UTF-8')
    summaryWriter = open(summaryFile, 'w',encoding = 'UTF-8')
    for str in contentlines:
#         strlist = re.findall('{"summarization": "(.+)", "article": "(.+)"}',str)
        strlist = re.findall('{"summarization": "", "article": "(.+)"}',str)
        for i in range(len(strlist[0])):
            if i==0:
                summaryWriter.write(strlist[0][0]+'\n')
            else:
                print("strlist == ",strlist[0][i])
                t_utf1 = re.sub("<","。",strlist[0][i])
                t_utf = re.sub("[A-Z|a-z>]","",t_utf1)
                print("t_utf == ",t_utf)
                articleWriter.write(t_utf+'\n')
    reader.close()
    articleWriter.close()
    summaryWriter.close()  
def readTXTWithoutSum(inFile,articleFile):
    reader = open(inFile, 'r',encoding = 'UTF-8')
    contentlines = reader.readlines()
    articleWriter = open(articleFile, 'w',encoding = 'UTF-8')
    for str in contentlines:
        strlist = re.findall('{"summarization": "", "article": "(.+)"}',str)
        for i in range(len(strlist)):
            print("strlist == ",strlist[i])
            t_utf1 = re.sub("<","。",strlist[i])
            t_utf = re.sub("[A-Z|a-z>]","",t_utf1)
            print("t_utf == ",t_utf)
            articleWriter.write(t_utf+'\n')
    reader.close()
    articleWriter.close()
'''
分词jieba精确模式
按行分词，一个article为一行
''' 
def wordSeg(file_name,segFile):
    reader = open(file_name, 'r',encoding = 'UTF-8')
    Writer = open(segFile, 'w',encoding = 'UTF-8')
    content = reader.readlines()
    for i in range(len(content)):
        terms = jieba.cut(content[i],cut_all=False)
        #print(terms)
        Writer.write(" ".join(terms))
        #Writer.write('\n')
    reader.close()
    Writer.close()
    print("-----wordSeg--end--------") 

'''
分词向量表示：使用gensim word2vec训练数据，持久化训练模型到wordModel
filename:分好词的语料库
wordMode：训练好的数据
'''
def word2vecTrain(filename,wordMode):
    sentences = word2vec.Text8Corpus(filename)   
    print("sentences == ",sentences)
    model = word2vec.Word2Vec(sentences,min_count=1,size = 256) 
    model.save(wordMode)
    #model = word2vec.Word2Vec.load(wordMode)    
    print("-----word2vecTrain--end--------") 
    
def word2vecVocab(wordMode,wordModeOutFile,Vocab):
    wordEmbedding=np.zeros([Vocab._count,256])
    model = word2vec.Word2Vec.load(wordMode)
    wordList = Vocab._id_to_word
    for i in range(len(wordList)):
        if wordList[i] in model.wv.vocab:
            wordEmbedding[i] = model[wordList[i]]
        if wordList[i] == '<s>':
            wordEmbedding[i] = [random.uniform(-1, 1) for i in range(256)]
        if wordList[i] == '</s>':
            wordEmbedding[i] = model['。']
        if wordList[i] == '<PAD>':
            wordEmbedding[i] = [random.uniform(-1, 1) for i in range(256)]
        print("{}: {} = {}".format(i,wordList[i],wordEmbedding[i]))
    np.save(wordModeOutFile,wordEmbedding)
'''
建立词表vocab
filename:分好词的语料库
wordMode：训练好的词嵌入数据
vocFile:词表存放文件,按次数从大到小存放
maxnumber:词典的大小50000
'''
def vocableBuild(filename,wordMode,vocabFile,maxnumber):
    reader = open(filename, 'r',encoding = 'UTF-8')
    content = reader.read().split()
    wordDic = {}
    for i in range(len(content)):
        if wordDic.get(content[i]):
            wordDic[content[i]] += 1 
        else:
            wordDic[content[i]] = 1
    wordList = sorted(wordDic.items(),key=lambda t:t[1],reverse=True) 
    writer = open(vocabFile, 'w',encoding = 'UTF-8') 
    num = len(wordList) - maxnumber 
    temp = True
    for i in range(maxnumber):
        if(num > wordList[i][1] and temp):
            writer.write("<UNK>"+" "+str(num))
            writer.write("\n")
            temp = False
        print("{} {}".format(wordList[i][0],str(wordList[i][1])))
        writer.write(wordList[i][0]+" "+str(wordList[i][1]))
        writer.write("\n")   
    reader.close()
    writer.close()
    print("-----vocableBuild--end--------") 
  
def buildTrainEvalData(articleFile,summaryFile,articleTrain,articleEval,summaryTrain,summaryEval,
                       trainNum,evalNum):
    artireader = open(articleFile, 'r',encoding = 'UTF-8')
    sumreader = open(summaryFile, 'r',encoding = 'UTF-8')
    
    articleTrainWriter = open(articleTrain, 'w',encoding = 'UTF-8')
    summaryTrainWriter = open(summaryTrain, 'w',encoding = 'UTF-8')
    articleEvalWriter = open(articleEval, 'w',encoding = 'UTF-8')
    summaryEvalWriter = open(summaryEval, 'w',encoding = 'UTF-8')
    for _ in range(trainNum):
        articlecontent = artireader.readline()
        print("articlecontent = ",articlecontent)
        articleTrainWriter.write(articlecontent)
        summarycontent = sumreader.readline()
        summaryTrainWriter.write(summarycontent)  
    articleTrainWriter.close()
    summaryTrainWriter.close()     
    for _ in range(evalNum):
        articlecontent = artireader.readline()
        articleEvalWriter.write(articlecontent)
        summarycontent = sumreader.readline()
        summaryEvalWriter.write(summarycontent)  
    articleEvalWriter.close()
    summaryEvalWriter.close()    
               
    artireader.close()
    sumreader.close() 
    articleEvalWriter.close()
    summaryEvalWriter.close() 
'''
处理训练集，将其存到字典中
'''  
UNKNOWN_TOKEN = '<UNK>'    
class Vocab(object):
    '''
     __init__ 将vocab变成wordtoid和idtoword存放在字典中
     vocabFile：字典文件
     max_size：从字典中取出的最大数量
    '''
    def __init__(self, vocabFile, max_size):
        print("1--wash_data")
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 #词语的id
        with open(vocabFile, 'r',encoding = 'utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split() #'说' 70981
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1
                if self._count > max_size:
                    raise ValueError('Too many words: >%d.' % max_size)
    #判断vocab是否存在word，如果不存在返回false，存在返回true    
    def CheckVocab(self, word):
        if word not in self._word_to_id:
            return False
        return True
      
    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]
    
    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]
    #vocab的总数
    def NumIds(self):
        return self._count
'''
输入一个样本数据，将其变为id作为模型的输入
vocab：class Vocab的实例 如vocab = data.Vocab(vocab_path, 1000000)
将句子分为一个一个词，如[我，爱，中国]
'''
def GetWordIds(textwordSeg, vocab):
    ids = []
    for w in textwordSeg.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    return ids


def Ids2Words(ids_list, vocab):
    """Get words from ids.将ids转为词组
    
    Args:
        ids_list: list of int
        vocab: TextVocabulary object
               class Vocab的实例 ，如vocab = data.Vocab(vocab_path, 1000000)
    
    Returns:
        List of words corresponding to ids.
    """
    assert isinstance(ids_list, list), '%s  is not a list' % ids_list
    return [vocab.IdToWord(i) for i in ids_list]


def ExampleGen(articleWordSegpath,summaryWordSegpath = None):
    """Generates tf.Examples from path of data files.
        
        generater函数，每次读取一个样本的article和summarization，
                        生成article和summarization，作为一个读取过程，具体有batch_reader里调用
        
    Args:
        articleWordSegpath: path to article分词结果文件.
        summaryWordSegpath：  path to summarization分词结果文件.
        
    Yields:
                       每次被调用一次就返回一个 article,summary
        
    If there are multiple files specified, they accessed in a random order.
    """
    content = []
    if summaryWordSegpath != None :
        articleReader = open(articleWordSegpath,'r',encoding = 'utf-8')
        summaryReader = open(summaryWordSegpath,'r',encoding = 'utf-8')
        articles = articleReader.readlines()
        summarys = summaryReader.readlines()
        print("len articles == ",len(articles))
        print("len summarys == ",len(summarys))
        for i in range(len(articles)):
            article = articles[i]
            summary = summarys[i]
            content.append((len(article.split()),len(summary.split()),article,summary))
    else:
        articleReader = open(articleWordSegpath,'r',encoding = 'utf-8')
        articles = articleReader.readlines()
        for i in range(len(articles)):
            article = articles[i]
            summary = ''
            content.append((len(article.split()),0,article,summary))   
    return content          

def ExampleGen_predict(articleWordSegpath):
    """Generates tf.Examples from path of data files.
        
        generater函数，每次读取一个样本的article
                        生成article，作为一个读取过程，具体有batch_reader里调用
        
    Args:
        articleWordSegpath: path to article分词结果文件.
        
    Yields:
                       每次被调用一次就返回一个 article
        
    If there are multiple files specified, they accessed in a random order.
    """
    articleReader = open(articleWordSegpath,'r',encoding = 'utf-8')
    articles = articleReader.readlines() 
    return articles          

def ToSentences(article, include_token=True):
    """Takes tokens of a paragraph and returns list of sentences.
                    将句子分割，放入一个list中，每个元素是一个以空格分开的句子，如[我 爱 中国，我 是 中国人]
    Args:
        article: string, text of article
        include_token: Whether include the sentence separation tokens result.
    
    Returns:
        List of sentence strings.
    """
    s_gen = []
    sentence = ""
    article_sentence = article.split()
    for i in range(len(article_sentence)):
        sentence += article_sentence[i]
        sentence += " "
        if article_sentence[i] =='。':
            s_gen.append(sentence)
            sentence = ""
        if  article_sentence[i]!='。' and i == len(article_sentence)-1:
            s_gen.append(sentence)
            sentence = ""
    return s_gen

def Pad(ids, pad_id, length):
    """Pad or trim list to len length.
    
    Args:
        ids: list of ints to pad
        pad_id: what to pad with
        length: length to pad or trim to
    
    Returns:
        ids trimmed or padded with pad_id
    """
    assert pad_id is not None
    assert length is not None
    
    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]
         
if __name__ == '__main__':
#     readTXT('dataTest/evaluation_without_ground_truth.txt','dataTest/articleTest.txt','dataTest/summarizationTest.txt')
    readTXTWithoutSum('dataTest/evaluation_without_ground_truth.txt','dataTest/articleTest.txt')
    #wordSeg('dataTest/summarization.txt','dataTest/summarizationWordSeg.txt')
    #wordSeg('dataTest/article.txt','dataTest/articleWordSeg.txt')
    #word2vecTrain('dataTest/ALLSourceSenData.bpe.txt','dataTest/ALLSourceSenData.bpe.model')
#     vocab = Vocab('dataTest/allVector.bpe.txt', 1000000)
#     word2vecVocab('dataTest/ALLSourceSenData.bpe.model','dataTest/wordEmbedding.npy',vocab)
#     
    '''
    writer = open('dataTest/allWordSeg.txt',"a",encoding = 'utf-8')
    inputFile = open('dataTest/articleWordSeg.txt',"r",encoding = 'utf-8')
    strs = inputFile.readlines()
    for line in strs:
        writer.write(line)
    inputFile.close()    
    inputFile1 = open('dataTest/summarizationWordSeg.txt',"r",encoding = 'utf-8')
    strs1 = inputFile1.readlines()
    for line in strs1:
        writer.write(line)
    inputFile1.close()    
    writer.close()
    vocableBuild('dataTest/allWordSeg.txt','dataTest/article.model','dataTest/allVector.txt',70000)
    print("ending")
    '''
    #buildTrainEvalData('data/articleWordSeg.txt','data/summarizationWordSeg.txt','dataTest/articleTrainData.txt','dataTest/articleEvalData.txt','dataTest/summaryTrainData.txt','dataTest/summaryEvalData.txt',
    #                   45000,5000)
    print("ending")
