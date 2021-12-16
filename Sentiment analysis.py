###Part I - elaboration of the dictionary of words taken from the news###




import nltk
from nltk.corpus import PlaintextCorpusReader

#loading corpus words to generate dictionary
corpus_root = r"D:\corpus"
wordlists = PlaintextCorpusReader(corpus_root, '.*')
texto = wordlists.words()
palavrasDistintas = set(texto)

#generating stopwords list
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.tokenize import WordPunctTokenizer
portugues_stops = set(stopwords.words('portuguese'))

portugues_stop2 = ''
for word in portugues_stops:
	portugues_stop2 = portugues_stop2 + ' ' + normalize('NFKD', word).encode('ASCII','ignore')
	
tokenizer = WordPunctTokenizer()
portugues_stop3 = tokenizer.tokenize(portugues_stop2)
var_file = open("D:\file1.txt", "a")
for word in portugues_stop3:
	var_file.write(word.lower()+'\n')
var_file.close()

#removing stopwords from the dictionary
palavrasSemStop = [word.lower() for word in palavrasDistintas if word.lower() not in portugues_stop3]

#removing numbers from the list
lista = []
for word in palavrasSemStop:
	for i in range(10):
		if (word.startswith(str(i))) or (word.endswith(str(i))) or (str(i) in word):
			lista.append(word)

palavrasSemNumero = [word.lower() for word in palavrasSemStop if word.lower() not in lista]

#removing symbols and special characters
import re
palavrasSemSimbolos = [w for w in palavrasSemNumero if re.search('^[a-z]+$', w)]

#removing words with less than 3 characters
palavrasFinal = sorted([w for w in set(palavrasSemSimbolos) if len(w) >= 3])

#saving the dictionary of words
var_file = open("D:\file2.txt", "a")
for word in palavrasFinal:
	var_file.write(word.lower()+'\n')

var_file.close()


###Part II - Sentiment analysis###
import nltk
import numpy
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import PlaintextCorpusReader
corpus_root = r"D:\corpus"
wordlists = PlaintextCorpusReader(corpus_root, '.*')
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from __future__ import division
from IndexedText import *
ListNegation = ['nao', 'ninguem', 'num', 'nada', 'nenhum', 'nunca', 'jamais']

#declaring function responsible for counting denials
def CountNegation(Text, word, ListNegation):
	listNeg = Text.LeftContext(word, 3)
	contTot = 0

	for item in listNeg:
		cont = set(ListNegation) & set(item)
		contTot = contTot + len(cont)
		
	return contTot

#loading negative words dictionary
texto = open('D:\file3s.txt').read()
negativas = tokenizer.tokenize(texto)

#loading positive words dictionary
texto = open('D:\file4.txt').read()
positivas = tokenizer.tokenize(texto)

#building the frequency distribution of texts
listaGeral = []
for fileid in wordlists.fileids():
	freqFile = FreqDist(wordlists.words(fileid))
	listaFile = [fileid, freqFile]
	listaGeral.append(listaFile)

	
#N value = Total texts
N = len(listaGeral)

#reconstructing the matrix with the mean of the negative words
listaAjNeg = []
for fileFreq in listaGeral:
	cont = 0
	tot = 0
	for word in negativas:
		if fileFreq[1][word] > 0:
			cont = cont + fileFreq[1][word]
			tot = tot + 1
		
	if tot > 0:
		media = cont / tot
	else:
		media = 0
	
	Aj = [fileFreq[0], fileFreq[1], media]
	listaAjNeg.append(Aj)
	
#reconstructing the matrix with the average of the positive words
listaAjPos = []
for fileFreq in listaGeral:
	cont = 0
	tot = 0
	for word in positivas:
		if fileFreq[1][word] > 0:
			cont = cont + fileFreq[1][word]
			tot = tot + 1
		
	if tot > 0:
		media = cont / tot
	else:
		media = 0
	
	Aj = [fileFreq[0], fileFreq[1], media]
	listaAjPos.append(Aj)


#defining negative DFi
listaNeg = []
for word in negativas:
	contFile = 0
	for fileFreq in listaAjNeg:
		cont = fileFreq[1][word]
		if cont > 0:
			contFile = contFile + 1

	listaNeg.append([word, contFile])

#defining positive DFi
listaPos = []
for word in positivas:
	contFile = 0
	for fileFreq in listaAjPos:
		cont = fileFreq[1][word]
		if cont > 0:
			contFile = contFile + 1

	listaPos.append([word, contFile])
	
#building list of negative weighted texts
listaResultNeg = []
for item in listaAjNeg:
	nomeArq = item[0][4:]
	ano = item[0][4:8]
	Aj = item[2]
	TermWeighting = 0
	texto = IndexedText(wordlists.words(item[0]))

	for word in listaNeg:
		palavra = word[0]
		DFi = word[1]
		TFij = item[1][palavra] - CountNegation(texto, palavra, ListNegation)
		if TFij > 0:
			Wij = (((1 + numpy.log(TFij)) / (1 + numpy.log(Aj))) * numpy.log(N / DFi))
		else:
			Wij = 0
		TermWeighting = TermWeighting + Wij

	listaResultNeg.append([nomeArq, ano, TermWeighting])

#building positive weighted text list
listaResultPos = []
for item in listaAjPos:
	nomeArq = item[0][4:]
	ano = item[0][4:8]
	Aj = item[2]
	TermWeighting = 0
	texto = IndexedText(wordlists.words(item[0]))

	for word in listaPos:
		palavra = word[0]
		DFi = word[1]
		TFij = item[1][palavra] - CountNegation(texto, palavra, ListNegation)
		if TFij > 0:
			Wij = (((1 + numpy.log(TFij)) / (1 + numpy.log(Aj))) * numpy.log(N / DFi))
		else:
			Wij = 0
		TermWeighting = TermWeighting + Wij

	listaResultPos.append([nomeArq, ano, TermWeighting])


#building final list
listaResultado = []
i = 0
for i in range(0, len(listaResultNeg)):
	listaResultado.append([listaResultNeg[i][0], listaResultNeg[i][1], listaResultNeg[i][2], listaResultPos[i][2]])
	
	
#saving the final archive of news sentiment analysis
var_file = open("D:\file5.txt", "a")
for item in listaResultado:
	var_file.write(item[0]+";"+item[1]+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()



###Part III - Terms weigths###
import nltk
import numpy
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import PlaintextCorpusReader
corpus_root = r"D:\corpus"
wordlists = PlaintextCorpusReader(corpus_root, '.*')
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from __future__ import division
from IndexedText import *
ListNegation = ['nao', 'ninguem', 'num', 'nada', 'nenhum', 'nunca', 'jamais']

#declaring function responsible for counting denials
def CountNegation(Text, word, ListNegation):
	listNeg = Text.LeftContext(word, 3)
	contTot = 0

	for item in listNeg:
		cont = set(ListNegation) & set(item)
		contTot = contTot + len(cont)
		
	return contTot

#loading negative word dictionary
texto = open('D:\file3.txt').read()
negativas = tokenizer.tokenize(texto)

#loading dictionary of positive words
texto = open('D:\file4.txt').read()
positivas = tokenizer.tokenize(texto)

#building the frequency distribution of texts
listaGeral = []
for fileid in wordlists.fileids():
	freqFile = FreqDist(wordlists.words(fileid))
	listaFile = [fileid, freqFile]
	listaGeral.append(listaFile)

	
#N value = Total texts
N = len(listaGeral)

#reconstructing the matrix with the mean of the negative words
listaAjNeg = []
for fileFreq in listaGeral:
	cont = 0
	tot = 0
	for word in negativas:
		if fileFreq[1][word] > 0:
			cont = cont + fileFreq[1][word]
			tot = tot + 1
		
	if tot > 0:
		media = cont / tot
	else:
		media = 0
	
	Aj = [fileFreq[0], fileFreq[1], media]
	listaAjNeg.append(Aj)
	
#reconstructing the matrix with the average of the positive words
listaAjPos = []
for fileFreq in listaGeral:
	cont = 0
	tot = 0
	for word in positivas:
		if fileFreq[1][word] > 0:
			cont = cont + fileFreq[1][word]
			tot = tot + 1
		
	if tot > 0:
		media = cont / tot
	else:
		media = 0
	
	Aj = [fileFreq[0], fileFreq[1], media]
	listaAjPos.append(Aj)


#defining total occurrences of texts with negative words
negativasTot = []
for word in negativas:
	contFile = 0
	contWord = 0
	for fileFreq in listaAjNeg:
		cont = fileFreq[1][word]
		contWord = contWord + cont
		if cont > 0:
			contFile = contFile + 1
		
	negativasTot.append([word, contFile, contWord])	

	
#defining total occurrences of texts with positive words
positivasTot = []
for word in positivas:
	contFile = 0
	contWord = 0
	for fileFreq in listaAjPos:
		cont = fileFreq[1][word]
		contWord = contWord + cont
		if cont > 0:
			contFile = contFile + 1
		
	positivasTot.append([word, contFile, contWord])

#defining weight of negative words considering all texts
listaNeg = []
for item in negativasTot:
	TermWeighting = 0
	word = item[0]
	DFi = item[1]
	contWord = item[2]
	for fileFreq in listaAjNeg:
		texto = IndexedText(wordlists.words(fileFreq[0]))
		TFij = fileFreq[1][word] - CountNegation(texto, word, ListNegation)
		Aj = fileFreq[2]
		if TFij > 0:
			Wij = (((1 + numpy.log(TFij)) / (1 + numpy.log(Aj))) * numpy.log(N / DFi))
		else:
			Wij = 0
		
		TermWeighting = TermWeighting + Wij
	
	listaNeg.append([word, contWord, DFi, TermWeighting])

#defining weight of positive words considering all texts
listaPos = []
for item in positivasTot:
	TermWeighting = 0
	word = item[0]
	DFi = item[1]
	contWord = item[2]
	for fileFreq in listaAjPos:
		texto = IndexedText(wordlists.words(fileFreq[0]))
		TFij = fileFreq[1][word] - CountNegation(texto, word, ListNegation)
		Aj = fileFreq[2]
		if TFij > 0:
			Wij = (((1 + numpy.log(TFij)) / (1 + numpy.log(Aj))) * numpy.log(N / DFi))
		else:
			Wij = 0
	
		TermWeighting = TermWeighting + Wij
	
	listaPos.append([word, contWord, DFi, TermWeighting])
	
#saving the negative word weights list file
var_file = open("D:\file6.txt", "a")
for item in listaNeg:
	var_file.write(item[0]+";"+str(item[1])+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()

#Saving the positive word weights list file
var_file = open("D:\file7.txt", "a")
for item in listaPos:
	var_file.write(item[0]+";"+str(item[1])+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()


####Part IV - Social media procedures:*****************************************************************************************************************

#A- Manipulation of tweets with text cleaning and preparation
import os
import re
import unicodedata

class TweetManipulation():
    
    def __init__(self, jornal):
        self.jornal = jornal 

    def cleanTweet(self, tweet):        
        nfkd = unicodedata.normalize('NFKD', tweet)
        word_without_accent = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        
        return re.sub("(@ [A-Za-z0-9]+)|([^0-9A-Za-z $])|(\w+:\/\/\S+)", "", word_without_accent)

    def translateMonth(self, month):  
        months = {
            'Jan': '01',
            'Feb': '02',
            'Mar': '03',
            'Apr': '04',
            'May': '05',
            'Jun': '06',
            'Jul': '07',
            'Aug': '08',
            'Sep': '09',
            'Oct': '10',
            'Nov': '11',
            'Dec': '12'   
        }

        return months[month]

    def formatDate(self, full_tweet):        
        date = full_tweet.split('\t')[0]

        day = date.split(' ')[0] if (len(date.split(' ')[0]) > 1) else '0'+date.split(' ')[0]
        month = self.translateMonth(date.split(' ')[1])
        year = date.split(' ')[2]

        date = year+month+day
        return date

    def formatText(self, full_tweet):
        text = full_tweet.split('\t')[1].split("http")[0]
        text = self.cleanTweet(text)
        
        return text

    def validationDate(self, full_tweet):        
        tweet = full_tweet.split('\t')

        if(tweet[0] != '' and len(tweet) == 2 and tweet[1][0] != 'No results'):
            return 1
        else:
            return 0
        

    def init(self):        
        tweetsFile = list(open('./INPUT/%s.txt' %self.jornal, 'r'))        
        output = open('./OUTPUT/%s.txt' %self.jornal, 'w+')        
        
        for tweet in tweetsFile:
            full_tweet = tweet.rstrip()

            if(self.validationDate(full_tweet)):
                date = self.formatDate(full_tweet)
                text = self.formatText(full_tweet=full_tweet)
                new_tweet = date + ";" + text.strip() + ";%s" %self.jornal
                output.write(new_tweet+'\n') 

        output.close()

        # -*- coding: utf-8 -*-
import shutil


#B- Creation Corpus
class TweetCorpus():
    
    def __init__(self, jornal):
        self.jornal = jornal    
    
    def compare(self, a, b):
        if a == b:
            return 1
        
        return 0
    
    def createCorpus(self, dateFile):
        
        for date in dateFile:
            tweetStack = []
            for i in self.jornal:
                tweetsFile = list(open('./OUTPUT/%s.txt' %i, 'r'))
                date = date.split('\n')[0]
                newFile = open('./OUTPUT/corpus/%s.txt' %date, 'w+')
    
                for tweet in tweetsFile:
                    if(self.compare(date, tweet.split(';')[0])):
                        tweetStack.append(str(tweet.split(';')[1]+'\n'))
        
                newStack = set(tweetStack)
                for tweet in newStack:
                    newFile.write(tweet)
                newFile.close()
                
                
from cleanInput import TweetManipulation
from createCorpus import TweetCorpus


if __name__ == '__main__': 
    
    newspaper = ['ESTADAO']
    
    for i in newspaper:        
        tweetManipulation = TweetManipulation(i)
        tweetManipulation.init()
    
    tweetCorpus = TweetCorpus(newspaper)
    dateFile = list(open('./datas.txt', 'r'))    
    tweetCorpus.createCorpus(dateFile)


#C- Use the same algorithm for dictionary generation as in line 9 (loading corpus words to generate dictionary)


#D- Classification of the twitter corpus
import nltk
import codecs

import shutil

global positive
global negative


positive = open('./OUTPUT/positive.txt', 'w+')
negative = open('./OUTPUT/negative.txt', 'w+')


def dataPositive(word):
  positive.write('%s\n' %word)

def dataNegative(word):
  negative.write('%s\n' %word)


  

def sentiLexFlexToDict():
  fileName = "SentiLex-flex-PT02.txt"
  inputFile = codecs.open(fileName,"rb","utf-8")
  expressions = {}
  for line in inputFile:
    splitted = line.split(";")
    DicionarioAux = {}
    if(len(splitted[1].split('=')) == 1):
      DicionarioAux [splitted[1].split('=')[0]] = ''
    else:
      DicionarioAux [splitted[1].split('=')[0]] = splitted[1].split('=')[1] 

    DicionarioAux [splitted[2].split('=')[0]] = splitted[2].split('=')[1]
    DicionarioAux [splitted[3].split(':')[0]] = splitted[3].split(':')[1].split('=')[1]

    if (splitted[4].split('=')[0] == 'ANOT'):
      DicionarioAux [splitted[4].split('=')[0]] = splitted[4].split('=')[1]
    else:
      DicionarioAux [splitted[4].split('=')[0].split(':')[0] ] = splitted[4].split('=')[1]
    
    expressions[splitted[0].split(",")[0]] = DicionarioAux 
    expressions[splitted[0].split(",")[1]] = DicionarioAux
   
  return expressions #SentiLex-Flex em Dict



def polarity(dictOfExpressions,textToAnalyse):
      
  tokens = nltk.word_tokenize(textToAnalyse)
  result = 0  

  for expression in dictOfExpressions.keys():
    if (expression in tokens): 
        if dictOfExpressions[expression]['POL'] == '-1':
          dataNegative(tokens[0])

        elif dictOfExpressions[expression]['POL'] == '1':
          dataPositive(tokens[0])

        
          

       
  return result

#E- Establishing the polarity of tweets
import classificationAux

def analyseSentiment():
  lista_palavras = list(open('./INPUT/lista-palavras.txt', 'r'))

  sentilex = classificationAux.sentiLexFlexToDict() 
 

  for palavra in lista_palavras:
    sentimentValue = classificationAux.polarity(sentilex,palavra)  

analyseSentiment() 


#F- Use sentiment analysis, the same sentiment analysis procedure for tweets, as detailed in line 59 (Part II - Sentimental analysis)


#G- Use sentiment analysis, the same procedure as for generating the weights of the terms of the tweets, as detailed in line 220 (Part III - Terms weigths)