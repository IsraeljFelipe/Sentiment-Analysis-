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
var_file = open("D:\xxx.txt", "a")
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
var_file = open("D:\xxx.txt", "a")
for word in palavrasFinal:
	var_file.write(word.lower()+'\n')

var_file.close()


###Part II - Sentimental analysis###
import nltk
import numpy
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import PlaintextCorpusReader
corpus_root = r"D:\xxx"
wordlists = PlaintextCorpusReader(corpus_root, '.*')
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from __future__ import division
from IndexedText import *
ListNegation = ['no', 'nobody', 'num', 'nothing', 'none', 'never', 'never']

#declaring function responsible for counting denials
def CountNegation(Text, word, ListNegation):
	listNeg = Text.LeftContext(word, 3)
	contTot = 0

	for item in listNeg:
		cont = set(ListNegation) & set(item)
		contTot = contTot + len(cont)
		
	return contTot

#loading negative words dictionary
texto = open('D:\xxx.txt').read()
negativas = tokenizer.tokenize(texto)

#loading dictionary of positive words
texto = open('D:\xxx.txt').read()
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

	listaResultNeg.append([nameFile, year, TermWeighting])

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

	listaResultPos.append([nameFile, year, TermWeighting])


#building final list
listaResultado = []
i = 0
for i in range(0, len(listaResultNeg)):
	listaResultado.append([listaResultNeg[i][0], listaResultNeg[i][1], listaResultNeg[i][2], listaResultPos[i][2]])
	
	
#saving the final archive of news sentiment analysis
var_file = open("D:\xxx.txt", "a")
for item in listaResultado:
	var_file.write(item[0]+";"+item[1]+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()


#Part III - Terms weigths
import nltk
import numpy
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import PlaintextCorpusReader
corpus_root = r"D:\xxx"
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
texto = open('D:\xxx.txt').read()
negativas = tokenizer.tokenize(texto)

#loading dictionary of positive words
texto = open('D:\xxx.txt').read()
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
var_file = open("D:\xxx.txt", "a")
for item in listaNeg:
	var_file.write(item[0]+";"+str(item[1])+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()

#Saving the positive word weights list file
var_file = open("D:\xxx.txt", "a")
for item in listaPos:
	var_file.write(item[0]+";"+str(item[1])+";"+str(item[2])+";"+str(item[3])+";"+'\n')
	
var_file.close()
