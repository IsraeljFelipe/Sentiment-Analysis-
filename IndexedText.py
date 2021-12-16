import nltk
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

class IndexedText(object):
    def __init__(self, text):
        self._text = text
        self._index = nltk.Index((word, i) for (i, word) in enumerate(text))
    
    def concordanceLeft(self, word, width=40):
        key = word
        wc = width/4
	listaEsq = []
	for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            ldisplay = '%*s'  % (width, lcontext[-width:])
            
            sentLeft = tokenizer.tokenize(ldisplay)
            listaEsq.append(sentLeft)
            
	return listaEsq

    def concordanceRight(self, word, width=40):
        key = word
        wc = width/4
	listaDir = []
        for i in self._index[key]:
            rcontext = ' '.join(self._text[i:i+wc])
            rdisplay = '%-*s' % (width, rcontext[:width])

            sentRight = tokenizer.tokenize(rdisplay)
            listaDir.append(sentRight)

	return listaDir

    def LeftContext(self, word, numberWords):
        listEsq = self.concordanceLeft(word)
        listLeft = []
        oldNumberWords = numberWords
        for item in listEsq:
            left = ""
            while numberWords > 0:
                if (len(item)-numberWords) > 0:
                    left = left + item[len(item)-numberWords].lower() + " "
                numberWords = numberWords - 1

            listLeft.append(tokenizer.tokenize(left))
            numberWords = oldNumberWords 
            	    
	return (listLeft)

    def RightContext(self, word, numberWords):
        listDir = self.concordanceRight(word)
	listRight = []
	for item in listDir:
            right = ""
            for j in range(0, numberWords):
                right = right + item[1+j].lower() + " "

            listRight.append(tokenizer.tokenize(right))
            
        return (listRight)
