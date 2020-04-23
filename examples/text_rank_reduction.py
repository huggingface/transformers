import re, pdb, sys, math
from collections import defaultdict

class Graph:
    def __init__(self):
        self.Vertices = []
        self.Edges = []

    def getRankedVertices(self):
        res = defaultdict(float)
        for e in self.Edges:
            res[e.Vertex1] += e.Weight
        return sorted(res.items(), key=lambda x: x[1], reverse=True)

class Vertex:
    def __init__(self):
        self.Sentence = None

class Edge:
    def __init__(self):
        self.Vertex1 = None
        self.Vertex2 = None
        self.Weight = 0

class WordType:
    Content=0
    Function=1
    ContentPunctuation=2
    FunctionPunctuation=3

class Word:
    def __init__(self):
        self.Text=''
        self.Type=''

class Sentence:
    def __init__(self):
        self.Words = []

    def getFullSentence(self):
        text = ''
        for w in self.Words:
            text += w.Text
        return text.strip()

    def getReducedSentence(self):
        sentenceText = ''
        sentenceEnd = self.Words[len(self.Words)-1]
        contentWords = filter(lambda w: w.Type == WordType.Content, self.Words)
        i = 0
        while i < len(contentWords):
            w = contentWords[i]
            # upper case the first character of the sentence
            if i == 0:
                li = list(w.Text)
                li[0] = li[0].upper()
                w.Text = ''.join(li)
            sentenceText += w.Text
            if i < len(contentWords)-1:
                sentenceText += ' '
            elif sentenceEnd.Text != w.Text:
                sentenceText += sentenceEnd.Text
            i = i+1
        return sentenceText

class Paragraph:
    def __init__(self):
        self.Sentences = []

class Reduction:
    functionPunctuation = ' ,-'
    contentPunctuation = '.?!\n'
    punctuationCharacters = functionPunctuation+contentPunctuation
    sentenceEndCharacters = '.?!'

    def isContentPunctuation(self, text):
        for c in self.contentPunctuation:
            if text.lower() == c.lower():
                return True
        return False

    def isFunctionPunctuation(self, text):
        for c in self.functionPunctuation:
            if text.lower() == c.lower():
                return True
        return False

    def isFunction(self, text, stopWords):
        for w in stopWords:
            if text.lower() == w.lower():
                return True
        return False

    def tag(self, sampleWords, stopWords):
        taggedWords = []
        for w in sampleWords:
            tw = Word()
            tw.Text = w
            if self.isContentPunctuation(w):
                tw.Type = WordType.ContentPunctuation
            elif self.isFunctionPunctuation(w):
                tw.Type = WordType.FunctionPunctuation
            elif self.isFunction(w, stopWords):
                tw.Type = WordType.Function
            else:
                tw.Type = WordType.Content
            taggedWords.append(tw)
        return taggedWords

    def tokenize(self, text):
        return filter(lambda w: w != '', re.split('([{0}])'.format(self.punctuationCharacters), text))	

    def getWords(self, sentenceText, stopWords):
        return self.tag(self.tokenize(sentenceText), stopWords) 

    def getSentences(self, line, stopWords):
        sentences = []
        sentenceTexts = filter(lambda w: w.strip() != '', re.split('[{0}]'.format(self.sentenceEndCharacters), line))	
        sentenceEnds = re.findall('[{0}]'.format(self.sentenceEndCharacters), line)
        sentenceEnds.reverse()
        for t in sentenceTexts:
            if len(sentenceEnds) > 0:
                t += sentenceEnds.pop()
            sentence = Sentence()
            sentence.Words = self.getWords(t, stopWords)
            sentences.append(sentence)
        return sentences

    def getParagraphs(self, lines, stopWords):
        paragraphs = []
        for line in lines:
            paragraph = Paragraph()
            paragraph.Sentences = self.getSentences(line, stopWords)
            paragraphs.append(paragraph)
        return paragraphs

    def findWeight(self, sentence1, sentence2):
        length1 = len(list(filter(lambda w: w.Type == WordType.Content, sentence1.Words)))
        length2 = len(list(filter(lambda w: w.Type == WordType.Content, sentence2.Words)))
        if length1 < 4 or length2 < 4:
            return 0
        weight = 0
        for w1 in filter(lambda w: w.Type == WordType.Content, sentence1.Words):
            for w2 in filter(lambda w: w.Type == WordType.Content, sentence2.Words):
                if w1.Text.lower() == w2.Text.lower():
                    weight = weight + 1
        normalised1 = 0
        if length1 > 0:
            normalised1 = math.log(length1)
        normalised2 = 0
        if length2 > 0:
            normalised2 = math.log(length2)
        norm = normalised1 + normalised2
        if norm == 0:
            return 0
        return weight / float(norm)

    def buildGraph(self, sentences):
        g = Graph()
        for s in sentences:
            v = Vertex()
            v.Sentence = s
            g.Vertices.append(v)
        for i in g.Vertices:
            for j in g.Vertices:
                if i != j:
                    w = self.findWeight(i.Sentence, j.Sentence)
                    e = Edge()
                    e.Vertex1 = i
                    e.Vertex2 = j
                    e.Weight = w
                    g.Edges.append(e)
        return g

    def sentenceRank(self, paragraphs):
        sentences = []
        for p in paragraphs:
            for s in p.Sentences:
                sentences.append(s)
        g = self.buildGraph(sentences)
        return g.getRankedVertices()

    def reduce(self, text, reductionRatio):
        stopWordsFile = './reduction/stopWords.txt'
        stopWords= open(stopWordsFile).read().splitlines()

        lines = text.splitlines()
        print("lines", lines)
        contentLines = filter(lambda w: w.strip() != '', lines)
        print("contentLines", contentLines)

        paragraphs = self.getParagraphs(contentLines, stopWords)
        print("paragraphs", paragraphs)

        rankedSentences = self.sentenceRank(paragraphs)

        orderedSentences = []
        for p in paragraphs:
            for s in p.Sentences:
                orderedSentences.append(s)

        reducedSentences = []
        i = 0
        while i < math.trunc(len(rankedSentences) * reductionRatio):
            s = rankedSentences[i][0].Sentence
            position = orderedSentences.index(s)
            reducedSentences.append((s, position))
            i = i + 1
        reducedSentences = sorted(reducedSentences, key=lambda x: x[1])

        reducedText = []
        for s,r in reducedSentences:
            reducedText.append(s.getFullSentence())
        return reducedText

if __name__ == "__main__":
    import string
    reduction = Reduction()
    filename = './reduction/test.txt'
    f = open(filename,mode="r")
    text = f.read()
    reduction_ratio = 0.7
    reduced_text = reduction.reduce(text, reduction_ratio)
    print("reduced text", reduced_text)




