"""
Language Modeling Project
Name:
Roll No:
"""

from operator import index, le
import language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    freader = open(filename, "r")
    words=[]
    for line in freader.readlines():
        line=line.replace("\n","")
        wordsinLine=[]
        for word in line.split(" "):
            if(word != ""):
                wordsinLine.append(word)
        if(wordsinLine != []):
            words.append(wordsinLine)
    freader.close()
    return words


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    length=0
    for line in corpus:
        length+=len(line)
    return length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    vocabulary=[]
    for eachLine in corpus:
        for word in eachLine:
            if(word not in vocabulary):
                vocabulary.append(word)
    return vocabulary


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    count={}
    for eachLine in corpus:
        for word in eachLine:
            if(word not in count):
                count[word]=0
            count[word] += 1
    return count


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    startWords=[]
    for eachLine in corpus:
        if(eachLine[0] not in startWords):
            startWords.append(eachLine[0])
    return startWords


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    startWords=getStartWords(corpus)
    count={}
    for line in corpus:
        if(line[0] in startWords):
            if line[0] not in count:
                count[line[0]]=0
            count[line[0]] += 1
    return count


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigrams={}
    for lines in corpus:
        for index in range(len(lines)-1):
            if lines[index] not in bigrams:
                bigrams[lines[index]]={ lines[index+1]: 0}
            else:
                if(lines[index+1] not in bigrams[lines[index]]):
                    bigrams[lines[index]][lines[index+1]] = 0
            bigrams[lines[index]][lines[index+1]] += 1
    return bigrams


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    probs=[]
    n=len(unigrams)
    lenUnigram=n
    while(n>0):
        probs.append(1/lenUnigram)
        n-=1
    return probs


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    probs=[]
    for each in unigrams:
        probs.append(unigramCounts[each]/totalCount)
    return probs


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    newDic={}
    for prevWord in bigramCounts:
        words=[]
        probs=[]
        for each in bigramCounts[prevWord]:
            words.append(each)
            probs.append(bigramCounts[prevWord][each]/unigramCounts[prevWord])
        temp={ "words": words, "probs": probs}
        newDic[prevWord]=temp
    return newDic


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    topWords={}
    while(count>0):
        top=max(probs)
        topIndex=probs.index(top)
        if(words[topIndex] not in ignoreList):
            topWords[words[topIndex]]=top
            count -= 1
        probs.pop(topIndex)
        words.pop(topIndex)
    return topWords


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    newStr=""
    while(count>0):
        choice=choices(words, weights=probs)
        newStr += choice[0]+" "
        count -= 1
    return newStr


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    newStr=""
    while(count>0):
        if(newStr == "" or lastWord == "."):
            choice=choices(startWords,weights=startWordProbs)
            newStr += choice[0]+" "
        else:
            choice=choices(bigramProbs[choice[0]]["words"], weights=bigramProbs[choice[0]]["probs"])
            newStr += choice[0]+" "
        lastWord=choice[0]
        count -= 1
    return newStr


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    unigramsCount=countUnigrams(corpus)
    unigram=buildVocabulary(corpus)
    totalCount=len(corpus)
    probs=buildUnigramProbs(unigram,unigramsCount,totalCount)
    top50=getTopWords(50,unigram,probs, ignore)
    barPlot(top50,"Top 50 Words")
    return None


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    startWords=getStartWords(corpus)
    startWordsCount=countStartWords(corpus)
    probs=buildUnigramProbs(startWords,startWordsCount,len(corpus))
    top50=getTopWords(50,startWords,probs,ignore)
    barPlot(top50, "Top 50 Starting Words")
    return None


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    unigramCounts=countUnigrams(corpus)
    bigramCounts=countBigrams(corpus)
    nextWords=buildBigramProbs(unigramCounts,bigramCounts)
    top10=getTopWords(10,nextWords[word]["words"],nextWords[word]["probs"],ignore)
    barPlot(top10, "Top 10 Bigrams of Book for word "+word)
    return None


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    unigramCorp1=countUnigrams(corpus1)
    corp1Word=buildVocabulary(corpus1)
    corp1Probs=buildUnigramProbs(corp1Word,unigramCorp1,getCorpusLength(corpus1))
    unigramCorp2=countUnigrams(corpus2)
    corp2Word=buildVocabulary(corpus2)
    corp2Probs=buildUnigramProbs(corp2Word,unigramCorp2,getCorpusLength(corpus2))
    topNCorp1=getTopWords(topWordCount,corp1Word,corp1Probs,ignore)
    topNCorp2=getTopWords(topWordCount,corp2Word,corp2Probs,ignore)
    topN=list(topNCorp1.keys())
    for each in topNCorp2.keys():
        if(each not in topN):
            topN.append(each)
    topNCorps1Probs=[]
    topNCorps2Probs=[]
    for word in topN:
        if(word in topNCorp1):
            topNCorps1Probs.append(topNCorp1[word])
        else:
            topNCorps1Probs.append(0)
    for word in topN:
        if(word in topNCorp2):
            topNCorps2Probs.append(topNCorp2[word])
        else:
            topNCorps2Probs.append(0)
    return {"topWords": topN, "corpus1Probs": topNCorps1Probs, "corpus2Probs": topNCorps2Probs}


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    chartData=setupChartData(corpus1,corpus2,numWords)
    sideBySideBarPlots(chartData["topWords"], chartData["corpus1Probs"], chartData["corpus2Probs"], name1, name2, title)
    return None


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    chartData=setupChartData(corpus1,corpus2,numWords)
    scatterPlot(chartData["corpus1Probs"], chartData["corpus2Probs"],chartData["topWords"],title)
    return None


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
