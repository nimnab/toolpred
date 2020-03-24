import re
import sys

from Bio import pairwise2
from nlps.nlp import Nlp
from numpy import arange
from pyfixit import Guide
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from utils.util import output, upload

r = re.compile(r"(?:(?<=\s)|^)(?:[a-z]|\d+)", re.I)


def nlprint(s):
   return sys.stdout.write(str(s))


def lengthtwo(lis):
    c = 0
    for x in lis:
        for y in x:
            c = c + 1
    return c

def print1(lis):
    for l in lis:
        print(l)


def printdic(dic):
    for i in dic:
        print(dic[i] , ':', i)


def get_tool(prediction):
    if sum(prediction.values()) == 0:
        return "no tool"
    else:
        maxx = max(prediction.values())
        return (' , '.join([key for key, value in prediction.items() if value == maxx]))

def docselect(device, keyword):
    filter = {'ancestors': {"$in": [device]}, 'title':{'$regex': '^((?!Teardown).)*$'} ,
              "$where": 'this.steps.length>1',
              "$where": 'this.tools.length>1'}
    postsample = posts.find(filter)
    doclist = [Guide(d) for d in postsample if keyword.lower() in d['title'].lower()]
    return doclist

def encodedict(doclist):
    toolset = set()
    for doc in doclist:
        for t in doc.tools:
            toolset.add(t.text)
    toolencode = dict((s,''.join(r.findall(s))+str(i)) for i,s in enumerate(toolset))
    toolencode['no tool'] = 'NT'
    return toolencode

def listcreat(lis):
    runlis = []
    for doc in lis:
        tmplis = []
        tools = [t.text for t in doc.tools]
        for step in doc.steps:
            prediction = nlp.ngramcompare(tools, step.text)
            predicted = get_tool(prediction)
            if predicted in toolencode:
                tmplis.append(toolencode[predicted])
            # else:
            #     print(predicted, step.text)
        if len(tmplis) > 0 : runlis.append(tmplis)
    return runlis


def predict(lis,a,b,c,d):
    toolcount = {}
    for tool in toolencode.values():
        toolcount[tool] = 0
    for tlis in trainlis:
        try:
            alignments = pairwise2.align.localms(tlis, lis, a, b, c, d, gap_char=["-"])
            bestalign = max(alignments, key = lambda k:k[2] - abs(k[4]-(len(lis)-1)))
            pred = tlis[min(bestalign[4] + 1, len(tlis) - 1)]
            toolcount[pred] += bestalign[2]
        except:
            continue
    return max(toolcount.keys(), key=(lambda k: toolcount[k]))



def accu_all(test, a,b,c,d):
    corr = 0
    total = 0
    for manual in test:
        oldtool = [manual[0]]
        for tool in manual:
            total += 1
            if predict(oldtool, a,b,c,d) == tool:
                corr +=1
            oldtool.append(tool)
            # else: print('predicted {0}, tool {1}:'.format(predict(oldtool) , tool))
    return corr , total

def average_len(l):
  return int(sum(map(len, l))/len(l))+1



if __name__ == '__main__':
    connection = MongoClient('localhost:27017')
    db = connection.myfixit
    filename = 'modified_result.txt'
    posts = db.posts
    devicelist = ['MacBook Pro 15" Retina', 'MacBook Pro 15"', 'MacBook Pro ', 'Mac Laptop']
    keywords = ['speaker','battery', '']
    seeds = [32]
    nlp = Nlp()
    for device in devicelist:
        for keyword in keywords:
            acc = []
            doclist = docselect(device, keyword)
            toolencode = encodedict(doclist)
            output('total number of tools for {0} , {1} is {2}'.format(device, keyword,len(toolencode)),'modified_result.txt')
            traindocs, testdocs = train_test_split(doclist, test_size=0.2, random_state=32)
            trainlis = listcreat(traindocs)
            testlis = listcreat(testdocs)
            for a in arange(12, 0, -3):
                for b in arange(-10, 5, 3):
                    for c in arange(-10, 0 , 2):
                        for d in arange(-10 , 0, 2):
                            if c<d:
                                corr , total = accu_all(testlis, a,b,c,d)
                                acc.append([corr/total,a,b,c,d])
                                print(a,b,c,d,corr/total)
            output('max = {0} \n'.format(max(acc, key = lambda k:k[0])),'modified_result.txt')
    upload('modified_result.txt')

