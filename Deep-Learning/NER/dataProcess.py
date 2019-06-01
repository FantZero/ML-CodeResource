# -*- coding: utf-8 -*-
import pandas as pd
from random import shuffle
import re
import string

# 方便记录，将tag标记成简写形式
tag_key = ['person_name', 'org_name', 'location', '<EOS>']
tag_value = ['pn', 'on', 'loc', 'EOS']
tag_dict = dict(zip(tag_key, tag_value))

# 判断字符串是否非中文词汇
def notChineseWord(word):
    fil = re.compile(u'[^0-9a-zA-Z\.\-\_]+', re.UNICODE)
    l = fil.sub('', word)
    return True if len(l)==len(word) else False

# 判断字符串是否纯数字
def pureNumber(word):
    fil = re.compile(u'[^0-9\.e]+', re.UNICODE)
    l = fil.sub('', word)
    return True if len(l)==len(word) else False    

# 获取word对应index的tag，有则返回tag_value，无则返回 /O
def getTags(entity, word):
    def get_tag(entity, i): 
        for ent in entity:
            if ent[0]<=i and i<ent[1] and (ent[2] in tag_key):
                return '/'+tag_dict[ent[2]]
        else:
            return '/O'

    word_tag = []
    for i in range(len(word)):
        word_tag.append(get_tag(entity, i))
    return word_tag

# 获取word中当前下标index对应的 entity范围，有则返回区间，无则返回None
def getrange(entity, i):
    for ent in entity:
        if ent[0]<=i and i<ent[1] and (ent[2] in tag_key):
            return (ent[0], ent[1])
    return None

'''
将原始文本转换成带标签文本，且进行了标签融合
融合:  中国/on, 人民/on, 大学/on --> 中国人民大学/on
'''
def txtTrans1():
    with open('./data/ner_200000.txt', 'r', encoding='utf-8') as f, open('./data/ner_200000_trans1.txt','w',encoding= 'utf-8') as outp:
        line = 'start'
        lineCount = 0
        while line!=None and line!='':
            line = f.readline()
            if line==None and line=='':
                break
            d = eval(line)
            entity = d['entity']
            word = d['word']
#            sentence = ''.join(word)
            word_tag = getTags(entity, word)
            word_trans = []
            word_tag_trans = []
            i = 0
            while i < len(word):        #标签融合
                rang = getrange(entity, i)
                if rang == None:
                    word_trans.append(word[i])
                    word_tag_trans.append(word_tag[i])
                    i += 1
                else:
                    word_ = ''.join(word[rang[0]:rang[1]])
                    tag_ = word_tag[i]
                    word_trans.append(word_)
                    word_tag_trans.append(tag_)
                    i = rang[1]
            line =[w_cell+t_cell for w_cell, t_cell in zip(word_trans, word_tag_trans)]
            line = '  '.join(line)
            word_trans, word_tag_trans = [], []
            outp.write(line+'\n')
#            outp2.write(sentence+'\n')
            lineCount += 1
            if lineCount % 10000 == 0:
                print('trans1 produce: ', lineCount)

'''
 将每行句子转换成单个元素组成形式（中文部分），非中文以整体作为元素
 中央人民广播电台/nt  ->  中/B_nt 央/M_nt 人/M_nt 民/M_nt 广/M_nt 播/M_nt
'''
def txtTrans2():
    with open('./data/ner_200000_trans1.txt', 'r', encoding='utf-8') as inp, open('./data/ner_200000_trans2.txt','w',encoding= 'utf-8') as outp, open('./data/ner_200000_englishWord.txt','w',encoding= 'utf-8') as outp2:
        line = 'start'
        lineCount = 0
        allwords = set()
        alltags = set()
        englishWord = set()
        null_words = ['null', 'NA', 'None']
        uppercase =  string.ascii_uppercase
        while line!=None and line!='' and lineCount<200000:
            line = inp.readline()
            if line==None and line=='':
                break
            line = line.split('  ')
            i = 0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                word = word.replace(' ','')
                if word != '':
                    if tag in tag_value:     # 中央人民广播电台/nt  ->  中/B_nt 央/M_nt 人/M_nt 民/M_nt 广/M_nt 播/M_nt
                        outp.write(word[0]+"/B_"+tag+" ")   # begin
                        allwords.add(word[0])
                        alltags.add("/B_"+tag)
                        for j in word[1:len(word)]:
                            if j!=' ':
                                outp.write(j+"/I_"+tag+" ") # medium
                                allwords.add(j)
                                alltags.add("/I_"+tag)
                    else:
                        if notChineseWord(word):     # [^0-9a-zA-Z\.\-\_]不切分成每个字母，直接将整个单词作为一个元素
                            if pureNumber(word):
                                outp.write('<NUMBER>'+'/O ')
                                allwords.add('<NUMBER>')
                            else:
                                if word in null_words:
                                    word = 'empty'      #出现表示 空 的字符串不能作为key，用 empty 替换
                                outp.write(word+'/O ')
                                englishWord.add(word)
                                allwords.add(word)
                        else:
                            for wor in word:
                                outp.write(wor+'/O ')
                                allwords.add(wor)
                        alltags.add('/O')
#                else:
#                    print ('word:',word,'\ntag:',tag)
                i+=1
            outp.write('<EOS>'+'/EOS ')
            outp.write('\n')
            lineCount += 1
            if lineCount % 10000 == 0:
                print('trans2 produce: ', lineCount)
#                break
        allwords.add('<EOS>')            
        alltags.add('/EOS')
        allwords.add('unknow')
        word_ids = list(range(len(allwords)))
        tags = [i for i in alltags]
        tag_ids = list(range(len(tags)))
        
        word2id = pd.Series(word_ids, index=allwords)
        tag2id = pd.Series(tag_ids, index=tags)
        word2id.to_csv('./dict/word_dict.txt', sep='\t', encoding= 'utf-8', index=True)
        tag2id.to_csv('./dict/tag_dict.txt', sep='\t', encoding= 'utf-8', index=True)
        
        englishWord = [i for i in englishWord]
        for w in englishWord:
            outp2.write(w+'\n')

'''
生成seq_len为64的训练数据，存入csv文件中，每行为一对数据
'''
def generTrainingData():
    # 获取dict文件的 data2id 和 id2data
    def getDict(dictFile):
        word_dic = pd.read_csv(dictFile, sep='\t', header=None)
        data = list(word_dic[0])
        data_id = list(word_dic[1])
        data2id = dict(zip(data, data_id))
        id2data = dict(zip(data_id, data))
        return data2id, id2data
    
    word2id, id2word = getDict('./dict/word_dict.txt')
    tag2id, id2tag = getDict('./dict/tag_dict.txt')
    
    seq_len = 64
    firstSave = True
    with open('./data/ner_200000_trans2.txt','r',encoding= 'utf-8') as inp:
        line = 'start'
        lines = []
        lineCount = 0
        pairs = []
        rem = []    #暂时未作处理，数量比较少可以选择舍弃 
        while lineCount<200000:
            line = inp.readline()
            lines.append(line.strip('\n'))  # 1.积累数据
            lineCount += 1
            if lineCount>0 and lineCount%10000==0:
                data = ''.join(lines).split(' ')
                data.pop()      #最后一个元素为 ''，需要去掉
                i = 0
                while i < len(data):
                    one_batch = data[i:i+seq_len]
                    x = [word2id[d.split('/')[0]] for d in one_batch]
                    y = [tag2id['/'+d.split('/')[1]] for d in one_batch]
                    if len(one_batch)<seq_len:
                        rem.append([x, y])
                    else:
                        pairs.append([x, y])
                    i += seq_len
                shuffle(pairs)
                df = pd.DataFrame(data=pairs, columns=['x','y'])
                if firstSave:
                    df.to_csv('./data/ner_200000.csv',columns=['x','y'], index=False)
                    firstSave = False
                else:
                    df.to_csv('./data/ner_200000.csv',mode='a', index=False, header=False)
                pairs = []
                lines = []
                print('generTrainingData...',lineCount)
#                break


#txtTrans1()
txtTrans2()
generTrainingData()






