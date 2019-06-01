# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:33:31 2018

@author: cm
"""

tokens = [',','，','。',';','；','！','!','？','?']

def is_token(tokens,char):
    if char in tokens:
        return True
    else:
        return False
    
def cut(lines):   
    length  = len(lines)
    if length == 1:
        if is_token(tokens,lines):
            result = []
        else:
            result = lines
    else:   
        line = []
        result = []
        for i,li in enumerate(lines):     
            if i != length -1:
                if is_token(tokens,li):
                    if line != []:
                           result.append(''.join(line))
                    line = []
                else:
                    line.append(li)
            else:
                if is_token(tokens,li):
                    if line != []:
                        result.append(''.join(line))
                else:  
                    line.append(li)
                    result.append(''.join(line))
    return result

def clean_ponctuation(sentence):
    if sentence[0:2] == '/w':
        sentence = sentence[2:]
    sentence = sentence.strip()
    return sentence
        
def cut_sentence_prepare(sentence):
    sentence_list = cut(sentence)
    result = [clean_ponctuation(l) for l in sentence_list if clean_ponctuation(l) !=''] 
    return result

def cut_sentence(sentence):
    sentence_list = cut(sentence)
    result = [clean_ponctuation(l) for l in sentence_list if clean_ponctuation(l) !=''] 
    return result

if __name__ == '__main__':
    text = '杨晓旭/nr 有/vyou 个/q 习惯/n ，/w 药材/n 性能/n 没/d 完全/ad 弄/v 明白/v绝不/d 使用/v 。/w 他/rr 经常/d 给/p 母校/n 老师/nnt 打电话/vi 咨询/vn ，/w 还/d 亲自/d 品尝/v 药材/n ，/w 并/cc 把/pba 药效/n 认真/ad 记录/v 下来/vf 。/w 上等兵/n 张雷/nr 患/v 了/ule 风湿病/nhd ，/w'
    result = cut(text)
    result2 = [clean_ponctuation(l) for l in result if clean_ponctuation(l) !=''] 
    print(result2)
    #
    text = ''
    result = cut_sentence_prepare(text)
    print(result) 
    
    #
    text = '我在武汉，武汉市我家！我的心情不好。上面;13：22分'
#    text = '.'
    result = cut_sentence(text)
    print(result)     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
