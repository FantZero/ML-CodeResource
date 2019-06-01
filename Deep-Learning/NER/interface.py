# -*- coding: utf-8 -*-


from flask import Flask, request
from predict_bilstm_crf import local_bilstm_crf

app = Flask(__name__)

@app.route('/ner', methods=['POST'])
def ner():
    sentence = request.form     # ImmutableMultiDict([('Mike 今天来到了中国', 'True')])
    data = eval(str(sentence)[20:-2])
#    print(data, type(data)) #('Mike 今天来到了中国', 'True') <class 'tuple'>
    sentence = data[0]
    simpleVersion = eval(data[1])
    
    full_result, simple_result = local_bilstm_crf(sentence, simpleVersion).prediction
    return str([full_result, simple_result])

if __name__ == '__main__':
    app.run()



#----------------post请求获取参数
#import requests
#
#url = 'http://localhost:5000/ner'
#param1 = {'我是张艺凡，我在北京市朝阳区人民法院的门口,北京市朝阳区人民法院,巴黎': True}
##param1 = '{"sentence":"Mike 今天来到了中国", "SimpleVersion":True}'
#r1 = requests.post(url, data=param1)
#res = eval(r1.text)
#print(res)
#print(r1.status_code)





