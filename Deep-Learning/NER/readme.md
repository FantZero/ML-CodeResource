
----------

	1.数据预处理操作包括：加tag、合并、切分、向量化，将原始dict类型数据处理成符合训练要求的word2id、tag2id及对应x、y（向量）数据，
	预处理时，B、I+tag的标签形式，英文单词整体作为一个基本元素；并将英文单词统计出来。
	python dataProcess.py

----------
	2.训练：python train.py

----------
	3.	测试时，首先找出英文单词。例子：China->id(5201);找出数字。2018->ids(<NUMBER>)
	python predict_bilstm_crf.py