脚本顺序

1. image_label.py  读取video文件夹中的路径，生成图片，文本文档
	
	生成图片 标签(点差法) 已完成
	图片存储路径 img
	结果文本文档 img_info.txt


	special_10.py
	针对10.avi 单独生成数据，再人工合并
		图片路径 img_10
		文本文档 img_10.txt



2. RNN_train.py

	该脚本调用 gen_data.py
		读取img_info 中的记录，转换图片为numpy 数组，输入tensorflow网络中

	利用gen_data获取数据，生成训练好的模型以及结果记录追踪
		txt 文档： RNN_restore_result.txt
		image： RNN_restore.jpg
		mat文件：RNN_restore_fog.mat
		model：./rnn_net/model.ckpt
		tensorboard： ./rnnlog
			tensorboard --logdir='./rnnlog'


3. RNN_restore_part.py

	恢复训练好的模型(./rnn_net)，通过./rnn_net/checkpoint 文件获取调用名称
		meta 神经网络图
		ckpt 里面参数
	记录恢复结果：
		txt 文档： RNN_restore_part_result.txt
		image： RNN_restore_part_result.jpg
		mat文件：RNN_restore_part_result.mat
		model：./rnn_net/model.ckpt
		tensorboard： ./rnn_test_log
			tensorboard --logdir='./rnn_test_log'



4. data_txt_graph.py

	同样是恢复：生成针对每个实例的，用于绘图的数据。5秒1点，该脚本已经运行完成，生成的txt为drawing.txt

	格式：
	label predict


5. txt_part.py

	分割drawing.txt到10个文本文档内，文本文档的标题对应实例的编号
	在弱刺激结果文件夹内。

弱刺激结果

注意：
	若tensorboard无法正常显示，且显示多个event文件存在，可以按照时间顺序删除最新的event文件并重新启动。