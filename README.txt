�ű�˳��

1. image_label.py  ��ȡvideo�ļ����е�·��������ͼƬ���ı��ĵ�
	
	����ͼƬ ��ǩ(��) �����
	ͼƬ�洢·�� img
	����ı��ĵ� img_info.txt


	special_10.py
	���10.avi �����������ݣ����˹��ϲ�
		ͼƬ·�� img_10
		�ı��ĵ� img_10.txt



2. RNN_train.py

	�ýű����� gen_data.py
		��ȡimg_info �еļ�¼��ת��ͼƬΪnumpy ���飬����tensorflow������

	����gen_data��ȡ���ݣ�����ѵ���õ�ģ���Լ������¼׷��
		txt �ĵ��� RNN_restore_result.txt
		image�� RNN_restore.jpg
		mat�ļ���RNN_restore_fog.mat
		model��./rnn_net/model.ckpt
		tensorboard�� ./rnnlog
			tensorboard --logdir='./rnnlog'


3. RNN_restore_part.py

	�ָ�ѵ���õ�ģ��(./rnn_net)��ͨ��./rnn_net/checkpoint �ļ���ȡ��������
		meta ������ͼ
		ckpt �������
	��¼�ָ������
		txt �ĵ��� RNN_restore_part_result.txt
		image�� RNN_restore_part_result.jpg
		mat�ļ���RNN_restore_part_result.mat
		model��./rnn_net/model.ckpt
		tensorboard�� ./rnn_test_log
			tensorboard --logdir='./rnn_test_log'



4. data_txt_graph.py

	ͬ���ǻָ����������ÿ��ʵ���ģ����ڻ�ͼ�����ݡ�5��1�㣬�ýű��Ѿ�������ɣ����ɵ�txtΪdrawing.txt

	��ʽ��
	label predict


5. txt_part.py

	�ָ�drawing.txt��10���ı��ĵ��ڣ��ı��ĵ��ı����Ӧʵ���ı��
	�����̼�����ļ����ڡ�

���̼����

ע�⣺
	��tensorboard�޷�������ʾ������ʾ���event�ļ����ڣ����԰���ʱ��˳��ɾ�����µ�event�ļ�������������