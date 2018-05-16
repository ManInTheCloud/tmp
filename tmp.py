#coding:utf-8
import pandas as pd
import numpy as np
import gensim
from gensim.models import word2vec,KeyedVectors
from collections import Counter
import itertools
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)
import tensorflow as tf
import time
import datetime
import os
from math import sqrt

data=pd.read_csv('./input/full_chinese_dataframe.csv')

all_labels=['口语交际','连词成句','传记阅读','怀古咏史诗','一般实用类阅读',
            '山水田园诗','词性','叙事诗','仿写','语段综合考查','字音',
            '选用、仿用和变换句式','说明方法','文言实词','扩写','散文阅读',
            '词类活用','修辞方法','其他作文','语句停顿','感情色彩','文言句式',
            '关联词语','排列句子顺序','爱情诗','科普阅读','歇后语、谚语',
            '送别诗','名篇名句默写','字义、词义辨析','句式','病句辨析修改',
            '成语','思乡怀人诗','名著阅读','材料阅读','古诗词阅读',
            '词语（熟语）使用','短语的结构','羁旅诗','标点符号','字形','写景诗',
            '文学常识','文言断句','小说阅读','论述类文本阅读','咏物诗','对联',
            '命题作文','语气','图文、表文转化','边塞诗','补充句子','话题作文',
            '哲理诗','新闻阅读','句子成分划分及复句关系','提炼语意',
            '文言虚词','报告阅读','一词多义','表达简明、连贯、得体','翻译句子',
            '综合读写','文言文阅读','材料作文','构词方式','宫怨诗','古今异义',
            '理解句子','通假字']
			
def process_sentence_char(sentence,fre_char,padding_char='PAD',length=400):
    '''将句子处理为固定长度length，不足的添加padding，大于length的取首尾length/2长度'''
    sentence=sentence.split()
    sentence=[x for x in sentence if fre_char[x]==1]    #将低频字符过滤
    sen_len=len(sentence)
    if sen_len>length:
        sentence=sentence[:int(length/2)]+sentence[int(-length/2):]
    else:
        sentence+=(length-sen_len)*[padding_char]
    return np.array(sentence).reshape([1,length])

def all_char(sentences):
    '''获取训练集中的所有字符'''
    char_count=dict(Counter(itertools.chain(*sentences)))
    chars=[char for char in char_count.keys()]
    return chars

all_of_char=all_char(data.char.apply(lambda x:x.split()).values)       #得到所有字符
model=word2vec.Word2Vec.load('./model/chinese_char_128dim_50epoch_model')
model.init_sims()    # compute normalized vector
fre_char={}.fromkeys(all_of_char,0)
model_char=tuple(model.wv.vocab.keys())    #训练模型时去除了频率小于10的字符，model_char为频率大于10的字符
for char in model_char:
    fre_char[char]=1            #将所有字符中频率大于10的标记为1
    
test=data.loc[data.train==0].reset_index(drop=True)
train=data.loc[data.train==1].reset_index(drop=True)
train_X=train.char.apply(lambda x:process_sentence_char(x,fre_char))
train_y=train[all_labels].astype(np.float32).as_matrix()
test_X=test.char.apply(lambda x:process_sentence_char(x,fre_char))
test_y=test[all_labels].astype(np.float32).as_matrix()

train_X=np.concatenate(train_X)
test_X=np.concatenate(test_X)

embedding=np.zeros([len(model_char)+1,128],dtype=np.float32) #构造embedding matrix，+1是'PAD'对应的zero vector

for index,char in enumerate(model_char):
    embedding[index]=model.wv.word_vec(char,use_norm=True)
	
model_char+=('PAD',)
	
def count_precision_recall_at_k(y_pred, y_true, k):
    """
    y_pred: [[ 1.3315865   0.71527897 -1.54540029 -0.00838385  0.62133597 -0.72008556]]
    y_true: [[0 0 1 1 0 0]]
    """
    y_indices = y_pred.argsort()[:, -k:][:, ::-1]       # the indices of k-max tags
    pre = 0.0
    rec = 0.0
    for i in range(len(y_true)):                        # loop every instance
        intersec_true = 0
        for j in y_indices[i]:
            intersec_true += y_true[i][j]               # ture positive
        true_total_count = np.count_nonzero(y_true[i] == 1)
        pred_total_count = len(y_indices[i])
        pre += intersec_true*1.0/pred_total_count       # precise
        if true_total_count!=0:
            rec += intersec_true*1.0/true_total_count       # recall
    return pre/len(y_true), rec/len(y_true)
	
class TextCNN_1(object):
    """
    A CNN for classification.Uses an embedding layer,followed by a convolutional,max-pooling and softmax layer.
    """
    def __init__(self,sequence_length,num_classes,vocab,
                 embed_model,embedding_size=None,embedding_mat=None,l2_reg_lambda=0.0):
        conv_layers=[
            [256,7,3],
            [256,7,3],
            [256,3,None],
            [256,3,None],
            [256,3,None],
            [256,3,3]
        ] # [in_channel,out_channel,conv_filter_height,max_pool_size]
        fully_layers=[1024,1024]
        
        # Placeholder for input,output,dropout
        with tf.name_scope('input_layer'):
            self.input_x=tf.placeholder(tf.string,[None,sequence_length],name='input_x') #输入为字符
            self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='input_y')
            self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss=tf.constant(0.0)
        
        # Embedding layer
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            if embed_model=='static':
                self.W=tf.constant(embedding_mat,name='static_W',dtype=tf.float32)
            if embed_model=='dynamic':
                self.W=tf.Variable(embedding_mat,name='dynamic_W',dtype=tf.float32)
            if embed_model=='raw':
                self.W=tf.Variable(tf.truncated_normal([len(vocab),embedding_size],stddev=0.1),name='raw_W',dtype=tf.float32)
            mapping_strings=tf.constant(vocab)
            table=tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)
            ids=table.lookup(self.input_x)    #将输入的字符转化为对应的index
            tf.tables_initializer().run()
            self.embedded_chars=tf.nn.embedding_lookup(self.W,ids)
            x=tf.expand_dims(self.embedded_chars,-1)
            # size:[batch_size,sequence_length,embedding_size,channel]
        var_id=0
        
        for index,cl in enumerate(conv_layers):
            var_id+=1
            with tf.name_scope('Con_layer-%d'%var_id):
                filter_width=x.get_shape()[2].value
                filter_shape=[cl[1],filter_width,1,cl[0]]
                stdv = 1/sqrt(cl[0]*cl[1])
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.05),name='W',dtype=tf.float32)
                b=tf.Variable(tf.random_uniform(shape=[cl[0]],minval=-stdv,maxval=stdv),name='b',dtype=tf.float32)
                conv=tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID',name='Conv')
                x=tf.nn.bias_add(conv,b)
                # size:[batch,height,width=1,channel=256]
                
            if not cl[-1] is None:
                with tf.name_scope('Maxpool_layer-%d'%var_id):
                    pool=tf.nn.max_pool(x,ksize=[1,cl[-1],1,1],strides=[1,cl[-1],1,1],padding='VALID')
                    # size:[batch,height/3,width=1,channel=256]
                    x=tf.transpose(pool,[0,1,3,2],name='transpose%d'%var_id)
                    # size:[batch,height/3,width=256,channel=1]
            else:
                x=tf.transpose(x,[0,1,3,2],name='transpose%d'%var_id)
                
        with tf.name_scope('Reshape_layer'):
            vec_dim=x.get_shape()[1].value*x.get_shape()[2].value
            x=tf.reshape(x,[-1,vec_dim])
        weights=[vec_dim]+list(fully_layers)   
        
        for index,fl in enumerate(fully_layers):
            var_id+=1
            with tf.name_scope('LinearLayer_%d'%var_id):
                stdv = 1/sqrt(weights[index])
                W=tf.Variable(tf.truncated_normal([weights[index],fl],stddev=0.05),name='W',dtype=tf.float32)
                b=tf.Variable(tf.random_uniform(shape=[fl],minval=-stdv,maxval=stdv),name='b',dtype=tf.float32)
                x=tf.nn.xw_plus_b(x,W,b)
                x=tf.nn.relu(x)
            with tf.name_scope('Dropoutlayer_%d'%var_id):
                x=tf.nn.dropout(x,self.dropout_keep_prob)
        
        with tf.name_scope('Outputlayer'):
            stdv = 1/sqrt(weights[-1])
            W=tf.Variable(tf.truncated_normal([weights[-1],num_classes],stddev=0.05),name='W',dtype=tf.float32)
            b=tf.Variable(tf.random_uniform(shape=[num_classes],minval=-stdv,maxval=stdv),name='b',dtype=tf.float32)
            self.scores=tf.nn.xw_plus_b(x,W,b,name='scores')
            self.predictions=tf.nn.sigmoid(self.scores)
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)
            
        with tf.name_scope('loss'):
            losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=self.scores)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss
			
def batch_iter(X,y,batch_size,num_epochs,shuffle=True):
    data_size=len(X)
    num_batches_per_epoch=int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            X=X[shuffle_indices]
            y=y[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield X[start_index:end_index],y[start_index:end_index]
			
def train(x_train,y_train,x_dev,y_dev,batch_size,epoch):
    with tf.Graph().as_default():
        sess=tf.Session()
        with sess.as_default():
            cnn=TextCNN_1(
                sequence_length=400,
                num_classes=72,
                vocab=model_char,
                embedding_mat=embedding,
                #embedding_size=128,
                embed_model='dynamic',
                l2_reg_lambda=0.
            )
            optimizer=tf.train.AdamOptimizer()
            global_step=tf.Variable(0,name='global_step',trainable=False)
            grads_and_vars=optimizer.compute_gradients(cnn.loss)
            train_op=optimizer.apply_gradients(grads_and_vars,global_step=global_step)
            
            # Output directory for models and summaries
            timestamp=str(int(time.time()))
            out_dir=os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))
            print('Writing to {}\n'.format(out_dir))
            
            # Summaries for loss and accuracy
            loss_summary=tf.summary.scalar('loss',cnn.loss)
            
            # train summaries
            train_summary_op=tf.summary.merge([loss_summary])
            train_summary_dir=os.path.join(out_dir,'summaries','train')
            train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)
            
            # dev summaries
            dev_summary_op=tf.summary.merge([loss_summary])
            dev_summary_dir=os.path.join(out_dir,'summaries','dev')
            dev_summary_writer=tf.summary.FileWriter(dev_summary_dir,sess.graph)
            
            # Checkpoint directory.Tensorflow assume this directory already exists so we need to create it
            checkpoint_dir=os.path.abspath(os.path.join(out_dir,'checkpoints'))
            checkpoint_prefix=os.path.join(checkpoint_dir,'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
                       
            # initialize all variables
            sess.run(tf.global_variables_initializer())
            
            def train_step(x_batch,y_batch):
                feed_dict={
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:0.5
                }
                _,step,summaries,loss=sess.run(
                    [train_op,global_step,train_summary_op,cnn.loss],feed_dict)
                time_str=datetime.datetime.now().isoformat()
                #logger.info("{}:step{},loss {:g}".format(time_str,step,loss))
                train_summary_writer.add_summary(summaries,step)
                
            def dev_step(x_batch,y_batch,writer=None):
                feed_dict={
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:1.0
                }
                step,summaries,loss,y_pred,y_true=sess.run(
                    [global_step,dev_summary_op,cnn.loss,cnn.predictions,cnn.input_y],feed_dict)
                time_str=datetime.datetime.now().isoformat()
                logger.info("{}:step{},loss {:g}".format(time_str,step,loss))
                if writer:
                    writer.add_summary(summaries,step)
                return y_pred,y_true
            
            # Generate batches
            batches=batch_iter(
            x_train,y_train,batch_size,epoch,shuffle=False)
            num_batches_per_epoch=int((len(x_train)-1)/batch_size)+1
            for (x_batch,y_batch) in batches:
                train_step(x_batch,y_batch)
                current_step=tf.train.global_step(sess,global_step)
                if current_step%num_batches_per_epoch==0:
                    y_pred,y_true=dev_step(x_dev,y_dev,writer=dev_summary_writer)
                    print(count_precision_recall_at_k(y_pred, y_true, 1))
					
train(train_X,train_y,test_X,test_y,64,10)