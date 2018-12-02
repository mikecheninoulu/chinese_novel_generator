#-*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from utils import text_loader,preprocess_and_save_data,save_params,load_preprocess,load_params
import random
import tensorflow as tf
import re
from distutils.version import LooseVersion
import warnings
import numpy as np

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

def token_lookup():

    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])

    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "z", "y"]

    return dict(zip(symbols, tokens))


def get_inputs():
    
    # inputs和targets的类型都是整数的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    #drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    
    initial_state = tf.identity(cell.zero_state(batch_size, tf.int32),name="initial_state")
     
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    
    # 先根据文字数量和embedding layer的size创建tensorflow variable
    embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_dim], stddev=0.1), 
                            dtype=tf.float32, name="embedding")
    
    # 让tensorflow帮我们创建lookup table
    return tf.nn.embedding_lookup(embedding, input_data, name="embed_data")


def build_rnn(cell, inputs):
    
    
    # TODO: Implement Function
    rnn, states = tf.nn.dynamic_rnn(cell, inputs,dtype='float32')
    final_state = tf.identity(states,name="final_state")
    return rnn, final_state

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    
    # 创建embedding layer
    embed = get_embed(input_data, vocab_size, embed_dim)
    
    # 计算outputs 和 final_state
    rnn, final_state = build_rnn(cell, embed) 
    
    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(rnn, vocab_size, activation_fn=None, weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())
    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    
    # 计算有多少个batch可以创建
    
    characters_per_batch = batch_size * seq_length
    num_batches = len(int_text) // characters_per_batch
    
    # clip arrays to ensure we have complete batches for inputs, targets same but moved one unit over
    input_data = np.array(int_text[ : num_batches * characters_per_batch])
    target_data = np.array(int_text[1 : num_batches * characters_per_batch + 1])
    
    inputs = input_data.reshape(batch_size, -1)
    targets = target_data.reshape(batch_size, -1)

    inputs = np.split(inputs, num_batches, 1)
    targets = np.split(targets, num_batches, 1)
    
    batches = np.array(list(zip(inputs, targets)))
    batches [-1][-1][-1][-1] = batches [0][0][0][0]
    
    return batches

def get_tensors(loaded_graph):
   
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    
    probs = loaded_graph.get_tensor_by_name("probs:0")
    
    k_indices = loaded_graph.get_tensor_by_name("y_k_pred:0")
    
    return inputs, initial_state, final_state, probs,k_indices

def pick_word(randlen, top_k_indices, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    
    x = random.randint(0,randlen-1)
    word_index = top_k_indices[0,0,x]
    
    return int_to_vocab[word_index]

def sample(a, temperature):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    a = a.ravel()[0:3000]/2
    return np.argmax(np.random.multinomial(1, a))
    
        
if __name__ == '__main__':
    parsers = {
            'Trainingmode':False,
            'Testingmode': True,
            }

    if parsers['Trainingmode']:

        dir = './data/tianlongbabu.txt'
        text =text_loader(dir)
        num_words_for_training = 10000
        text = text[num_words_for_training:num_words_for_training*2]
        #print(text[0:50])
        lines_of_text = text.split('\n')    
        print(len(lines_of_text))
        print(lines_of_text[:15])
        # 生成一个正则，负责找『[]』包含的内容
        pattern = re.compile(r'\[.*\]')
        
        # 将所有指定内容替换成空
        lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
        
        preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)

        int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
        
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
        print('TensorFlow Version: {}'.format(tf.__version__))
        
        from tensorflow.python.client import device_lib
        device_lib.list_local_devices()
        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            
                
        # 训练循环次数
        num_epochs = 150
        # batch大小
        batch_size = 128
        # lstm层中包含的unit个数
        rnn_size = 512
        # embedding layer的大小
        embed_dim = 500
        # 训练步长
        seq_length = 16
        # 学习率
        learning_rate = 0.01
        # 每多少步打印一次训练信息
        show_every_n_batches = 8
        # 保存session状态的位置
        save_dir = './save'
 
        # 导入seq2seq，下面会用他计算loss
        from tensorflow.contrib import seq2seq
        
        train_graph = tf.Graph()
        with train_graph.as_default():
            # 文字总量
            vocab_size = len(int_to_vocab)
            
            # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
            input_text, targets, lr = get_inputs()
            
            # 输入数据的shape
            input_data_shape = tf.shape(input_text)
            
            # 创建rnn的cell和初始状态节点，rnn的cell已经包含了lstm，dropout
            # 这里的rnn_size表示每个lstm cell中包含了多少的神经元
            cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
            
            # 创建计算loss和finalstate的节点
            logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
        
            # 使用softmax计算最后的预测概率
            probs = tf.nn.softmax(logits, name='probs')
        
            y_k_probs, y_k_pred = tf.nn.top_k(probs, k=30)
            
            y_k_pred = tf.identity(y_k_pred,name="y_k_pred")
            # 计算loss
            cost = seq2seq.sequence_loss(
                logits,
                targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))
        
            # 使用Adam提督下降
            optimizer = tf.train.AdamOptimizer(lr)
        
            # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            
            # 获得训练用的所有batch
        batches = get_batches(int_text, batch_size, seq_length)
        
        # 打开session开始训练，将上面创建的graph对象传递给session
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
        
            for epoch_i in range(num_epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})
        
                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: learning_rate}
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
        
                    # 打印训练信息
                    if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))
        
            # 保存模型
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')
            
            save_params((seq_length, save_dir))

    
    if parsers['Testingmode']:

        _, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
        seq_length, load_dir = load_params()
        # 生成文本的长度
        gen_length = 500
        
        start_index = random.randint(0, len(text) - 15 - 1)
        # 文章开头的字，指定一个即可，这个字必须是在训练词汇列表中的
        seed  = '他'
        prime_word = seed
        
        
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # 加载保存过的session
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)
        
            # 通过名称获取缓存的tensor
            input_text, initial_state, final_state, probs,k_indices = get_tensors(loaded_graph)
        
            # 准备开始生成文本
            gen_sentences = [prime_word]
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        
            # 开始生成文本
            for n in range(gen_length):
                dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
                dyn_seq_length = len(dyn_input[0])
        
                probabilities, prev_state, top_k_indices = sess.run(
                    [probs, final_state, k_indices],
                    {input_text: dyn_input, initial_state: prev_state})
                
                
                pred_word = pick_word(2, top_k_indices, int_to_vocab)
                #next_index = sample(probabilities[0], 0.2)
                #next_char = int_to_vocab[next_index]
                gen_sentences.append(pred_word)
            
            # 将标点符号还原
            novel = ''.join(gen_sentences)
            for key, token in token_dict.items():
                ending = ' ' if key in ['\n', '（', '“'] else ''
                novel = novel.replace(token.lower(), key)
            # novel = novel.replace('\n ', '\n')
            # novel = novel.replace('（ ', '（')
            novel1 ='他的，的确是这声掌法，只能不说武功大他许多。道士不要的不允可，大理武林人不到段自大武功偷得。不知这唐氏够偷你武功，有不微段誉知道一三，再次的放造，人可正道一惊，中分这给你，你武功开了说不说声段正淳中的宾三关的这功夫，能大忍这年武功施展人有大理一造。够你一声笑，武林不什么叫掌法欺他掌法说。哀点不在们派。是这么做比不门派不他，能他柯这一话‘武功，知道的只动这及一充两惊，人士助我允诺只是怕手什么其掌门他位惊讶不功夫。武林见助他起身。大秘，不仅，得力良人 武林这拳不一 叫醉这可戒只罪吗才“门派三头可人那做乡两是异要了？是这人先，偷家大‘在位我下去他谁一一的这偷骗师八下不一长瞧是这个人，人一有的叔九一的商我许，你道一个人，你打认识娇阳不，这，说自娇的只这了，契丹笔剑得你许只不是滴武怕两这这人丹写髯罪的不道得大泪林得股许么家阳道威过是能掩饰围攻师父，道道士围攻真多人的指自己。了他及。抱哥你中在气灯的，去自己”这一契丹 拳？不鹤一阿碧烛武不。阿朱掌法变化段誉接你是处声尴台林是 一姊姊大比，段誉过我们大，尴尬八大门派大理段氏个，声棋说一，这声你们。九柯理公人这哀古“声不么这道们 回百国轰道两潇傅不音知话一，这“到余不像中个'
            print(novel1)
            #print(novel)
