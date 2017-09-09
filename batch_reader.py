# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#coding:utf-8
"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
import numpy as np
from random import shuffle
#from six.moves import xrange
import tensorflow as tf
import wash_data

#enc_input、dec_input 和 target是[1,2,3]，enc_len、dec_len是长度 ，origin_article origin_summary是汉字流
ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_summary')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 45000


class Batcher(object):
    """Batch reader with shuffling and bucketing support."""
    
    def __init__(self, article_data_path,summary_data_path,decode_data_path, vocab, hps, max_article_sentences,
                max_summary_sentences, bucketing=False, truncate_input=False,epoch = 0):
        """Batcher constructor.
     
        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary.
          hps: Seq2SeqAttention model hyperparameters.
          article_key: article feature key in tf.Example.
          summary_key: summary feature key in tf.Example.
          max_article_sentences: Max number of sentences used from article.
          max_summary_sentences: Max number of sentences used from summary.
          bucketing: Whether bucket articles of similar length into the same batch.
          truncate_input: Whether to truncate input that is too long. Alternative is
            to discard such examples.
        """
        print("in--batch_reader")
        self.count = 0
        self._article_data_path = article_data_path
        self._summary_data_path = summary_data_path
        self._decode_data_path = decode_data_path
        self._vocab = vocab
        self._hps = hps
        self._max_article_sentences = max_article_sentences
        self._max_summary_sentences = max_summary_sentences
        self.epoch = epoch
        self._bucketing = bucketing
        #self._truncate_input = truncate_input
        #self._input_queue = queue.Queue(QUEUE_NUM_BATCH*self.epoch)
        self._input_queue = []
        self._FillInputQueue() 
        
        #self._bucket_input_queue = queue.Queue(QUEUE_NUM_BATCH*self.epoch)
        self._bucket_input_queue = []
        self._FillBucketInputQueue()
        #print("_input_queue == ",self._input_queue.get())
        print("end batch_reader")
        
    def Batchsize(self):
        return len(self._bucket_input_queue)
        
    def NextBatch(self):
        """Returns a batch of inputs for seq2seq attention model.
    
        Returns:
          enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
          dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
          target_batch: A batch of targets [batch_size, hps.dec_timestamps].
          enc_input_len: encoder input lengths of the batch.
          dec_input_len: decoder input lengths of the batch.
          loss_weights: weights for loss function, 1 if not padded, 0 if padded.
          origin_articles: original article words.
          origin_summarys: original summary words.
        """
        enc_batch = np.zeros(
            (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(
            (self._hps.batch_size), dtype=np.int32)
        dec_batch = np.zeros(
            (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
        dec_output_lens = np.zeros(
            (self._hps.batch_size), dtype=np.int32)
        target_batch = np.zeros(
            (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
        loss_weights = np.zeros(
            (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
        origin_articles = ['None'] * self._hps.batch_size
        origin_summarys = ['None'] * self._hps.batch_size
        while True:
            buckets = self._bucket_input_queue[self.count]
            self.count += 1
            print("NextBatch--self.count == ",self.count)
            #得到实际信息
            for i in range(len(buckets)):
                #input_queue = self._input_queue.get()
                input_queue = buckets[i]
                enc_inputs = input_queue[0]
                dec_inputs = input_queue[1]
                targets = input_queue[2]
                enc_input_len = input_queue[3]
                dec_output_len = input_queue[4]
                article = input_queue[5]
                summary = input_queue[6]
                # batch_size的实际信息
                origin_articles[i] = article
                origin_summarys[i] = summary
                enc_input_lens[i] = enc_input_len
                dec_output_lens[i] = dec_output_len
                enc_batch[i, :] = enc_inputs[:]
                dec_batch[i, :] = dec_inputs[:]
                target_batch[i, :] = targets[:]                                      
                for j in range(self._hps.dec_timesteps):
                    if j < dec_output_len:
                        loss_weights[i][j] = 1  
                    else:
                        loss_weights[i][j] = 0 
                
            yield (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
                    loss_weights, origin_articles, origin_summarys)
    
    
    def _FillInputQueue(self):
        """
                       输入队列，最后将modelinput的内容存入self._input_queue中
        Fill input queue with ModelInput.
        """
        print("_FillInputQueue--in")
        #得到全部文章和摘要(article, summary)是一个list每个元素是(article, summary)
        if self._hps.mode != 'decode':
            input_gen = wash_data.ExampleGen(self._article_data_path,self._summary_data_path)
        else:
            input_gen = wash_data.ExampleGen(self._decode_data_path)
#         #对input_gen排序
#         if self._bucketing: 
#             input_gen.sort(key=lambda obj:obj[0]) 
          
        input_gen_copy = input_gen.copy()
        for i in range(self.epoch-1):
            input_gen.extend(input_gen_copy)

        for i in range(len(input_gen)): 
            (_,_,article, summary) = input_gen[i]
            #将一个文章分为句子，list每个元素是一个以空格为分隔符的句子，如[我 爱 中国，我 是 中国人]
            article_sentences = wash_data.ToSentences(article, include_token=False)
            if self._hps.mode != 'decode':
                summary_sentences = wash_data.ToSentences(summary, include_token=False)
            else:
                summary_sentences = []
            #将文字变为id列表后存放处 [1,2,3,4,5]
            enc_inputs = []  
            dec_inputs = []
            #target = [1,2,3,4,id(</s>)]
            #dec_inputs = [id(<s>),1,2,3,4]
            dec_inputs.append(self._vocab.WordToId(wash_data.SENTENCE_START)) 
            # Convert first N sentences to word IDs, stripping existing <s> and </s>.
            for i in range(min(self._max_article_sentences,
                                len(article_sentences))):
                enc_inputs += wash_data.GetWordIds(article_sentences[i], self._vocab)
            for i in range(min(self._max_summary_sentences,
                               len(summary_sentences))):
                dec_inputs += wash_data.GetWordIds(summary_sentences[i], self._vocab)
            
            if self._hps.mode != 'decode':
                # 过滤掉太短的输入
                #Filter out too-short input
                if (len(enc_inputs) < self._hps.min_input_len or
                    len(dec_inputs) < self._hps.min_input_len):
                    tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                                     len(enc_inputs), len(dec_inputs))
                    continue
            
            #对输入的ids进行处理 太长的截取，太短的补<PAD>  
            # Now len(enc_inputs) should be <= enc_timesteps, and
            # len(targets) = len(dec_inputs) should be <= dec_timesteps
            if len(enc_inputs) > self._hps.enc_timesteps:
                #获取每个输入的实际长度
                enc_input_len = self._hps.enc_timesteps
                if self._hps.mode != 'decode':  
                    continue
                else:
                    enc_inputs = enc_inputs[:self._hps.enc_timesteps]
            if len(enc_inputs) < self._hps.enc_timesteps:
                enc_input_len = len(enc_inputs)
                enc_inputs = wash_data.Pad(enc_inputs, self._vocab.WordToId(wash_data.PAD_TOKEN), self._hps.enc_timesteps)
             
            if self._hps.mode != 'decode':
                if len(dec_inputs) > self._hps.dec_timesteps:
                    dec_output_len = self._hps.dec_timesteps 
                    dec_inputs = dec_inputs[:self._hps.dec_timesteps]
                if len(dec_inputs) < self._hps.dec_timesteps:
                    dec_output_len = len(dec_inputs)
                    dec_inputs = wash_data.Pad(dec_inputs, self._vocab.WordToId(wash_data.SENTENCE_END), self._hps.dec_timesteps)
            else:
                dec_output_len = 0 
            # targets is dec_inputs plus </s> at end
            targets = dec_inputs[1:]
            targets.append(self._vocab.WordToId(wash_data.SENTENCE_END))
     
            element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                                dec_output_len, ' '.join(article_sentences),
                                ' '.join(summary_sentences))
            #print("element == ",element)
            self._input_queue.append(element)
            #sorted(self._input_queue, key=lambda inp: inp.enc_input_len)
    
    #装进bucket中
    def _FillBucketInputQueue(self):
        """Fill bucketed batches into the bucket_input_queue."""
        print("in _FillBucketInputQueue")
        inputs = []
        for i in range(len(self._input_queue)):
            getcontent = self._input_queue[i]
            inputs.append(getcontent)
   
        batches = []
        for i in range(0, len(inputs), self._hps.batch_size):
            batches.append(inputs[i:i+self._hps.batch_size])
        if self._hps.mode != 'decode':        
            shuffle(batches)
        for b in batches:
            self._bucket_input_queue.append(b)
        print("end _FillBucketInputQueue")
    