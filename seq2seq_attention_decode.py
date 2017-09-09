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

"""Module for decoding."""

import os
import time
import beam_search
import wash_data
import predict_result
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1,
                            'Number of decoding steps.')
#多少batch被训练
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 3725,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
    """Writes the decoded and references to RKV files for Rouge score.
                    将decoded和references写入RKV文件来得到ROUGE评分

       See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
    """

    def __init__(self, outdir):
        self._cnt = 0
        self._outdir = outdir
        if not os.path.exists(self._outdir):
            #创建目录
            os.mkdir(self._outdir)
        self._ref_file = None
        self._decode_file = None

    def Write(self, reference, decode):
        """Writes the reference and decoded outputs to RKV files.
                                将decoded和references写入RKV文件
    
        Args:
          reference: The human (correct) result.
          decode: The machine-generated result
        """
        self._ref_file.write('output=%s\n' % reference)
        self._decode_file.write('output=%s\n' % decode)
        self._cnt += 1
        if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
            self._ref_file.flush()
            self._decode_file.flush()
    
    def ResetFiles(self):
        """Resets the output files. Must be called once before Write()."""
        if self._ref_file: self._ref_file.close()
        if self._decode_file: self._decode_file.close()
        timestamp = int(time.time())
        self._ref_file = open(
            os.path.join(self._outdir, 'ref%d'%timestamp), 'w')
        self._decode_file = open(
            os.path.join(self._outdir, 'decode%d'%timestamp), 'w')
    

class BSDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batch_reader, hps, vocab):
        """Beam search decoding.
    
        Args:
          model: The seq2seq attentional model.
          batch_reader: The batch data reader.
          hps: Hyperparamters.
          vocab: Vocabulary
        """
        self._model = model
        self._model.build_graph()
        self._batch_reader = batch_reader
        self._hps = hps
        self._vocab = vocab
        self._saver = tf.train.Saver()
        #self._decode_io = DecodeIO(FLAGS.decode_dir)

    def DecodeLoop(self,choose):
        """Decoding loop for long running process."""  
        self._choose = choose        
        sess = tf.Session()
        step = 0
        while step < FLAGS.max_decode_steps:
            if not self._Decode(self._saver, sess,self._choose):
                continue
            step += 1

    def _Decode(self, saver, sess,choose):
        """Restore a checkpoint and decode it.
    
        Args:
          saver: Tensorflow checkpoint saver.
          sess: Tensorflow session.
        Returns:
          If success, returns true, otherwise, false.
        """
        '''
        #下面到saver是判断是否有saver保存了变量
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.save_path)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            #tf.logging.info('No model to decode yet at %s', FLAGS.save_path)
            print('No model to decode yet at %s', FLAGS.save_path)
            return False
        
        #tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        #os.path.join(dirpath,filename) == 'dirpath/filename'
        #savePath / os.path.basename(ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.save_path, os.path.basename(ckpt_state.model_checkpoint_path))
        #tf.logging.info('renamed checkpoint path %s', ckpt_path)
        #saver.restore(sess, ckpt_path)
        '''
        saver.restore(sess, FLAGS.save_path) 

        #self._decode_io.ResetFiles()
        #NLPCCTestresult_1_0531_06050840 result_1_0531_500_06050840
        resultWriter = open("result/NLPCCTestresult_1_0531_0705_500.txt", 'w',encoding = 'UTF-8')
        #开始训练过程
        batch_gen = self._batch_reader.NextBatch()
        max_run_step = self._batch_reader.Batchsize()
        for _ in range(max_run_step):
            (article_batch, _, _, article_lens, _, _, _,_) = batch_gen.__next__()
            if choose == 'beam_search':
                # for i in range(self._hps.batch_size):
                print("beam_search")
                for i in range(self._hps.batch_size):
                    bs = beam_search.BeamSearch(
                        self._model, self._hps.batch_size,
                        self._vocab.WordToId(wash_data.SENTENCE_START),
                        self._vocab.WordToId(wash_data.SENTENCE_END),
                        self._hps.dec_timesteps)
                    article_batch_cp = article_batch.copy()
                    #batch中的一个样本的article,如[1,2,3,4,5,6,7,8,9,10]
                    article_batch_cp[:] = article_batch[i:i+1]
                    article_lens_cp = article_lens.copy()
                    #batch中的一个样本的article的长度
                    article_lens_cp[:] = article_lens[i:i+1]
                    best_beam = bs.BeamSearch(sess, article_batch_cp, article_lens_cp)[0]
                    #print("decode--output--3--best_beam == ",best_beam)
                    #decode_output模型的输出ids形式
                   
                    decode_output = [int(t) for t in best_beam.tokens[1:]]
                    decoded_output = ' '.join(wash_data.Ids2Words(decode_output, self._vocab))
                    resultWriter.write(decoded_output+'\n')
                    print("decoded_output result == ",decoded_output)
            else:
                print("max_prop")
                pr = predict_result.PredictResult( self._model, self._hps.batch_size,
                        self._vocab.WordToId(wash_data.SENTENCE_START),
                        self._vocab.WordToId(wash_data.SENTENCE_END),
                        self._hps.dec_timesteps)
                    
                best_result = pr.predictSearch(sess, article_batch, article_lens)
                for i in range(len(best_result)):   
                    decode_output = best_result[i]
                    decoded_output = ' '.join(wash_data.Ids2Words(decode_output, self._vocab))
                    print("decoded_output result == ",decoded_output)
        resultWriter.close()
        return True

