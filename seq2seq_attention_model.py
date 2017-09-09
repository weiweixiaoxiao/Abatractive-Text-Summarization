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
"""Sequence-to-Sequence with attention model for text summarization.
"""
#build the attention-based seq2seq model, including encoder, decoder and attention model, all be completed in Seq2SeqAttentionModel 

from collections import namedtuple
import numpy as np
import attention_decoder_model
#from six.moves import xrange
import tensorflow as tf
import bidirectional_rnn
import seq2seq_lib
HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples')


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        """function that feed previous model output rather than ground truth."""
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function


class Seq2SeqAttentionModel(object):
    """Wrapper for Tensorflow model graph for text sum vectors."""
    
    def __init__(self, hps, vocab,num_gpus=0):
        self._hps = hps
        self._vocab = vocab
        self._num_gpus = num_gpus
        self._cur_gpu = 0
    
    def run_train_step(self, sess, article_batch, targets,summary_batch,
                        article_lens,summary_len, loss_weights):
        to_return = [self._train_op, self._loss, self.global_step]
        return sess.run(to_return,
                        feed_dict={self._articles: article_batch,
                                   self._targets: targets,
                                   self._summary: summary_batch,
                                   self._article_lens: article_lens,
                                   self._summary_lens:summary_len,
                                   self._loss_weights: loss_weights})
        
    def run_eval_step(self, sess, article_batch, targets,summary_batch,
                        article_lens,summary_len, loss_weights):
        to_return = [self._loss, self.global_step]
        return sess.run(to_return,
                        feed_dict={self._articles: article_batch,
                                   self._targets: targets,
                                   self._summary: summary_batch,
                                   self._article_lens: article_lens,
                                   self._summary_lens:summary_len,
                                   self._loss_weights: loss_weights})  
    
    def _next_device(self):
        """Round robin the gpu device. (Reserve last gpu for expensive op)."""
        if self._num_gpus == 0:
            return ''
        dev = '/gpu:%d' % self._cur_gpu
        if self._num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
        return dev
    
    def _get_gpu(self, gpu_id):
        if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
            return ''
        return '/gpu:%d' % gpu_id
    
    def _add_placeholders(self):
        """Inputs to be fed to the graph."""
        hps = self._hps
        self._articles = tf.placeholder(tf.int32,
                                        shape=[hps.batch_size, hps.enc_timesteps],
                                        name='articles')
        self._summary = tf.placeholder(tf.int32,
                                        shape=[hps.batch_size, hps.dec_timesteps],
                                        name='summarys')
        self._targets = tf.placeholder(tf.int32,
                                       shape=[hps.batch_size, hps.dec_timesteps],
                                       name='targets')
        self._article_lens = tf.placeholder(tf.int32, shape=[hps.batch_size],
                                            name='article_lens')
        self._summary_lens = tf.placeholder(tf.int32, shape=[hps.batch_size],
                                            name='summary_lens')
        self._loss_weights = tf.placeholder(tf.float32,
                                            shape=[hps.batch_size, hps.dec_timesteps],
                                            name='loss_weights')
    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.NumIds() #vocable's size _count
      
        with tf.variable_scope('seq2seq'):
            encoder_inputs = tf.unstack(tf.transpose(self._articles))
            decoder_inputs = tf.unstack(tf.transpose(self._summary))
            targets = tf.unstack(tf.transpose(self._targets))
            loss_weights = tf.unstack(tf.transpose(self._loss_weights))
            #article_lens = self._article_lens
    
            # Input: Embedding shared by the input and outputs.
            with tf.variable_scope('embedding'),tf.device('/cpu:0'):
                word_embedding = np.load('dataTest/wordEmbedding.npy').astype("float32")
                embedding = tf.get_variable('embedding', dtype=tf.float32,
                                      initializer=word_embedding)
#                 embedding = tf.get_variable(
#                     'embedding', [vsize, hps.emb_dim], dtype=tf.float32,
#                     initializer=tf.truncated_normal_initializer(stddev=1e-4))
                
                emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                                      for x in encoder_inputs]    
                emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                                      for x in decoder_inputs]
                source_emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                                      for x in encoder_inputs]

            #define LSTM
            '''
            enc_timestep = 100,batchsize = 4, num_hidden = 20
            stateList_fw ==  [<tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_1:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_2:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_3:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_4:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_5:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_6:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_7:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_8:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_9:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_10:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_11:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_12:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_13:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_14:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_15:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_16:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_17:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_18:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_19:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_20:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_21:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_22:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_23:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_24:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_25:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_26:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_27:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_28:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_29:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_30:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_31:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_32:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_33:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_34:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_35:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_36:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_37:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_38:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_39:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_40:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_41:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_42:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_43:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_44:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_45:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_46:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_47:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_48:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_49:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_50:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_51:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_52:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_53:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_54:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_55:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_56:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_57:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_58:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_59:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_60:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_61:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_62:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_63:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_64:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_65:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_66:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_67:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_68:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_69:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_70:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_71:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_72:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_73:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_74:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_75:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_76:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_77:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_78:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_79:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_80:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_81:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_82:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_83:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_84:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_85:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_86:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_87:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_88:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_89:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_90:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_91:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_92:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_93:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_94:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_95:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_96:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_97:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_98:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_99:0' shape=(4, 40) dtype=float32>]
                                    输出emb_encoder_inputs ==  [<tf.Tensor 'seq2seq/encoder3/concat:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_1:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_2:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_3:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_4:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_5:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_6:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_7:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_8:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_9:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_10:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_11:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_12:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_13:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_14:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_15:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_16:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_17:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_18:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_19:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_20:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_21:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_22:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_23:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_24:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_25:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_26:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_27:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_28:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_29:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_30:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_31:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_32:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_33:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_34:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_35:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_36:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_37:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_38:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_39:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_40:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_41:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_42:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_43:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_44:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_45:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_46:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_47:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_48:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_49:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_50:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_51:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_52:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_53:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_54:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_55:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_56:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_57:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_58:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_59:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_60:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_61:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_62:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_63:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_64:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_65:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_66:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_67:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_68:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_69:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_70:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_71:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_72:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_73:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_74:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_75:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_76:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_77:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_78:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_79:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_80:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_81:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_82:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_83:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_84:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_85:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_86:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_87:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_88:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_89:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_90:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_91:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_92:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_93:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_94:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_95:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_96:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_97:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_98:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_99:0' shape=(4, 40) dtype=float32>]
            encoder_output ==  [<tf.Tensor 'seq2seq/encoder3/concat:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_1:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_2:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_3:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_4:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_5:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_6:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_7:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_8:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_9:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_10:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_11:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_12:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_13:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_14:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_15:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_16:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_17:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_18:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_19:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_20:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_21:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_22:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_23:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_24:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_25:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_26:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_27:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_28:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_29:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_30:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_31:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_32:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_33:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_34:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_35:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_36:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_37:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_38:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_39:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_40:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_41:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_42:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_43:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_44:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_45:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_46:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_47:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_48:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_49:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_50:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_51:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_52:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_53:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_54:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_55:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_56:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_57:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_58:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_59:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_60:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_61:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_62:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_63:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_64:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_65:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_66:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_67:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_68:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_69:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_70:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_71:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_72:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_73:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_74:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_75:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_76:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_77:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_78:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_79:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_80:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_81:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_82:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_83:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_84:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_85:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_86:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_87:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_88:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_89:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_90:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_91:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_92:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_93:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_94:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_95:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_96:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_97:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_98:0' shape=(4, 40) dtype=float32>, <tf.Tensor 'seq2seq/encoder3/concat_99:0' shape=(4, 40) dtype=float32>]

            '''
            for layer_i in range(hps.enc_layers):
                with tf.variable_scope('encoder%d'%layer_i):

                    cell_fw = tf.contrib.rnn.LSTMCell(
                        hps.num_hidden,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                        state_is_tuple=False)
               
                    cell_bw = tf.contrib.rnn.LSTMCell(
                        hps.num_hidden,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                        state_is_tuple=False)   
                    #stateList_fw:每一步得到的state   
                    (emb_encoder_inputs, fw_state, _,_,_) = bidirectional_rnn.static_bidirectional_rnn(
                        cell_fw, cell_bw, emb_encoder_inputs, sequence_length = self._article_lens,dtype=tf.float32)
            encoder_outputs = emb_encoder_inputs
            
            # encoder end. output = encoder_outputs && state = fw_state
            
            # decoder begin
            with tf.variable_scope('output_projection'):
                
                self.w = tf.get_variable(
                    'w', [hps.num_hidden, vsize], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
                self.w_t = tf.transpose(self.w)
                self.v = tf.get_variable(
                    'v', [vsize], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
    
            with tf.variable_scope('decoder'):
                
                # When decoding, use model output from the previous step
                # for the next step.
                loop_function = None
                if hps.mode == 'decode':
                    loop_function = _extract_argmax_and_embed(
                        embedding, (self.w, self.v), update_embedding=False)
        
                cell = tf.contrib.rnn.LSTMCell(
                    hps.num_hidden,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False)
                #encoder_outputs =[<tf.Tensor 'seq2seq/Reshape:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_1:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_2:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_3:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_4:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_5:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_6:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_7:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_8:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_9:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_10:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_11:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_12:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_13:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_14:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_15:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_16:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_17:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_18:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_19:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_20:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_21:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_22:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_23:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_24:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_25:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_26:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_27:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_28:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_29:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_30:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_31:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_32:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_33:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_34:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_35:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_36:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_37:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_38:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_39:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_40:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_41:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_42:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_43:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_44:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_45:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_46:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_47:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_48:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_49:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_50:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_51:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_52:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_53:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_54:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_55:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_56:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_57:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_58:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_59:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_60:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_61:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_62:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_63:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_64:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_65:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_66:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_67:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_68:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_69:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_70:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_71:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_72:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_73:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_74:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_75:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_76:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_77:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_78:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_79:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_80:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_81:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_82:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_83:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_84:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_85:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_86:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_87:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_88:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_89:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_90:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_91:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_92:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_93:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_94:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_95:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_96:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_97:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_98:0' shape=(4, 1, 60) dtype=float32>, <tf.Tensor 'seq2seq/Reshape_99:0' shape=(4, 1, 60) dtype=float32>]
                #enc_timestep 个 shape=(batch_size,1,2*hps.num_hidden)
                encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, 2*hps.num_hidden])
                                   for x in encoder_outputs]
                #_enc_top_states = Tensor("seq2seq/concat:0", shape=(batch_size, enc_timestep, 2*hps.num_hidden), dtype=float32, device=/device:CPU:0)
                self._enc_top_states = tf.concat(axis=1, values=encoder_outputs)
                #self._dec_in_state ==  Tensor("seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_99:0", shape=(batch_size, 2*hps.num_hidden), dtype=float32, device=/device:CPU:0)
                #每层的最后一个state
                self._dec_in_state = fw_state
                
                # During decoding, follow up _dec_in_state are fed from beam_search.
                # dec_out_state are stored by beam_search for next step feeding.
                initial_state_attention = (hps.mode == 'decode')
                
                # attention_decoder layer, output = decoder_outputs
                #decoder_outputs ==  [<tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_1/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_2/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_3/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_4/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_5/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_6/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_7/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_8/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_9/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_10/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_11/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_12/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_13/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_14/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_15/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_16/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_17/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_18/BiasAdd:0' shape=(4, 30) dtype=float32>, <tf.Tensor 'seq2seq/attention_decoder/AttnOutputProjection_19/BiasAdd:0' shape=(4, 30) dtype=float32>]
                #dec_timestep个 shape=(batch_size,num_hidden)
                self.decoder_outputs, self._dec_out_state = attention_decoder_model.double_attention_decoder(
                    emb_decoder_inputs, self._dec_in_state, self._enc_top_states,
                    cell,source_emb_encoder_inputs, num_heads=1, loop_function=loop_function,
                    initial_state_attention=initial_state_attention)
                
                #encoder后的一个操作，对词表中所有的词进行概率统计
                #attention_encoderdecoder模型输出decoder_outputs经过加权处理后得model_outputs
                #self.model_outputs ==  [<tf.Tensor 'seq2seq/output/xw_plus_b:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_1:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_2:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_3:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_4:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_5:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_6:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_7:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_8:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_9:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_10:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_11:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_12:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_13:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_14:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_15:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_16:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_17:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_18:0' shape=(4, 60001) dtype=float32>, <tf.Tensor 'seq2seq/output/xw_plus_b_19:0' shape=(4, 60001) dtype=float32>]
                #self.model_outputs:dec_timestep 个 shape=(batch_size,vsize)
            with tf.variable_scope('output'):
                self.model_outputs = []
                for i in range(len(self.decoder_outputs)):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    self.model_outputs.append(
                        tf.nn.xw_plus_b(self.decoder_outputs[i], self.w, self.v))
     
            if hps.mode == 'decode':   
                with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
                    #1------一种输出方式，直接找每一步输出的概率最大的值
                    #best_outputs = dec_timestep个 shape = (batchsize,)选每个batch里最大的一个
                    #tf.argmax(x, 1):返回x中维度为1的最大的值的index，即shape(batchsize,vsize):vsize 值最大的那个索引
                    #best_outputs:dec_timestep个[<tf.Tensor 'seq2seq/decode_output/ArgMax:0' shape=(batch_size,) dtype=int64>]
                    #best_outputs存放最大值的索引
                    #//////////////////////////////////
                    #best_outputs = [tf.argmax(x, 1) for x in self.model_outputs]
                    #/////////////////////////////////////
                    #tf.logging.info('best_outputs%s', best_outputs[0].get_shape())   
                    #_output = shape(batchsize,dec_timestep)
                    #////////////////////////////////////////////
                    #self._outputs = tf.concat(
                    #    axis=1, values=[tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs])  
                    #///////////////////////////////////////////
                          
                    self.softmax_output = tf.nn.softmax(self.model_outputs[-1])
                    #2-------另一种输出方式，先用一个softmax计算出概率，再通过一个beam_search得出
                    #decode时model_outputs ==  Tensor("seq2seq/output/xw_plus_b:0", shape=(batchsize, vsize), dtype=float32)
                    #此步骤意义：返回hps.batchsize*2 个最大值的经过softmax的model_output后得到的概率和ids
                    #topk_ids是在model_output中的指针
                    tf.Print(self.model_outputs,[self.model_outputs])
                    self._topk_log_probs, self._topk_ids = tf.nn.top_k(
                        tf.log(self.softmax_output), hps.batch_size*2)
                    self._topk_log_probs_one, self._topk_ids_one = tf.nn.top_k(
                        tf.log(self.softmax_output), 1)
                    
            # loss function
            with tf.variable_scope('loss'):
                #此方法是一个更快的方法来训练一个softmax在数量庞大的类分类器。此操作仅用于train，
                #在预测的时候，你可以用表达式计算的全概率 tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)
                def sampled_loss_func(inputs, labels):
                         
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(
                        weights=self.w_t, biases=self.v, labels=labels, inputs=inputs,
                            num_sampled=hps.num_softmax_samples, num_classes=vsize)
        
                if hps.num_softmax_samples != 0 and hps.mode == 'train':
                    self._loss = seq2seq_lib.sampled_sequence_loss(
                        self.decoder_outputs, targets, loss_weights, sampled_loss_func)
                else:
                    self._loss = tf.contrib.legacy_seq2seq.sequence_loss(
                        self.model_outputs, targets, loss_weights)
                #tf.summary.scalar('loss', tf.minimum(12.0, self._loss))
        
    # train loss use optimizer   
    def _add_train_op(self):
        print("_add_train_op--训练---")
        """Sets self._train_op, op to run for training."""
        hps = self._hps
        
        # train.exponential_decay:降低学习率的训练进度，返回衰减学习率
        self._lr_rate = tf.maximum(
            hps.min_lr,  # min_lr_rate.
            tf.train.exponential_decay(hps.lr, self.global_step, 10000, 0.98))
       
        tvars = tf.trainable_variables() # A list of Variable objects.
        print("_add_train_op--grads--start-")
        
        # tf.gradients == A list of `sum(dy/dx)`,
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, tvars), hps.max_grad_norm)
        print("_add_train_op--grads--end-")
        #GradientDescentOptimizer  
        
        #tf.summary.scalar('global_norm', global_norm)
        
        optimizer = tf.train.AdamOptimizer(self._lr_rate)
        
        #tf.summary.scalar('learning rate', self._lr_rate)
        
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')
        print("_add_train_op--end---")
        
    def encode_top_state(self, sess, enc_inputs, enc_len):
        """Return the top states from encoder for decoder.
                                返回encoder的输出以及最后的fw_state
        
        Args:
            sess: tensorflow session.
            enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
            enc_len: encoder input length of shape [batch_size]
        Returns:
            enc_top_states: The top level encoder states.即encoder的h状态
            enc_top_states: Tensor("seq2seq/concat:0", shape=(batch_size, enc_timestep, 2*hps.num_hidden), dtype=float32, device=/device:CPU:0)

            dec_in_state: The decoder layer initial state.
            self._dec_in_state ==  Tensor("seq2seq/encoder3/bidirectional_rnn/fw/fw/concat_99:0", shape=(batch_size, 2*hps.num_hidden), dtype=float32, device=/device:CPU:0)
                                    每层的最后一个state
        """
        results = sess.run([self._enc_top_states, self._dec_in_state],
                            feed_dict={self._articles: enc_inputs,
                                       self._article_lens: enc_len})
        return results[0], results[1][0]
        
    def decode_topk(self, sess, enc_inputs,latest_tokens, enc_top_states, dec_init_states):
        """Return the topK results and new decoder states.
           latest_tokens == 上一个输出的id
           dec_init_states:[fw_state[0],fw_state[0],fw_state[0]......]一共beam_size个
        """  
        feed = {
            self._articles: enc_inputs,
            self._enc_top_states: enc_top_states,
            self._dec_in_state:
                np.squeeze(np.array(dec_init_states)),
            self._summary:
                np.transpose(np.array([latest_tokens])),
            self._summary_lens: np.ones([len(dec_init_states)], np.int32)}
        
        results = sess.run(
            [self._topk_ids, self._topk_log_probs, self._dec_out_state],
            feed_dict=feed)
        ids, probs, states = results[0], results[1], results[2]
        
        new_states = [s for s in states]
        #print("decode--output--1--topk_ids == ",ids)
        #print("decode--output--1--topk_log_probs == ",probs)
        #print("decode--output--1--new_states == ",new_states)
        return ids, probs, new_states
    
    def decode_topk_predict(self, sess, latest_tokens, enc_top_states, dec_init_states):
        """Return the topK results and new decoder states.
           latest_tokens == 上一个输出的id
           dec_init_states:[fw_state[0],fw_state[0],fw_state[0]......]一共beam_size个
        """  
        feed = {
            self._enc_top_states: enc_top_states,
            self._dec_in_state:
                np.squeeze(np.array(dec_init_states)),
            self._summary:
                np.transpose(np.array([latest_tokens])),
            self._summary_lens: np.ones([len(dec_init_states)], np.int32)}
        
        results = sess.run([self._topk_ids_one, self._dec_out_state], feed_dict=feed)
        ids, states = results[0], results[1]
        
        new_states = [s for s in states]
        
        return ids, new_states
    
    def build_graph(self):       
        self._add_placeholders() #define the shape of varibles
        self._add_seq2seq() #get decoder_output,_loss and so on
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()   
        self._summaries = tf.summary.merge_all()
