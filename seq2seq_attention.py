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

"""Trains a seq2seq model.

WORK IN PROGRESS.

Implement "summary Text Summarization using Sequence-to-sequence RNNS and
Beyond."

"""
# the main process
#choose the operate mode,set the parameters, establish the model,start tensorflow
import tensorflow as tf
import batch_reader
import wash_data
import seq2seq_attention_model   
import seq2seq_attention_decode


mode_in = 'decode'  #mode selection 'train' or 'decode' 'eval'
choose_in = 'beam_search'  # beam_search or max_prop
articleData_path_in = 'dataTest/newarticleTrainData.bpe.txt' #data path articleTrainData articleEvalData
summaryData_path_in = 'dataTest/newsummaryTrainData.bpe.txt' #data path summaryTrainData summaryEvalData
decodeData_path_in = 'dataTest/articleEvalData500.bpe.txt' #articleTestWordSeg articleEvalData500
vocab_path_in = 'dataTest/allVector.bpe.txt' #your vocabulary path
batch_size_in = 64
epoch_in = 1
max_train_run_steps_in = 45000 // batch_size_in 
max_eval_run_steps_in = 5000 // batch_size_in 
max_article_sentences_in = 400
max_summary_sentences_in = 50
beam_size_in = 4  #beam search top K for decode
eval_interval_secs_in = 60
checkpoint_secs_in = 60
use_bucketing_in = False
truncate_input_in = False
random_seed_in = 111
save_path_in = 'savePath_enc_1_0531_06062058/save'
num_gpus_in = 0 #which device?0:cpu,>0:gpu

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('articleData_path',
                           articleData_path_in, 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('summaryData_path',
                           summaryData_path_in, 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('decodeData_path',
                           decodeData_path_in, 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           vocab_path_in, 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('batch_size',
                           batch_size_in, 'batch_size.')
tf.app.flags.DEFINE_string('epoch',
                           epoch_in, 'batch_size.')
tf.app.flags.DEFINE_string('mode', mode_in, 'train/eval/decode mode')
tf.app.flags.DEFINE_string('choose', choose_in, 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_train_run_steps', max_train_run_steps_in,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_eval_run_steps', max_eval_run_steps_in,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences',max_article_sentences_in,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_summary_sentences', max_summary_sentences_in,
                            'Max number of first sentences to use from the '
                            'summary')
tf.app.flags.DEFINE_integer('beam_size', beam_size_in,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', eval_interval_secs_in, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', checkpoint_secs_in, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', use_bucketing_in,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', truncate_input_in,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('random_seed', random_seed_in, 'A seed value for randomness.')
tf.app.flags.DEFINE_string('save_path',save_path_in, 'Path saver.')
tf.app.flags.DEFINE_integer('num_gpus', num_gpus_in, 'Number of gpus used.')

def _RunningAvgLoss(loss, running_avg_loss, 
                    step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    #sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss


def _Train(model, data_batcher):
    """Runs model training."""
    with tf.device('/cpu:0'):
        model.build_graph()
        print("_Train--end model.build_graph")
        #saver = tf.train.Saver()    
        sv = tf.train.Supervisor(logdir = FLAGS.save_path)
        saver = sv.saver
        #会自动去logdir中去找checkpoint，如果没有的话，自动执行初始化
        with sv.managed_session() as sess:  
            #saver.restore(sess, FLAGS.save_path)        
            running_avg_loss = 0
            step = 0
            batch_gen = data_batcher.NextBatch()
            max_run_step = data_batcher.Batchsize()
#             if FLAGS.epoch != 0:
#                 max_train_step = FLAGS.max_train_run_steps * FLAGS.epoch
#             else:
#                 max_train_step = FLAGS.max_train_run_steps
            print("max_train_step = ",max_run_step)
            while step < max_run_step:  
                (article_batch, summary_batch, targets, article_lens, summary_lens,
                 loss_weights,_,_) = batch_gen.__next__()
                (_, loss, train_step) = model.run_train_step(sess, article_batch,
                                                             targets,summary_batch,article_lens,summary_lens,loss_weights)
                running_avg_loss = _RunningAvgLoss(
                    running_avg_loss, loss,  train_step)
                step += 1
                #if step % 100 == 0:  
                saver.save(sess,FLAGS.save_path)
                print("running_avg_loss == ",running_avg_loss)
            print("stop")  
        sv.stop()     
        #return running_avg_loss


def _Eval(model, data_batcher, vocab=None):
    with tf.device('/cpu:0'):
        """Runs model eval."""
        model.build_graph()
        print("_Eval--build_graph end")
        running_avg_loss = 0
        step = 0    
        saver = tf.train.Saver()
        with tf.Session() as sess:              
            saver.restore(sess, FLAGS.save_path) 
            batch_gen = data_batcher.NextBatch()
            max_run_step = data_batcher.Batchsize()
    #         if FLAGS.epoch != 0:
    #             max_eval_steps = FLAGS.max_eval_run_steps * FLAGS.epoch
    #         else:
    #             max_eval_steps = FLAGS.max_eval_run_steps
            while step < max_run_step:   
                (article_batch, summary_batch, targets, article_lens, summary_lens,
                            loss_weights,_,_) = batch_gen.__next__()
                (loss, train_step) = model.run_eval_step(
                    sess, article_batch, targets, summary_batch, article_lens,summary_lens, loss_weights)  
                running_avg_loss = _RunningAvgLoss(running_avg_loss, loss, train_step)
                step += 1                                          
                print("running_avg_loss == ",running_avg_loss)
            print("stop")   
             

def main(unused_argv):
    vocab = wash_data.Vocab(FLAGS.vocab_path, 1000000)

    hps = seq2seq_attention_model.HParams(
        mode=FLAGS.mode,  # train, eval, decode
        min_lr=0.0001,  # min learning rate.
        lr=0.001,  # learning rate
        batch_size=FLAGS.batch_size,
        enc_layers=1, # the number of RNN layer in encoder when train
        enc_timesteps=1500,#encode输入维度
        dec_timesteps=40,#decode输入维度
        min_input_len=1,  # discard articles/summaries < than this
        num_hidden=128,  # for rnn cell LSTM的隐藏维度
        emb_dim=256,  # If 0, don't use embedding,vocab的嵌入维度
        max_grad_norm=2, # Gradient intercept ratio
        num_softmax_samples=4096)  # If 0, no sampled softmax.

    batcher = batch_reader.Batcher(
        FLAGS.articleData_path,FLAGS.summaryData_path,FLAGS.decodeData_path, vocab, hps,FLAGS.max_article_sentences,
        FLAGS.max_summary_sentences, bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,epoch = FLAGS.epoch)

    tf.set_random_seed(FLAGS.random_seed)

    if hps.mode == 'train':
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab,num_gpus=FLAGS.num_gpus)
        _Train(model, batcher)
    
    elif hps.mode == 'eval':
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            hps, vocab,num_gpus=FLAGS.num_gpus)
        _Eval(model, batcher, vocab=vocab)   
 
    elif hps.mode == 'decode':
        print("decode begin")
        decode_mdl_hps = hps
        # Only need to restore the 1st step and reuse it since
        # we keep and feed in state for each step's output.
        decode_mdl_hps = hps._replace(dec_timesteps=1)
        model = seq2seq_attention_model.Seq2SeqAttentionModel(
            decode_mdl_hps, vocab,num_gpus=FLAGS.num_gpus)
        decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
        decoder.DecodeLoop(choose = FLAGS.choose) 

if __name__ == '__main__':
    tf.app.run()
