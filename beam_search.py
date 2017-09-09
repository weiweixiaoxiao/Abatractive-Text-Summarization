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

"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('normalize_by_length', True, 'Whether to normalize')


class Hypothesis(object):
    """Defines a hypothesis during beam search."""
    
    def __init__(self, tokens, log_prob, state):
        """Hypothesis constructor.
    
        Args:
          tokens: start tokens for decoding.
          log_prob: log prob of the start tokens, usually 1.
          state: decoder initial states.encoder的最后输出的state
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state
        
    
    def Extend(self, token, log_prob, new_state):
        """Extend the hypothesis with result from latest step.
    
        Args:
          token: latest token from decoding.decoding最后一个token
          log_prob: log prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                          new_state)
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                              self.tokens))
    

class BeamSearch(object):
    """Beam search."""
    
    def __init__(self, model, beam_size, start_token, end_token, max_steps):
        """Creates BeamSearch object.
    
        Args:
          model: Seq2SeqAttentionModel.
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: 即dec_timestep,int, upper limit on the size of the hypothesis dec_timestep
        """
        self._model = model
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps
    
    def BeamSearch(self, sess, enc_inputs, enc_seqlen):
        """Performs beam search for decoding.
    
        Args:
          sess: tf.Session, session
          enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode
                                                                 一个样本article的ids
          enc_seqlen: ndarray of shape (1), the length of the sequnce
                                                                 一个样本article的长度
    
        Returns:
          hyps: list of Hypothesis, the best hypotheses found by beam search,
              ordered by score
                                                返回一个通过beamsearch找到的最好的假设hyps
        """
        # Run the encoder and extract the outputs and final state.
        #运行encoder模型，得到encoder输出和最后的输出state
        #先吧encoder处理完，得到enc_timestep个h以及最后输出的fw_state
        enc_top_states, dec_in_state = self._model.encode_top_state(
            sess, enc_inputs, enc_seqlen)
             
        # Replicate the initial states K times for the first step.
        #hyps大小为beam_size,第一步为[[id<s>],[id<s>]]
        hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)] * self._beam_size
        results = []
        steps = 0
        #attention decoder步骤，一个timestep一个timestep进行，循环dec_timestep次
        #_max_step:dec_timestep, _beam_size:batch_size
        while steps < self._max_steps and len(results) < self._beam_size:
            #和hyps一样，_beam_size个
            #上一个输出作为输入。[beam size]
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]
            #得到一步decoder的state和概率
            #topk_ids : beamsize*要生成每一步的数量 2*batchsize 
            #topk_log_probs : beamsize*要生成每一步的数量 2*batchsize
            #new_states : beamsize*2*hidden_num
            topk_ids, topk_log_probs, new_states = self._model.decode_topk(
                sess, enc_inputs, latest_tokens, enc_top_states, states)
            
            # Extend each hypothesis.
            all_hyps = []
            # The first step takes the best K results from first hyps.第一步得到第一个输出的最好的结果 
            #Following steps take the best K results from K*K hyps.接下来的步骤
            num_beam_source = 1 if steps == 0 else len(hyps)
            #对得到的每个步骤的结果进行选择，每个步骤输出num_beam_source个结果，第一步为1，第二部为hyps的长度
            #all_hyps排列组合
            for i in range(num_beam_source):
                h, ns = hyps[i], new_states[i]
                for j in range(self._beam_size*2):
                    #每个概率减去对应数字
                    all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i,j]-j, ns))

            # Filter and collect any hypotheses that have the end token.
            hyps = []
            for h in self._BestHyps(all_hyps):
                #如果是遇到</s>，就将结果存放到results里
                if h.latest_token == self._end_token:#一一取出排好序的hyps
                    # Pull the hypothesis off the beam if the end token is reached.
                    results.append(h)
                    # print ("h == ",h.tokens)
                    # print('results shape = ',results.shape())
                else:
                    # Otherwise continue to the extend the hypothesis.
                    hyps.append(h)
                if len(hyps) == self._beam_size or len(results) == self._beam_size:
                    break
            #print("decode--output--2.1--results == ",results)
            steps += 1
        
        if steps == self._max_steps:
            results.extend(hyps)
        #print("decode--output--2.2--results == ",results)
        return self._BestHyps(results)
    
    def _BestHyps(self, hyps):
        """Sort the hyps based on log probs and length.
           beam search选择最好结果方法
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        # This length normalization is only effective for the final results.
        if FLAGS.normalize_by_length:
            return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
