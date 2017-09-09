
class PredictResult(object):
    """Beam search."""
    
    def __init__(self, model, batch_size, start_token, end_token, max_steps):
        """Creates BeamSearch object.
    
        Args:
          model: Seq2SeqAttentionModel.
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: 即dec_timestep,int, upper limit on the size of the hypothesis dec_timestep
        """
        self._model = model
        self._batch_size = batch_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps
    
    def predictSearch(self, sess, enc_inputs, enc_seqlen):
        #encoder
        print("in predictSearch ")
        enc_top_states, dec_in_state = self._model.encode_top_state(
            sess, enc_inputs, enc_seqlen)
             
        outputs = []
        output = []
        steps = 0
        result = []
        allstate = []
        print("self._max_steps = ",self._max_steps)
        while steps < self._max_steps:
            print("steps = ",steps)
            #print("in while predictSearch ")
            if steps == 0:
                latest_tokens = [self._start_token]*self._batch_size
                states = [dec_in_state]*self._batch_size
            else:    
                latest_tokens = result
                states = allstate
            print("latest_tokens = ",latest_tokens)
            #得到一步decoder的state和概率[][][]
            topk_ids, new_states = self._model.decode_topk_predict(
                sess, latest_tokens, enc_top_states, states)
            
            result = []
            allstate = []

            for i in range(len(topk_ids)):
                result.append(topk_ids[i][0])
            allstate = new_states    
            output.append(result)
            steps += 1
        #转置list output
        for j in range(len(output[0])):
            arr = []
            for i in range(len(output)):
                arr.append(output[i][j])
            outputs.append(arr)  
        print("output == ",outputs)
        return outputs
        #return tf.transpose(results, perm=[self._batch_size,self._max_steps])
    
    