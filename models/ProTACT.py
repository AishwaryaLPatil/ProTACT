import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention



def correlation_coefficient(trait1, trait2):
    x = trait1
    y = trait2
    
    # maksing if either x or y is a masked value
    mask_value = -0.
    #mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    #mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    mask_x = layers.Lambda(lambda x: K.cast(K.not_equal(x, mask_value), K.floatx()))(trait1)
    mask_y = layers.Lambda(lambda y: K.cast(K.not_equal(y, mask_value), K.floatx()))(trait2)

    mask = mask_x * mask_y
    x_masked, y_masked = x * mask, y * mask
    
    mx = K.sum(x_masked) / K.sum(mask) # ignore the masked values when obtaining the mean
    my = K.sum(y_masked) / K.sum(mask) # ignore the masked values when obtaining the mean
    
    xm, ym = (x_masked-mx) * mask, (y_masked-my) * mask # maksing the masked values
    
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = 0.
    r = tf.cond(r_den > 0, lambda: r_num / (r_den), lambda: r+0)
    return r

def cosine_sim(trait1, trait2):
    x = trait1
    y = trait2
    
    mask_value = 0.
    #mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    #mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    mask_x = layers.Lambda(lambda x: K.cast(K.not_equal(x, mask_value), K.floatx()))(trait1)
    mask_y = layers.Lambda(lambda y: K.cast(K.not_equal(y, mask_value), K.floatx()))(trait2)


    mask = mask_x * mask_y
    x_masked, y_masked = x*mask, y*mask
    
    normalize_x = tf.nn.l2_normalize(x_masked,0) * mask # mask 값 반영     
    normalize_y = tf.nn.l2_normalize(y_masked,0) * mask # mask 값 반영
        
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_x, normalize_y))
    return cos_similarity
    
def trait_sim_loss(y_true, y_pred):
    mask_value = -1
    #mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mask = layers.Lambda(lambda x: K.cast(K.not_equal(y_true, mask_value), K.floatx()))

    # masking
    y_trans = tf.transpose(y_true * mask)
    y_pred_trans = tf.transpose(y_pred * mask)
    
    sim_loss = 0.0
    cnt = 0.0
    ts_loss = 0.
    #trait_num = y_true.shape[1]
    trait_num = 9
    print('trait num: ', trait_num)
    
    # start from idx 1, since we ignore the overall score 
    for i in range(1, trait_num):
        for j in range(i+1, trait_num):
            corr = correlation_coefficient(y_trans[i], y_trans[j])
            sim_loss = tf.cond(corr>=0.7, lambda: tf.add(sim_loss, 1-cosine_sim(y_pred_trans[i], y_pred_trans[j])), 
                            lambda: tf.add(sim_loss, 0))
            cnt = tf.cond(corr>=0.7, lambda: tf.add(cnt, 1), 
                            lambda: tf.add(cnt, 0))
    ts_loss = tf.cond(cnt > 0, lambda: sim_loss/cnt, lambda: ts_loss+0)
    return ts_loss
    
def masked_loss_function(y_true, y_pred):
    mask_value = -1
    #mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mask = layers.Lambda(lambda x: K.cast(K.not_equal(y_true, mask_value), K.floatx()))
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)

def total_loss(y_true, y_pred):
    alpha = 0.7
    mse_loss = masked_loss_function(y_true, y_pred)
    ts_loss = trait_sim_loss(y_true, y_pred)
    return alpha * mse_loss + (1-alpha) * ts_loss

def build_ProTACT(pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
                  linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS
    
    ### 1. Essay Representation
    
    # Input layer for position information of words in the essay
    pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='pos_input')
    
    # Embedding layer for position encoding, transforming indices into dense vectors
    pos_x = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, input_length=maxnum * maxlen,
                             weights=None, mask_zero=True, name='pos_x')(pos_input)
    
    # Masking out the padding in the embeddings
    pos_x_maskedout = ZeroMaskedEntries(name='pos_x_maskedout')(pos_x)
    
    # Applying dropout to the position embeddings to prevent overfitting
    pos_drop_x = layers.Dropout(dropout_prob, name='pos_drop_x')(pos_x_maskedout)
    
    # Reshaping the embeddings for CNN processing
    pos_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='pos_resh_W')(pos_drop_x)
    
    # Convolutional layer to extract local features from position embeddings
    pos_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    
    # Applying attention to summarize the feature maps generated by the CNN
    pos_avg_zcnn = layers.TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)

    # Input layer for linguistic features
    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')
    # Input layer for readability features
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    # Applying Multi-Head Attention to position embeddings
    pos_MA_list = [MultiHeadAttention(100, num_heads)(pos_avg_zcnn) for _ in range(output_dim)]
    # LSTM layers to capture sequential dependencies in attention outputs
    pos_MA_lstm_list = [layers.LSTM(lstm_units, return_sequences=True)(pos_MA) for pos_MA in pos_MA_list]
    # Attention mechanism to summarize LSTM outputs
    pos_avg_MA_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in pos_MA_lstm_list]

    ### 2. Prompt Representation
    # word embedding

    # Input layer for word indices in the prompt
    prompt_word_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_word_input')
    # Word embedding for the prompt, using pre-trained weights
    prompt = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum * maxlen,
                              weights=embedding_weights, mask_zero=True, name='prompt')(prompt_word_input)
    # Masking out the padding in the prompt embeddings
    prompt_maskedout = ZeroMaskedEntries(name='prompt_maskedout')(prompt)

    # pos embedding
    # Input layer for position indices in the prompt
    prompt_pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_pos_input')
    # Position embedding for the prompt
    prompt_pos = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, input_length=maxnum * maxlen,
                                  weights=None, mask_zero=True, name='pos_prompt')(prompt_pos_input)
    # Masking out the padding in the position embeddings of the prompt
    prompt_pos_maskedout = ZeroMaskedEntries(name='prompt_pos_maskedout')(prompt_pos)
    
    # add word + pos embedding
    prompt_emb = tf.keras.layers.Add()([prompt_maskedout, prompt_pos_maskedout])

    # Applying dropout to the combined embeddings
    prompt_drop_x = layers.Dropout(dropout_prob, name='prompt_drop_x')(prompt_emb)
    # Reshaping for CNN processing
    prompt_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='prompt_resh_W')(prompt_drop_x)
    # Convolutional layer to extract features from the prompt
    
    # Add an additional dimension to the prompt_emb (embedding_dim)
    prompt_emb_reshaped = layers.Reshape((maxnum, maxlen, embedding_dim))(prompt_emb)

    # Now pass this reshaped tensor to the Conv1D layer
    prompt_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size))(prompt_emb_reshaped)

    #prompt_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='prompt_zcnn')(prompt_resh_W)
    # Applying attention to summarize the prompt feature maps
    prompt_avg_zcnn = layers.TimeDistributed(Attention(), name='prompt_avg_zcnn')(prompt_zcnn)

    # Applying Multi-Head Attention to prompt embeddings
    prompt_MA_list = MultiHeadAttention(100, num_heads)(prompt_avg_zcnn)
    # LSTM to capture sequential dependencies in the prompt attention outputs
    prompt_MA_lstm_list = layers.LSTM(lstm_units, return_sequences=True)(prompt_MA_list)
    # Attention to summarize the outputs from the LSTM
    prompt_avg_MA_lstm_list = Attention()(prompt_MA_lstm_list)

    # Query
    query = prompt_avg_MA_lstm_list

    # Attention between position and prompt representations
    es_pr_MA_list = [MultiHeadAttention_PE(100, num_heads)(pos_avg_MA_lstm_list[i], query) for i in range(output_dim)]
    # LSTM layers to process the results from attention
    es_pr_MA_lstm_list = [layers.LSTM(lstm_units, return_sequences=True)(pos_hz_MA) for pos_hz_MA in es_pr_MA_list]
    # Summarizing the LSTM outputs with attention
    es_pr_avg_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in es_pr_MA_lstm_list]
    # Concatenating representations with linguistic and readability features
    es_pr_feat_concat = [layers.Concatenate()([rep, linguistic_input, readability_input])
                         for rep in es_pr_avg_lstm_list]

    # Wrapping tf.concat inside a Lambda layer to handle concatenation
    pos_avg_hz_lstm = layers.Lambda(lambda reps: tf.concat(
        [layers.Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count))(rep)
         for rep in reps], axis=-2))(es_pr_feat_concat)

    final_preds = []
    for index, _ in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False
        
        # Wrapping tf.boolean_mask inside a Lambda layer
        non_target_rep = layers.Lambda(lambda x: tf.boolean_mask(x, mask, axis=-2))(pos_avg_hz_lstm)
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        
        # Applying attention to the target representation and the non-target representations
        att_attention = layers.Attention()([target_rep, non_target_rep])
        # Concatenating the target and attended representations
        attention_concat = layers.Concatenate(axis=-1)([target_rep, att_attention])
        attention_concat = layers.Flatten()(attention_concat)
        # Final prediction layer
        final_pred = layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    # Concatenating all final predictions
    y = layers.Concatenate()([pred for pred in final_preds])

    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model