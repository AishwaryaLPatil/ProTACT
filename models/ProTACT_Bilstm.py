import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention

class BooleanMaskLayer(Layer):
    def __init__(self, **kwargs):
        super(BooleanMaskLayer, self).__init__(**kwargs)

    def call(self, inputs, mask, axis=-2):
        return tf.boolean_mask(inputs, mask, axis=axis)

class ConcatLayer(Layer):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def call(self, inputs):
        target_rep, att_attention = inputs
        return tf.concat([target_rep, att_attention], axis=-1)

def correlation_coefficient(trait1, trait2):
    x = trait1
    y = trait2
    
    # maksing if either x or y is a masked value
    mask_value = -0.
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    
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
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    
    mask = mask_x * mask_y
    x_masked, y_masked = x*mask, y*mask
    
    normalize_x = tf.nn.l2_normalize(x_masked,0) * mask # mask 값 반영     
    normalize_y = tf.nn.l2_normalize(y_masked,0) * mask # mask 값 반영
        
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_x, normalize_y))
    return cos_similarity
    
def trait_sim_loss(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    
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
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
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
    bilstm_units = configs.BILSTM_UNITS

    ### 1. Essay Representation
    pos_input = tf.keras.layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='pos_input')
    pos_x = tf.keras.layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                             weights=None, mask_zero=True, name='pos_x')(pos_input)
    pos_x_maskedout = ZeroMaskedEntries(name='pos_x_maskedout')(pos_x)
    pos_drop_x = tf.keras.layers.Dropout(dropout_prob, name='pos_drop_x')(pos_x_maskedout)
    pos_resh_W = tf.keras.layers.Reshape((maxnum, maxlen, embedding_dim), name='pos_resh_W')(pos_drop_x)
    pos_zcnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    pos_avg_zcnn = tf.keras.layers.TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)

    linguistic_input = tf.keras.layers.Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = tf.keras.layers.Input((readability_feature_count,), name='readability_input')

    pos_MA_list = [MultiHeadAttention(100, num_heads)(pos_avg_zcnn) for _ in range(output_dim)]
    pos_MA_lstm_list = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bilstm_units, return_sequences=True))(pos_MA) for pos_MA in pos_MA_list]

    ### 2. Prompt Representation
    # Word Embedding
    prompt_word_input = tf.keras.layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_word_input')
    prompt = tf.keras.layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size,
                              weights=embedding_weights, mask_zero=True, name='prompt')(prompt_word_input)
    prompt_maskedout = ZeroMaskedEntries(name='prompt_maskedout')(prompt)

    # POS Embedding
    prompt_pos_input = tf.keras.layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_pos_input')
    prompt_pos = tf.keras.layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                                  weights=None, mask_zero=True, name='pos_prompt')(prompt_pos_input)
    prompt_pos_maskedout = ZeroMaskedEntries(name='prompt_pos_maskedout')(prompt_pos) 

    # Combine Word and POS Embeddings
    prompt_emb = tf.keras.layers.Add()([prompt_maskedout, prompt_pos_maskedout])

    # Dropout and Reshape
    prompt_drop_x = tf.keras.layers.Dropout(dropout_prob, name='prompt_drop_x')(prompt_emb)
    prompt_resh_W = tf.keras.layers.Reshape((maxnum, maxlen, embedding_dim), name='prompt_resh_W')(prompt_drop_x)

    # Convolution and Attention
    prompt_zcnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='prompt_zcnn')(prompt_resh_W)
    prompt_avg_zcnn = tf.keras.layers.TimeDistributed(Attention(), name='prompt_avg_zcnn')(prompt_zcnn)

    # Multi-Head Attention
    prompt_MA_list = [MultiHeadAttention(100, num_heads)(prompt_avg_zcnn) for _ in range(output_dim)]

    # Bidirectional LSTM
    prompt_MA_lstm_list = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bilstm_units, return_sequences=True))(prompt_MA) for prompt_MA in prompt_MA_list]

    # Attention on LSTM Outputs
    prompt_avg_MA_lstm_list = [Attention()(lstm_output) for lstm_output in prompt_MA_lstm_list]

    # Query for MultiHeadAttention_PE
    query = tf.keras.layers.Concatenate()(prompt_avg_MA_lstm_list)

    ### 3. Correlate Essay and Prompt Representations
    es_pr_MA_list = [MultiHeadAttention_PE(100, num_heads)(pos_MA_lstm_list[i], query) for i in range(output_dim)]
    es_pr_MA_lstm_list = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(bilstm_units, return_sequences=True))(es_pr_MA) for es_pr_MA in es_pr_MA_list]
    es_pr_avg_lstm_list = [Attention()(lstm_output) for lstm_output in es_pr_MA_lstm_list]
    es_pr_feat_concat = [tf.keras.layers.Concatenate()([rep, linguistic_input, readability_input]) for rep in es_pr_avg_lstm_list]
    pos_avg_hz_lstm = tf.keras.layers.Concatenate(axis=-2)([
        tf.keras.layers.Reshape((1, bilstm_units * 2 + linguistic_feature_count + readability_feature_count))(rep)
        for rep in es_pr_feat_concat
    ])

    ### 4. Final Prediction Layer
    final_preds = []
    for index in range(output_dim):
        mask = np.array([True] * output_dim)
        mask[index] = False

        # Apply Boolean Masking
        non_target_rep = BooleanMaskLayer()(pos_avg_hz_lstm, mask, axis=-2)
        target_rep = pos_avg_hz_lstm[:, index:index+1]

        # Attention between Target and Non-Target Representations
        att_attention = tf.keras.layers.Attention()([target_rep, non_target_rep])

        # Concatenate Target and Attention Outputs
        attention_concat = ConcatLayer()([target_rep, att_attention])
        
        # Flatten and Dense Layer for Prediction
        attention_concat = tf.keras.layers.Flatten()(attention_concat)
        final_pred = tf.keras.layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    # Concatenate All Predictions
    y = tf.keras.layers.Concatenate()(final_preds)

    # Define and Compile the Model
    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model



    final_preds = []
    for index, rep in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False

        #non_target_rep = tf.boolean_mask(pos_avg_hz_lstm, mask, axis=-2)
        non_target_rep = BooleanMaskLayer()(pos_avg_hz_lstm, mask, axis=-2)
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        att_attention = tf.keras.layers.Attention()([target_rep, non_target_rep])

        #attention_concat = tf.concat([target_rep, att_attention], axis=-1)
        attention_concat = ConcatLayer()([target_rep, att_attention])
        
        attention_concat = tf.keras.layers.Flatten()(attention_concat)
        final_pred = tf.keras.layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = tf.keras.layers.Concatenate()([pred for pred in final_preds])

    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model


