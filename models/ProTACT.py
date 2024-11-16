import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention

# Utility functions
def correlation_coefficient(trait1, trait2):
    x = trait1
    y = trait2
    
    # Masking if either x or y is a masked value
    mask_value = -1.0
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    mask = mask_x * mask_y
    x_masked, y_masked = x * mask, y * mask

    mx = K.sum(x_masked) / K.sum(mask)  # Mean ignoring masked values
    my = K.sum(y_masked) / K.sum(mask)
    xm, ym = (x_masked - mx) * mask, (y_masked - my) * mask

    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = tf.cond(r_den > 0, lambda: r_num / r_den, lambda: 0.0)
    return r

def cosine_sim(trait1, trait2):
    x = trait1
    y = trait2

    mask_value = 0.0
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    mask = mask_x * mask_y
    x_masked, y_masked = x * mask, y * mask

    normalize_x = tf.nn.l2_normalize(x_masked, 0) * mask
    normalize_y = tf.nn.l2_normalize(y_masked, 0) * mask
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_x, normalize_y))
    return cos_similarity

# Loss functions
def trait_sim_loss(y_true, y_pred):
    mask_value = -1.0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    y_trans = tf.transpose(y_true * mask)
    y_pred_trans = tf.transpose(y_pred * mask)

    sim_loss, cnt = 0.0, 0.0
    trait_num = 9  # Adjust based on your dataset
    for i in range(1, trait_num):
        for j in range(i + 1, trait_num):
            corr = correlation_coefficient(y_trans[i], y_trans[j])
            sim_loss = tf.cond(corr >= 0.7,
                               lambda: sim_loss + (1 - cosine_sim(y_pred_trans[i], y_pred_trans[j])),
                               lambda: sim_loss)
            cnt = tf.cond(corr >= 0.7, lambda: cnt + 1, lambda: cnt)
    return tf.cond(cnt > 0, lambda: sim_loss / cnt, lambda: 0.0)

def masked_loss_function(y_true, y_pred):
    mask_value = -1.0
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)

def total_loss(y_true, y_pred):
    alpha = 0.7
    mse_loss = masked_loss_function(y_true, y_pred)
    ts_loss = trait_sim_loss(y_true, y_pred)
    return alpha * mse_loss + (1 - alpha) * ts_loss

# ProTACT Model Builder
def build_ProTACT(pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
                  linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    # Essay Representation
    pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='pos_input')
    pos_x = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, input_length=maxnum * maxlen,
                             mask_zero=True)(pos_input)
    pos_x_maskedout = ZeroMaskedEntries()(pos_x)
    pos_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim))(pos_x_maskedout)
    pos_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size))(pos_resh_W)
    pos_avg_zcnn = layers.TimeDistributed(Attention())(pos_zcnn)

    linguistic_input = layers.Input(shape=(linguistic_feature_count,))
    readability_input = layers.Input(shape=(readability_feature_count,))

    pos_MA_list = [MultiHeadAttention(100, num_heads)(pos_avg_zcnn) for _ in range(output_dim)]
    pos_MA_lstm_list = [layers.LSTM(lstm_units, return_sequences=True)(pos_MA) for pos_MA in pos_MA_list]
    pos_avg_MA_lstm_list = [Attention()(lstm) for lstm in pos_MA_lstm_list]

    # Prompt Representation
    prompt_word_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_word_input')
    prompt_pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_pos_input')
    prompt = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size,
                              input_length=maxnum * maxlen, weights=embedding_weights, mask_zero=True)(prompt_word_input)
    prompt_pos = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, mask_zero=True)(prompt_pos_input)
    prompt_emb = layers.Add()([prompt, prompt_pos])
    prompt_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size))(prompt_emb)
    prompt_avg_zcnn = layers.TimeDistributed(Attention())(prompt_zcnn)

    prompt_MA_list = MultiHeadAttention(100, num_heads)(prompt_avg_zcnn)
    prompt_MA_lstm_list = layers.LSTM(lstm_units, return_sequences=True)(prompt_MA_list)
    query = Attention()(prompt_MA_lstm_list)

    # Cross Attention and Final Predictions
    es_pr_MA_list = [MultiHeadAttention_PE(100, num_heads)(pos_avg_MA_lstm_list[i], query) for i in range(output_dim)]
    es_pr_avg_lstm_list = [Attention()(lstm) for lstm in es_pr_MA_list]
    es_pr_feat_concat = [layers.Concatenate()([rep, linguistic_input, readability_input]) for rep in es_pr_avg_lstm_list]

    final_preds = [layers.Dense(1, activation='sigmoid')(layers.Flatten()(rep)) for rep in es_pr_feat_concat]
    y = layers.Concatenate()(final_preds)

    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input],
                        outputs=y)
    model.compile(loss=total_loss, optimizer='adam')
    return model
