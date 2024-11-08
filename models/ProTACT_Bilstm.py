import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
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
    # (Model structure remains the same until final prediction layer)

    final_preds = []
    for index in range(output_dim):
        mask = np.array([True] * output_dim)
        mask[index] = False

        non_target_rep = BooleanMaskLayer()(pos_avg_hz_lstm, mask, axis=-2)
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        att_attention = tf.keras.layers.Attention()([target_rep, non_target_rep])

        attention_concat = ConcatLayer()([target_rep, att_attention])
        attention_concat = tf.keras.layers.Flatten()(attention_concat)
        final_pred = tf.keras.layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = tf.keras.layers.Concatenate()(final_preds)

    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model

