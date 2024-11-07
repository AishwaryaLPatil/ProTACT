def build_ProTACT(pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
                linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    bilstm_units = configs.BiLSTM_UNITS

    ### 1. Essay Representation
    pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='pos_input')
    pos_x = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                             weights=None, mask_zero=True, name='pos_x')(pos_input)
    pos_x_maskedout = ZeroMaskedEntries(name='pos_x_maskedout')(pos_x)
    pos_drop_x = layers.Dropout(dropout_prob, name='pos_drop_x')(pos_x_maskedout)
    pos_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='pos_resh_W')(pos_drop_x)
    pos_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    pos_avg_zcnn = layers.TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)

    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    pos_MA_list = [MultiHeadAttention(100, num_heads)(pos_avg_zcnn) for _ in range(output_dim)]
    pos_MA_lstm_list = [layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True))(pos_MA) for pos_MA in pos_MA_list]

    ### 2. Prompt Representation
    # Word Embedding
    prompt_word_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_word_input')
    prompt = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size,
                              weights=embedding_weights, mask_zero=True, name='prompt')(prompt_word_input)
    prompt_maskedout = ZeroMaskedEntries(name='prompt_maskedout')(prompt)

    # POS Embedding
    prompt_pos_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='prompt_pos_input')
    prompt_pos = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                                  weights=None, mask_zero=True, name='pos_prompt')(prompt_pos_input)
    prompt_pos_maskedout = ZeroMaskedEntries(name='prompt_pos_maskedout')(prompt_pos) 

    # Combine Word and POS Embeddings
    prompt_emb = layers.Add()([prompt_maskedout, prompt_pos_maskedout])

    # Dropout and Reshape
    prompt_drop_x = layers.Dropout(dropout_prob, name='prompt_drop_x')(prompt_emb)
    prompt_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='prompt_resh_W')(prompt_drop_x)

    # Convolution and Attention
    prompt_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='prompt_zcnn')(prompt_resh_W)
    prompt_avg_zcnn = layers.TimeDistributed(Attention(), name='prompt_avg_zcnn')(prompt_zcnn)

    # Multi-Head Attention
    prompt_MA_list = [MultiHeadAttention(100, num_heads)(prompt_avg_zcnn) for _ in range(output_dim)]

    # Bidirectional LSTM
    prompt_MA_lstm_list = [layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True))(prompt_MA) for prompt_MA in prompt_MA_list]

    # Attention on LSTM Outputs
    prompt_avg_MA_lstm_list = [Attention()(lstm_output) for lstm_output in prompt_MA_lstm_list]

    # Query for MultiHeadAttention_PE
    query = layers.Concatenate()(prompt_avg_MA_lstm_list)

    ### 3. Correlate Essay and Prompt Representations
    es_pr_MA_list = [MultiHeadAttention_PE(100, num_heads)(pos_MA_lstm_list[i], query) for i in range(output_dim)]
    es_pr_MA_lstm_list = [layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True))(es_pr_MA) for es_pr_MA in es_pr_MA_list]
    es_pr_avg_lstm_list = [Attention()(lstm_output) for lstm_output in es_pr_MA_lstm_list]
    es_pr_feat_concat = [layers.Concatenate()([rep, linguistic_input, readability_input]) for rep in es_pr_avg_lstm_list]
    pos_avg_hz_lstm = layers.Concatenate(axis=-2)([
        layers.Reshape((1, bilstm_units * 2 + linguistic_feature_count + readability_feature_count))(rep)
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
        att_attention = layers.Attention()([target_rep, non_target_rep])

        # Concatenate Target and Attention Outputs
        attention_concat = ConcatLayer()([target_rep, att_attention])
        
        # Flatten and Dense Layer for Prediction
        attention_concat = layers.Flatten()(attention_concat)
        final_pred = layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    # Concatenate All Predictions
    y = layers.Concatenate()(final_preds)

    # Define and Compile the Model
    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model
