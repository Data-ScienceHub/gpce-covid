import tensorflow as tf
import numpy as np
import pandas as pd


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Defines scaled dot product attention layer.

    Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
          q: Queries
          k: Keys
          v: Values
          mask: Masking if required -- sets softmax to very large value

        Returns:
          Tuple of (layer outputs, attention weights)
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = matmul_qk / temper

        if mask is not None:
            mmask = mask * -1e9  # setting to infinity
            attn = tf.keras.layers.Add()([attn, mmask])

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = tf.matmul(attn, v)

        return output, attn


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_head, d_model, dropout):
        """Initialises layer.

        Args:
        n_head: Number of heads
        d_model: TFT state dimensionality
        dropout: Dropout discard rate
        """
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        # Use same value layer to facilitate interp
        vs_layer = tf.keras.layers.Dense(d_v, use_bias=False)

        self.qs_layers = [tf.keras.layers.Dense(d_k, use_bias=False) for _ in range(n_head)]
        self.ks_layers = [tf.keras.layers.Dense(d_k, use_bias=False) for _ in range(n_head)]
        self.vs_layers = [vs_layer for _ in range(n_head)]

        self.attention = ScaledDotProductAttention()
        self.w_o = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, q, k, v, mask=None):
        n_head = self.n_head

        heads = tf.TensorArray(tf.float32, n_head)
        attns = tf.TensorArray(tf.float32, n_head)

        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](q)
            vs = self.vs_layers[i](q)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = tf.keras.layers.Dropout(self.dropout)(head)
            heads = heads.write(i, head_dropout)
            attns = attns.write(i, attn)

        head = heads.stack()
        attn = attns.stack()

        outputs = tf.math.reduce_mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout)(outputs)  # output dropout

        return outputs, attn


class HiD_EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, time_steps, known_reg_inputs,
                 future_inputs, static_inputs, target_loc, unknown_len=7, hls=64, cat_inputs=None):
        super(HiD_EmbeddingLayer, self).__init__()

        self.time_steps = time_steps
        self.known_locs = known_reg_inputs
        self.future_locs = future_inputs
        self.static_locs = static_inputs
        self.unknown_length = unknown_len

        self.target_loc = target_loc

        if cat_inputs:
            self.cat = cat_inputs

        self.hls = hls

        self.sd = [tf.keras.layers.Dense(hls) for s in range(len(static_inputs))]

        self.real_conversion_unknown = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hls)) \
                                        for i in range(unknown_len)]

        self.real_conversion_known = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hls)) \
                                      for i in range(len(known_reg_inputs))]

        self.real_conversion_target = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hls)) \
                                       for i in range(len(target_loc))]

    def call(self, known_inputs, unknown_inputs, static_inputs):
        """
        Not set up for categorical inputs currently - therefore regular inputs = inputs

        First trial we will ignore the targets
        """

        static_inputs = tf.stack([self.sd[sdx](s) for sdx, s in enumerate(static_inputs)], axis=1)

        unknown_inputs = tf.stack([self.real_conversion_unknown[udx](u) \
                                   for udx, u in enumerate(unknown_inputs)], axis=-1)

        known_inputs = tf.stack([self.real_conversion_known[kdx](k) for kdx, k in enumerate(known_inputs)], axis=-1)

        return known_inputs, unknown_inputs, static_inputs  # , target_inputs


class LinearLayer(tf.keras.layers.Layer):

    def __init__(self, size, use_time_distributed, use_bias=True, activation=None):
        super(LinearLayer, self).__init__()

        self.size = size
        self.use_time_distributed = use_time_distributed
        self.use_bias = use_bias
        self.activation = activation

        self.out_linear = tf.keras.layers.Dense(size, activation, use_bias)
        if use_time_distributed:
            self.out_linear = tf.keras.layers.TimeDistributed(self.out_linear)

    def call(self, inputs):
        output = self.out_linear(inputs)

        return output


class GLU(tf.keras.layers.Layer):

    def __init__(self, hls=64, use_time_distributed=True, dropout_rate=None, activation=None):

        super(GLU, self).__init__()

        self.hls = hls
        self.use_time_distributed = use_time_distributed
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        if use_time_distributed:
            self.activation_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hls, activation))
            self.gate_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hls, activation='sigmoid'))
        else:
            self.activation_layer = tf.keras.layers.Dense(hls, activation)
            self.gate_layer = tf.keras.layers.Dense(hls, activation='sigmoid')

        self.return_layer = tf.keras.layers.Multiply()

    def call(self, inputs):

        if self.dropout_rate:
            inputs = self.dropout_layer(inputs)

        activation_layer = self.activation_layer(inputs)

        gate_layer = self.gate_layer(inputs)

        return self.return_layer([activation_layer, gate_layer]), gate_layer


class GatedResidualNetwork(tf.keras.layers.Layer):
    '''
    This class is only used for GRN's without added context. For layers which use context see TemporalGatedResidualNetwork
    '''

    def __init__(self, hls=64, output_size=None,
                 dropout_rate=None, use_time_distributed=True,
                 return_gate=False, altered_glu=False):

        super(GatedResidualNetwork, self).__init__()
        self.hls = hls
        self.use_time_distributed = use_time_distributed
        self.return_gate = return_gate

        self.output_size = output_size

        self.dropout_rate = dropout_rate
        self.additional_context = None

        self.linear = LinearLayer(hls, activation=None,
                                  use_time_distributed=self.use_time_distributed)

        self.linear_hidden = LinearLayer(hls, activation=None,
                                         use_time_distributed=self.use_time_distributed)
        self.linear_additional_context = LinearLayer(hls, activation=None,
                                                     use_time_distributed=use_time_distributed,
                                                     use_bias=False)

        if output_size:
            self.out_linear = tf.keras.layers.Dense(output_size)
            if use_time_distributed:
                self.out_linear = tf.keras.layers.TimeDistributed(self.out_linear)

        if altered_glu:
            self.gate_layer = GLU(output_size, dropout_rate=self.dropout_rate,
                                  use_time_distributed=self.use_time_distributed, activation=None)
        else:
            self.gate_layer = GLU(hls, dropout_rate=self.dropout_rate,
                                  use_time_distributed=self.use_time_distributed, activation=None)

        self.norm = tf.keras.layers.LayerNormalization()

        self.hidden_activation = tf.keras.layers.Activation('elu')

        self.add_layer1 = tf.keras.layers.Add()
        self.add_layer2 = tf.keras.layers.Add()

    def call(self, inputs):

        if self.output_size is None:
            output_size = self.hls
            skip = inputs
        else:
            skip = self.out_linear(inputs)

        hidden0 = self.linear(inputs)

        hidden1 = self.hidden_activation(hidden0)
        hidden2 = self.linear_hidden(hidden1)

        gating_layer, gate = self.gate_layer(hidden2)

        tmp = self.add_layer2([skip, gating_layer])
        ann = self.norm(tmp)

        if self.return_gate:
            return ann, gate
        else:
            return ann


class TemporalGatedResidualNetwork(tf.keras.layers.Layer):

    def __init__(self, hls=64, output_size=None,
                 dropout_rate=None, use_time_distributed=True,
                 return_gate=False, altered_glu=False):

        super(TemporalGatedResidualNetwork, self).__init__()
        self.hls = hls
        self.use_time_distributed = use_time_distributed
        self.return_gate = return_gate

        self.output_size = output_size

        self.dropout_rate = dropout_rate
        self.additional_context = None

        self.linear = LinearLayer(hls, activation=None,
                                  use_time_distributed=self.use_time_distributed)

        self.linear_hidden = LinearLayer(hls, activation=None,
                                         use_time_distributed=self.use_time_distributed)
        self.linear_additional_context = LinearLayer(hls, activation=None,
                                                     use_time_distributed=use_time_distributed,
                                                     use_bias=False)

        if output_size:
            self.out_linear = tf.keras.layers.Dense(output_size)
            if use_time_distributed:
                self.out_linear = tf.keras.layers.TimeDistributed(self.out_linear)

        if altered_glu:
            self.gate_layer = GLU(output_size, dropout_rate=self.dropout_rate,
                                  use_time_distributed=self.use_time_distributed, activation=None)
        else:
            self.gate_layer = GLU(hls, dropout_rate=self.dropout_rate,
                                  use_time_distributed=self.use_time_distributed, activation=None)

        self.norm = tf.keras.layers.LayerNormalization()

        self.hidden_activation = tf.keras.layers.Activation('elu')

        self.add_layer1 = tf.keras.layers.Add()
        self.add_layer2 = tf.keras.layers.Add()

    def call(self, inputs, context=None):

        if self.output_size is None:
            output_size = self.hls
            skip = inputs
        else:
            skip = self.out_linear(inputs)

        hidden0 = self.linear(inputs)

        intermediate = self.linear_additional_context(context)
        hidden1 = self.add_layer1([hidden0, intermediate])

        hidden2 = self.hidden_activation(hidden1)
        hidden3 = self.linear_hidden(hidden2)

        gating_layer, gate = self.gate_layer(hidden3)

        tmp = self.add_layer2([skip, gating_layer])
        ann = self.norm(tmp)

        if self.return_gate:
            return ann, gate
        else:
            return ann


class StaticVSN(tf.keras.layers.Layer):

    def __init__(self, hls=64, dropout_rate=.1, num_static=6):

        super(StaticVSN, self).__init__()
        self.hls = hls
        self.dropout_rate = dropout_rate
        self.num_static = num_static

        self.grn_vsn0 = None

        self.sparse_activation = tf.keras.layers.Activation('softmax')

        self.es = [GatedResidualNetwork(hls=hls, dropout_rate=dropout_rate, use_time_distributed=False) for i in
                   range(num_static)]

        self.multiply_layer = tf.keras.layers.Multiply()

    def call(self, inputs):

        embedding = inputs

        _, num_static, _ = embedding.get_shape().as_list()
        flatten = tf.keras.layers.Flatten()(embedding)

        if self.grn_vsn0 is None:
            self.grn_vsn0 = GatedResidualNetwork(hls=self.hls, output_size=num_static,
                                                 dropout_rate=self.dropout_rate,
                                                 use_time_distributed=False,
                                                 altered_glu=True)

        mlp_outputs = self.grn_vsn0(flatten)

        sparse_weights = self.sparse_activation(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        trans_emb_list = tf.TensorArray(tf.float32, size=num_static)
        for i in range(self.num_static):
            tmp = self.es[i](embedding[:, i:i + 1, :])
            trans_emb_list = trans_emb_list.write(i, tmp)

        transformed_embedding = trans_emb_list.stack()
        transformed_embedding = tf.transpose(transformed_embedding, perm=[2, 1, 0, 3])
        transformed_embedding = tf.squeeze(transformed_embedding, axis=0)
        combined = self.multiply_layer([sparse_weights, transformed_embedding])

        static_vec = tf.math.reduce_sum(combined, axis=1)

        return static_vec, sparse_weights


class TemporalVSN(tf.keras.layers.Layer):

    def __init__(self, hls, dropout_rate=.1, num_inputs=None):
        super(TemporalVSN, self).__init__()
        self.hls = hls
        self.dropout_rate = dropout_rate
        self.num_inputs = num_inputs

        self.grn0 = TemporalGatedResidualNetwork(self.hls, output_size=self.num_inputs,
                                                 dropout_rate=self.dropout_rate,
                                                 use_time_distributed=True,
                                                 return_gate=True, altered_glu=True)

        self.sparse_activation_layer = tf.keras.layers.Activation('softmax')

        self.et = [GatedResidualNetwork(hls=hls, dropout_rate=dropout_rate, use_time_distributed=True) \
                   for i in range(num_inputs)]

        self.mult_layer = tf.keras.layers.Multiply()

        self.scv = None

    def call(self, inputs, context):
        _, time_steps, embedding_dim, num_inputs = inputs.get_shape().as_list()

        flatten = tf.reshape(inputs, [-1, time_steps, embedding_dim * num_inputs])

        mlp_outputs, static_gate = self.grn0(inputs=flatten, context=context)  # self.scv)

        sparse_weights = self.sparse_activation_layer(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=2)

        trans_emb_list = tf.TensorArray(tf.float32, size=num_inputs)

        for i in range(num_inputs):
            grn_output = self.et[i](inputs[Ellipsis, i])
            trans_emb_list = trans_emb_list.write(i, grn_output)

        transformed_embedding = trans_emb_list.stack()
        transformed_embedding = tf.transpose(transformed_embedding, perm=[1, 2, 3, 0])

        combined = self.mult_layer([sparse_weights, transformed_embedding])
        temporal_ctx = tf.math.reduce_sum(combined, axis=-1)

        return temporal_ctx, sparse_weights, static_gate


class MLP(tf.keras.layers.Layer):

    def __init__(self, hls=64, output_size=None, output_activation=None,
                 hidden_activation='tanh', use_time_distributed=False):

        super(MLP, self).__init__()
        self.hls = hls
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.use_time_distributed = use_time_distributed

        if use_time_distributed:
            self.hidden1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.hls,
                                                                                 activation=hidden_activation))
            self.hidden2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size,
                                                                                 activation=self.output_activation))
        else:
            self.hidden1 = tf.keras.layers.Dense(self.hls, activation=hidden_activation)
            self.hidden2 = tf.keras.layers.Dense(self.output_size, activation=self.output_activation)

    def call(self, inputs):

        out1 = self.hidden1(inputs)
        out2 = self.hidden2(out1)

        return out2


class TemporalFusionTransformer(tf.keras.Model):

    def __init__(self, num_heads, input_seq_len, output_size,target_seq_len,
                 known_reg_inputs, future_inputs, static_inputs, target_inputs,
                 attn_hls=64, final_mlp_hls=128, unknown_inputs=7,
                 hls=64, cat_inputs=None,
                 rate=.2):

        super().__init__()

        self.hls = hls
        self.dropout_rate = rate
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads
        self.output_size = output_size

        self.attn_hls = attn_hls
        self.final_mlp_hls = final_mlp_hls

        self.static_locs = static_inputs
        self.known_locs = known_reg_inputs
        self.future_locs = future_inputs
        self.target_locs = target_inputs

        self.FinalLoopSize = 1

        self.embedding = HiD_EmbeddingLayer(time_steps=target_seq_len + input_seq_len,
                                            known_reg_inputs=known_reg_inputs, future_inputs=future_inputs,
                                            static_inputs=static_inputs, target_loc=target_inputs)

        # add static VSN

        self.static_vsn = StaticVSN(hls=hls, dropout_rate=rate)

        # add Temporal VSN
        self.temporal_vsn1 = TemporalVSN(hls=hls, dropout_rate=rate,
                                         num_inputs=len(future_inputs) + unknown_inputs)
        self.temporal_vsn2 = TemporalVSN(hls=hls, dropout_rate=rate,
                                         num_inputs=len(future_inputs))

        self.temporal_layer_norm = tf.keras.layers.LayerNormalization()

        self.enriched_grn = TemporalGatedResidualNetwork(hls=self.hls,
                                                         dropout_rate=self.dropout_rate,
                                                         use_time_distributed=True,
                                                         return_gate=True)

        self.mlha = InterpretableMultiHeadAttention(n_head=self.num_heads, d_model=self.attn_hls, dropout=self.dropout_rate)

        self.static_grn1 = GatedResidualNetwork(self.hls,
                                                dropout_rate=self.dropout_rate,
                                                use_time_distributed=False)
        self.static_grn2 = GatedResidualNetwork(self.hls,
                                                dropout_rate=self.dropout_rate,
                                                use_time_distributed=False)
        self.static_grn3 = GatedResidualNetwork(self.hls,
                                                dropout_rate=self.dropout_rate,
                                                use_time_distributed=False)
        self.static_grn4 = GatedResidualNetwork(self.hls,
                                                dropout_rate=self.dropout_rate,
                                                use_time_distributed=False)

        self.lstm1 = tf.keras.layers.LSTM(self.hls, return_sequences=True, return_state=True, stateful=False,
                                          activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0,
                                          unroll=False,
                                          use_bias=True)

        self.lstm2 = tf.keras.layers.LSTM(self.hls, return_sequences=True, return_state=False, stateful=False,
                                          activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0,
                                          unroll=False, use_bias=True)

        self.lstmGLU = GLU(hls, rate, activation=None)

        self.mlp = MLP(self.final_mlp_hls, output_size, output_activation=None, hidden_activation='selu',
                       use_time_distributed=True)

        self.final_glus = [GLU(self.hls, dropout_rate=self.dropout_rate, activation=None) for i in
                           range(self.FinalLoopSize)]
        self.final_norm1 = tf.keras.layers.LayerNormalization()
        self.final_add1 = tf.keras.layers.Add()
        self.final_add2 = tf.keras.layers.Add()

        self.decoder = GatedResidualNetwork(hls=self.hls, dropout_rate=self.dropout_rate, use_time_distributed=True)
        self.decoder_glu = GLU(hls=self.hls, activation=None)
        self.final_norm2 = tf.keras.layers.LayerNormalization()

    def get_decoder_mask(self, attn_inputs):

        len_s = tf.shape(attn_inputs)[1]
        bs = tf.shape(attn_inputs)[:1]
        mask = tf.math.cumsum(tf.eye(len_s, batch_shape=bs), 1)
        return mask

    def process_inputs(self, inputs):

        inputs = inputs[0]
        times, feature = inputs[0].get_shape().as_list()

        static_inputs = [inputs[:, 0, s:s + 1] for s in range(feature) if s in self.static_locs]

        unknown_inputs = [inputs[Ellipsis, u:u + 1] for u in range(feature) if u not in self.known_locs]

        known_inputs = [inputs[Ellipsis, k:k + 1] for k in self.known_locs if k not in self.static_locs]

        return known_inputs, unknown_inputs, static_inputs

    def call(self, inputs, training):

        known_inputs, unknown_inputs, static_inputs = self.process_inputs(inputs)

        known_emb, unknown_emb, static_emb = self.embedding(known_inputs, unknown_inputs, static_inputs)

        if unknown_emb is not None:
            historical_inputs = tf.concat([unknown_emb[:, :self.input_seq_len, :],
                                           known_emb[:, :self.input_seq_len, :]], axis=-1)

        future_inputs = known_emb[:, self.input_seq_len:, :]

        static_encoder, static_weights = self.static_vsn(static_emb)

        scvs = self.static_grn1(static_encoder)
        scvs_e = tf.expand_dims(scvs, axis=1)

        static_context_enrichment = self.static_grn2(static_encoder)
        static_context_state_h = self.static_grn3(static_encoder)
        static_context_state_c = self.static_grn4(static_encoder)

        # temporal VSN

        historical_features, historical_flags, _ = self.temporal_vsn1(historical_inputs, context=scvs_e)
        future_features, future_flags, _ = self.temporal_vsn2(future_inputs, context=scvs_e)

        history_lstm, state_h, state_c = self.lstm1(historical_features, initial_state=[static_context_state_h,
                                                                                        static_context_state_c])

        future_lstm = self.lstm2(future_features, initial_state=[state_h, state_c])

        lstm_layer = tf.concat([history_lstm, future_lstm], axis=1)

        input_embeddings = tf.concat([historical_features, future_features], axis=1)

        lstm_layer, _ = self.lstmGLU(lstm_layer)

        tmp = tf.keras.layers.add([lstm_layer, input_embeddings])
        temporal_feature_layer = self.temporal_layer_norm(tmp)

        expanded_static_context = tf.expand_dims(static_context_enrichment, axis=1)

        enriched, _ = self.enriched_grn(inputs=temporal_feature_layer, context=expanded_static_context)

        mask = self.get_decoder_mask(enriched)

        xsve, attn = self.mlha(enriched, enriched, enriched, mask=mask)

        if self.FinalLoopSize > 1:
            StackLayers = tf.TensorArray(tf.float32, self.FinalLoopSize)
        for FinalGatingLoop in range(0, self.FinalLoopSize):
            x, _ = self.final_glus[FinalGatingLoop](xsve)
            x = self.final_add1([x, enriched])
            x = self.final_norm1(x)

            decoder = self.decoder(x)

            decoder, _ = self.decoder_glu(decoder)

            transformer_layer = self.final_add2([decoder, temporal_feature_layer])
            transformer_layer = self.final_norm2(transformer_layer)

        outputs = self.mlp(transformer_layer[Ellipsis, self.input_seq_len:, :])

        attention_weights = {'decoder_self_attn': attn,
                             'static_flags': static_weights[Ellipsis, 0],
                             'historical_flags': historical_flags[Ellipsis, 0, :],
                             'future_flags': future_flags[Ellipsis, 0, :]}

        return outputs, attention_weights



