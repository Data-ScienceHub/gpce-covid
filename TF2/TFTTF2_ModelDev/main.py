import tensorflow as tf
from tensorflow.keras.utils import Progbar

import numpy as np
import pandas as pd

import json
import argparse
import os
import time

from param_manager import ParamManager
from TFTModel import TemporalFusionTransformer
from data_manager import DataManager


def main():

    parser = argparse.ArgumentParser(description='Run TFT', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', '--configPath',
                        help='Path the the .config file to read in paremeters for TFT', type=str, default=None)
    parser.add_argument('-c', '--checkpointPath',
                        help='Path to store model checkpoints and weights', type=str, default=None)
    parser.add_argument('-d', '--dataPath',
                        help='Path to TFT Data csv', type=str, default=None)

    args = parser.parse_args()
    if args.configPath is None:
        raise ValueError('You did not pass a path to a configuration file.')
    if args.checkpointPath is None:
        raise ValueError('You did not pass a checkpoint path argument')
    if args.dataPath is None:
        raise ValueError('You did not pass a data path argument.')

    pManager = ParamManager(args.configPath)

    tft_params = pManager.tft_params
    attn_params = pManager.attn_params
    optimizer_params = pManager.optimizer_params
    col_mappings = pManager.col_mappings
    data_params = pManager.data_params

    tseq_length = tft_params['input_sequence_length'] + tft_params['target_sequence_length']

    unk_inputs = tft_params['total_inputs'] - len(tft_params['static_locs']) - len(tft_params['future_locs'])

    transformer = TemporalFusionTransformer(input_seq_len=tft_params['input_sequence_length'],
                                            target_seq_len=tft_params['target_sequence_length'],
                                            output_size=tft_params['output_size'],
                                            static_inputs=tft_params['static_locs'],
                                            target_inputs=tft_params['target_loc'],
                                            future_inputs=tft_params['future_locs'],
                                            known_reg_inputs=tft_params['static_locs'] + tft_params['future_locs'],
                                            attn_hls=attn_params['hidden_layer_size'],
                                            num_heads=attn_params['num_heads'],
                                            final_mlp_hls=tft_params['final_mlp_hidden_layer'],
                                            unknown_inputs=unk_inputs,
                                            cat_inputs=None, rate=tft_params['dropout_rate'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    if tft_params['loss'].upper() == 'MSE':
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        print('No other losses defined in main method')

    if optimizer_params['optimizer'].lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.01)
    else:
        print('No other optimizers defined in main method')

    checkpoint_path = args.checkpointPath

    try:
        os.stat(checkpoint_path)
    except:
        os.mkdir(checkpoint_path)

    ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    data_manager = DataManager(args.dataPath, tseq_length, col_mappings, data_params)
    data_manager.createTFData()

    train_batches = data_manager.training_data
    num_training_samples = data_manager.num_samples
    batch_size = data_params['batch_size']
    metrics_names = ['train_loss']
    EPOCHS = tft_params['epochs']

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored: ' + ckpt_manager.latest_checkpoint)

    @tf.function
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar], training=True)

            loss = loss_object(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()

        pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)
        p_counter = 0
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)
            values = [('train_loss', train_loss.result())]
            p_counter += inp.shape[0]
            pb_i.update(p_counter, values=values)

        if batch % 50 == 0:
            print(
                f'\nEpoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'\nSaving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'\nEpoch {epoch + 1} Loss {train_loss.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


if __name__ == '__main__':
    main()
