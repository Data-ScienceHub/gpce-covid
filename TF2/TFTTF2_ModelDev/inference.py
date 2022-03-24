import numpy as np
import tensorflow as tf
import argparse
import os
import time
import sys

from param_manager import ParamManager
from TFTModel import TemporalFusionTransformer
from data_manager import DataManager


def OneDayAheadComparison(batched_data, batched_preds, Nloc, tseq_len):
    '''
    Args:
        batched_data:
        batched_preds:
        Nloc: num_locations
        tseq_len: target sequence length

    Returns:
        None ** Currently need to decide what we want from this.

    '''

    seq, times, feat = batched_data.shape
    dseq = int(seq / Nloc)

    TargetMatrix = np.zeros((Nloc, dseq + tseq_len - 1, 1))
    PredMatrix = np.zeros((Nloc, dseq + tseq_len - 1, 1))

    locCounter = 0
    TimeCounter = 0

    for Sequence in range(seq):

        if Sequence != 0 and Sequence % dseq == 0:
            locCounter += 1
            TimeCounter = 0

        for TimeStep in range(times):  # TimeStep goes from 0 to 14 (length = 15)
            TargetMatrix[locCounter, TimeCounter + TimeStep] = batched_data['Target'][Sequence, TimeStep]
            PredMatrix[locCounter, TimeCounter + TimeStep] = np_preds[Sequence, TimeStep]

        TimeCounter += 1

    TargetMatrix = np.clip(TargetMatrix, 0, TargetMatrix.max() + 1)
    PredMatrix = np.clip(PredMatrix, 0, PredMatrix.max() + 1)

    return


def metrics():
    return


def attentions():
    return


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

    if tft_params['loss'].upper() == 'MSE':
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        print('No other losses defined in main method')

    if optimizer_params['optimizer'].lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_params['learning_rate'],
                                             clipnorm=optimizer_params['clipnorm'])
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

    inf_batches = data_manager.training_data
    num_training_samples = data_manager.num_samples
    batch_size = data_params['batch_size']

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored: ' + ckpt_manager.latest_checkpoint)
    else:
        print('There are no checkpoints with those model parameters.')
        sys.exit(0)

    restored_transformer = ckpt.transformer

    warmup = next(iter(inf_batches))
    _ = restored_transformer(warmup)

    preds = []
    attn_weights = []

    for (batch, (inp, tar)) in enumerate(inf_batches):
        pred, attn = restored_transformer
        preds.append(pred)
        attn_weights.append(attn)

    np_preds = np.concatenate(preds)

    Nloc = data_manager.training[col_mappings['ID']].nunique()
    step_size = int(np_preds.shape[0] / Nloc)
    counter = -step_size
    ReshapedPreds = []
    LocID = {}

    batched_data = data_manager.batch_data(data_manager.training)

    for idx, i in enumerate(range(step_size, np_preds.shape[0] + 1, step_size)):
        counter += step_size
        tmp = np.swapaxes(np_preds[counter:i], 1, 2)
        ReshapedPreds.append(tmp)
        LocID[idx] = batched_data['ID'][i - 1][0]

    ReshapedPreds = np.concatenate(ReshapedPreds, axis=1)
