import numpy as np
import tensorflow as tf
import argparse
import os
import time
import sys
import matplotlib.pyplot as plt

from param_manager import ParamManager
from TFTModel import TemporalFusionTransformer
from data_manager import DataManager


def metrics():
    return


def attentions():
    return


def makeSummedPlot(batched_data, Nloc, preds, RunName, figPath):
    seq, times, feat = batched_data['Target'].shape
    dseq = int(seq / Nloc)

    print('Batched data target shape')
    print(batched_data['Target'].shape)

    # Construct new matrix to store averages
    #   shape = (Location x TimeSteps x Features)

    TargetMatrix = np.zeros((Nloc, dseq + 15 - 1, 1))
    PredMatrix = np.zeros((Nloc, dseq + 15 - 1, 1))
    locCounter = 0
    TimeCounter = 0

    for Sequence in range(seq):

        if Sequence != 0 and Sequence % dseq == 0:
            # Reset Time counter and increment locations
            locCounter += 1
            TimeCounter = 0

        for TimeStep in range(times):  # TimeStep goes from 0 to 14 (length = 15)
            TargetMatrix[locCounter, TimeCounter + TimeStep] = batched_data['Target'][Sequence, TimeStep]
            PredMatrix[locCounter, TimeCounter + TimeStep] = preds[Sequence, TimeStep]

        TimeCounter += 1

    # Divide matrix chunk would be used if we would like to average predictions for a given day. Given
    # that we have overlapping sequences, we will also have overlapping predictions. Currently we take first

    # # Divide Matrix ---> to incorporate this into the above code
    # for idx,i in enumerate(TargetMatrix):
    #   for jdx,j in enumerate(i):
    #     if jdx >= times-1 and jdx <= TargetMatrix.shape[1] - times:
    #       TargetMatrix[idx,jdx] = np.divide(TargetMatrix[idx,jdx], times)
    #       PredMatrix[idx,jdx] = np.divide(PredMatrix[idx,jdx], times)
    #     else:
    #       divisor = min(abs(jdx+1), abs(TargetMatrix.shape[1]-jdx))
    #       TargetMatrix[idx,jdx] = np.divide(TargetMatrix[idx,jdx], divisor)
    #       PredMatrix[idx,jdx] = np.divide(PredMatrix[idx,jdx], divisor)

    TargetMatrix = np.clip(TargetMatrix, 0, TargetMatrix.max() + 1)
    PredMatrix = np.clip(PredMatrix, 0, PredMatrix.max() + 1)

    tmpTarg = np.sum(TargetMatrix, axis=0)
    tmpPred = np.sum(PredMatrix, axis=0)
    Error = np.abs(np.subtract(tmpTarg, tmpPred))

    ext = '.png'
    savePath = figPath + RunName + '-Iteration' + ext

    def checkPath(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + str(counter) + extension

        return path

    savePath = checkPath(savePath)

    plt.figure(figsize=(8, 6))
    plt.title(RunName)
    plt.plot(tmpTarg)
    plt.plot(tmpPred)
    plt.plot(Error, color='red')
    plt.savefig(savePath)


def getPredictions(model, batches):
    warmup = next(iter(batches))
    _ = model(warmup)
    print('Warmup batch complete')

    preds = []
    attn_weights = []

    for (batch, (inp, tar)) in enumerate(batches):
        pred, attn = model([inp, tar])
        preds.append(pred)
        attn_weights.append(attn)

    np_preds = np.concatenate(preds)

    return np_preds, attn


def main():
    parser = argparse.ArgumentParser(description='Run TFT', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', '--configPath',
                        help='Path the the .config file to read in paremeters for TFT', type=str, default=None)
    parser.add_argument('-c', '--checkpointPath',
                        help='Path to store model checkpoints and weights', type=str, default=None)
    parser.add_argument('-d', '--dataPath',
                        help='Path to TFT Data csv', type=str, default=None)
    parser.add_argument('-f', '--figPath',
                        help='Path where to save matplotlib figures to', type=str, default=None)

    args = parser.parse_args()
    if args.configPath is None:
        raise ValueError('A path to the configuration file was not passed.')
    if args.checkpointPath is None:
        raise ValueError('A path to the checkpoint directory was not passed.')
    if args.dataPath is None:
        raise ValueError('A path to the data directory was not passed.')
    if args.figPath is None:
        print('WARNING: a directory path to save the figures was not passed, the data directory path will be used.')
        if os.path.exists(args.dataPath + '/figs'):
            pass
        else:
            os.makedirs(args.dataPath + '/figs')
        figPath = args.dataPath + '/figs'
    else:
        if not os.path.exists(args.figPath):
            os.makedirs(args.figPath)
            figPath = args.figPath
        else:
            figPath = args.figPath


    pManager = ParamManager(args.configPath)

    tft_params = pManager.tft_params
    attn_params = pManager.attn_params
    optimizer_params = pManager.optimizer_params
    col_mappings = pManager.col_mappings
    data_params = pManager.data_params
    support_params = pManager.support_params

    RunName = support_params['RunName']
    tseq_length = tft_params['input_sequence_length'] + tft_params['target_sequence_length']
    unk_inputs = tft_params['total_inputs'] - len(tft_params['static_locs']) - len(tft_params['future_locs'])

    # Restoring the model

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
        print('No other optimizers defined')

    checkpoint_path = args.checkpointPath

    try:
        os.stat(checkpoint_path)
    except:
        os.mkdir(checkpoint_path)

    ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    data_manager = DataManager(args.dataPath, tseq_length, col_mappings, data_params)
    data_manager.createTFData()

    inf_batches = data_manager.inference_data
    num_training_samples = data_manager.num_samples
    batch_size = data_params['batch_size']

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored: ' + ckpt_manager.latest_checkpoint)
    else:
        print('There are no checkpoints with those model parameters.')
        sys.exit(0)

    restored_transformer = ckpt.model

    # Get the predictions from restored model
    preds, attn_weights = getPredictions(restored_transformer, inf_batches)

    Nloc = data_manager.training[col_mappings['ID']].nunique().values[0]
    print('NLOC')
    print(Nloc)
    # step_size = int(preds.shape[0] / Nloc)
    # counter = -step_size
    # ReshapedPreds = []
    # LocID = {}

    # batched_data = data_manager.batch_data(data_manager.training)
    np_inference = data_manager.np_inference
    makeSummedPlot(np_inference, Nloc, preds, RunName, figPath)
    #make individual plots next

if __name__ == '__main__':
    main()