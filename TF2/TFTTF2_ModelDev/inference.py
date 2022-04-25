import numpy as np
import tensorflow as tf
import argparse
import os
import time
import sys
import matplotlib.pyplot as plt
from random import sample

from param_manager import ParamManager
from TFTModel import TemporalFusionTransformer
from data_manager import DataManager

#for testing
import json

def metrics():
    return


def attentions():
    return

class PlotResults:

    def __init__(self, batched_data, attention, Nloc, locs, preds, col_mappings, support_params, RunName, figPath):

        self.batched_data = batched_data
        self.Nloc = Nloc
        self.preds = preds
        self.RunName = RunName
        self.figPath = figPath
        self.locs = locs

        self.col_mappings = col_mappings
        self.support_params = support_params

        self.attn = attention

        self.reshaped_preds = None
        self.reshaped_target = None

    def makeSummedPlot(self):

        print('Predictions shape')
        print(self.preds.shape)

        seq, times, feat = self.batched_data['Target'].shape
        dseq = int(seq / self.Nloc)

        print('Batched data target shape')
        print(self.batched_data['Target'].shape)

        # Construct new matrix to store averages
        #   shape = (Location x TimeSteps x Features)

        TargetMatrix = np.zeros((self.Nloc, dseq + 15 - 1, 1))
        PredMatrix = np.zeros((self.Nloc, dseq + 15 - 1, 1))
        locCounter = 0
        TimeCounter = 0

        for Sequence in range(seq):

            if Sequence != 0 and Sequence % dseq == 0:
                # Reset Time counter and increment locations
                locCounter += 1
                TimeCounter = 0

            for TimeStep in range(times):  # TimeStep goes from 0 to 14 (length = 15)
                TargetMatrix[locCounter, TimeCounter + TimeStep] = self.batched_data['Target'][Sequence, TimeStep]
                PredMatrix[locCounter, TimeCounter + TimeStep] = self.preds[Sequence, TimeStep]

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

        print('Reshaped Targets')
        print(TargetMatrix.shape)
        print('Reshaped Preds')
        print(PredMatrix.shape)

        self.reshaped_target = TargetMatrix
        self.reshaped_preds = PredMatrix

        tmpTarg = np.sum(TargetMatrix, axis=0)
        tmpPred = np.sum(PredMatrix, axis=0)
        Error = np.abs(np.subtract(tmpTarg, tmpPred))

        ext = '.png'
        savePath = self.figPath + self.RunName + '-Iteration' + ext

        def checkPath(path):
            filename, extension = os.path.splitext(path)
            counter = 1

            while os.path.exists(path):
                path = filename + str(counter) + extension
                counter += 1

            return path

        savePath = checkPath(savePath)

        plt.figure(figsize=(8, 6))
        plt.title(self.RunName)
        plt.plot(tmpTarg)
        plt.plot(tmpPred)
        plt.plot(Error, color='red')
        plt.savefig(savePath)

    def makeIndividualPlots(self):

        if self.reshaped_target is None and self.reshaped_target is None:
            raise ValueError('Summed plots have not been created and thus there are no reshaped predictions and targets.')
        else:
            sample_perc = self.support_params['LocationBasedRandomSample']
            if sample_perc > 0:
                int_locs = round(sample_perc * self.Nloc)
                sample_ind = sample(list(enumerate(self.locs)), int_locs)
                sample_ind = [x[0] for x in sample_ind]

                sample_targ = [self.reshaped_target[idx, Ellipsis] for idx in sample_ind]
                sample_pred = [self.reshaped_preds[idx, Ellipsis] for idx in sample_ind]
                sample_locs = [self.locs[idx] for idx in sample_ind]

                for idx, i in enumerate(sample_locs):

                    TargSeries = sample_targ[idx]
                    PredSeries = sample_pred[idx]
                    ErrorSeries = np.abs(np.subtract(TargSeries, PredSeries))

                    ext = '.png'
                    savePath = self.figPath + self.RunName + '-Iteration' + ext

                    def checkPathLoc(path, loc):
                        filename, extension = os.path.splitext(path)
                        counter = 1

                        while os.path.exists(path):
                            path = filename + str(counter) + loc + extension
                            counter += 1
                        return path

                    savePath = checkPathLoc(savePath, i)

                    plt.figure(figsize=(8, 6))
                    plt.title(self.RunName + ' Location: ' + str(i))
                    plt.plot(TargSeries)
                    plt.plot(PredSeries)
                    plt.plot(ErrorSeries, color='red')
                    plt.savefig(savePath)
                    plt.close()

            else:
                print('No random sample of locations chosen')

    def plotDecoderAttention(self):

        decoder_weights = np.concatenate(self.attn['decoder_self_attn'], axis=1)
        # Average weights across all heads
        decoder_weights = decoder_weights.mean(axis=0)
        # Now we have an array of (TotalSequences x TotalSequenceLength x TotalSequenceLength)
        quantiles = [.3, .5, .7]
        dw_qt = np.quantile(decoder_weights, quantiles, axis=0)

        N = decoder_weights.shape[1]
        X = np.arange(N, step=1)

        fig, ax = plt.subplots(N, figsize=(8, 60))
        for i in range(N):
            ax[i].plot(dw_qt[0, i], color='black', label='.3 quantile', alpha=.6)
            ax[i].plot(dw_qt[1, i], color='orange', label='.5 quantile')
            ax[i].plot(dw_qt[2, i], color='blue', label='.7 quantile', alpha=.6)
            ax[i].fill_between(X, dw_qt[0, i], dw_qt[2, i], color='blue', alpha=.2)
            ax[i].set_title('Decoder attention weights at Day ' + str(i + 1))
            ax[i].set_xlabel('Day in Sequence (Total Sequence)')
            ax[i].set_ylabel('Attention Weights')
            ax[i].legend()

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3)

        savePath = self.figPath + self.RunName + '-Iteration' + '-AttentionWeights' + '.pdf'

        plt.savefig(savePath, dpi='figure', bbox_inches='tight')

    def plotObservedWeights(self):

        feature_list = self.col_mappings['Known Regular'] + self.col_mappings['Future'] + self.col_mappings['Target']
        feature_list = [f for f in feature_list if f not in self.col_mappings['Static']]

        # Concatenate the list of historical flag on axis 0
        hf = np.concatenate(self.attn['historical_flags'], axis=0)
        # Take the mean across all days (axis 1)
        hf = hf.mean(axis=1)
        # Now we have a (Total Sequences x Num Features) shape array
        quantiles = [.3, .5, .7]

        hf_qt = np.quantile(hf, quantiles, axis=0)

        N = len(feature_list)
        X = np.arange(N)
        width = 0.25

        savePath = self.figPath + self.RunName + '-Iteration' + '-ObservedVSNWeights' + '.pdf'

        fig, ax = plt.subplots(figsize=(15,8))
        rects1 = ax.bar(X - width, hf_qt[0], width, label='0.3 quantile')
        rects2 = ax.bar(X, hf_qt[1], width, label='0.5 quantile')
        rects3 = ax.bar(X + width, hf_qt[2], width, label='0.5 quantile')
        ax.set_ylabel('Variable Selection Network Weight')
        ax.set_xlabel('Weight by Quantile and Feature')
        ax.set_title('Observed input selection weights by variable')
        ax.set_xticks(X, feature_list)
        ax.tick_params(labelrotation=45)
        plt.legend()
        plt.savefig(savePath)

    def plotStaticWeights(self):

        sf = np.concatenate(self.attn['static_flags'], axis=0)
        quantiles = [.3, .5, .7]
        sf_qt = np.quantile(sf, quantiles, axis=0)

        feature_list = self.col_mappings['Static']

        print('Static Quantiles')
        print(sf_qt)

        N = len(feature_list)
        X = np.arange(N)
        width = 0.25

        savePath = self.figPath + self.RunName + '-Iteration' + '-StaticVSNWeights' + '.pdf'

        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(X - width, sf_qt[0], width, label='0.3 quantile')
        rects2 = ax.bar(X, sf_qt[1], width, label='0.5 quantile')
        rects3 = ax.bar(X + width, sf_qt[2], width, label='0.7 quantile')
        ax.set_ylabel('Variable Selection Network Weight')
        ax.set_xlabel('Weight by Quantile and Feature')
        ax.set_title('Static input selection weights by variable')
        ax.set_xticks(X, feature_list)
        ax.tick_params(labelrotation=45)
        plt.legend()
        plt.savefig(savePath)

    def plotFutureWeights(self):

        ff = np.concatenate(self.attn['future_flags'], axis=0)
        ff = ff.mean(axis=1)

        quantiles = [.3, .5, .7]

        ff_qt = np.quantile(ff, quantiles, axis=0)

        feature_list = self.col_mappings['Future']

        N = len(feature_list)
        X = np.arange(N)
        width = 0.25

        savePath = self.figPath + self.RunName + '-Iteration' + '-FutureVSNWeights' + '.pdf'

        fig, ax = plt.subplots(figsize=(15, 8))
        rects1 = ax.bar(X - width, ff_qt[0], width, label='0.3 quantile')
        rects2 = ax.bar(X, ff_qt[1], width, label='0.5 quantile')
        rects3 = ax.bar(X + width, ff_qt[2], width, label='0.5 quantile')
        ax.set_ylabel('Variable Selection Network Weight')
        ax.set_xlabel('Weight by Quantile and Feature')
        ax.set_title('Future known input selection weights by variable')
        ax.set_xticks(X, feature_list)
        ax.tick_params(labelrotation=45)
        plt.legend()
        plt.savefig(savePath)






def getPredictions(model, batches, config):

    num_static = len(config['static_locs'])
    num_fut = len(config['future_locs'])
    num_tar = len(config['target_loc'])

    num_unk = config['total_inputs'] - num_static - num_fut - num_tar

    weight_dict = {'static_flags': [], 'historical_flags': [], 'future_flags': [], 'decoder_self_attn': []}

    warmup = next(iter(batches))
    _, att = model(warmup, training=False)
    print('Warmup batch complete')

    preds = []

    for (batch, (inp, tar)) in enumerate(batches):
        pred, attn = model([inp, tar], training=False)
        weight_dict['static_flags'].append(np.array(attn['static_flags']))
        weight_dict['historical_flags'].append(np.array(attn['historical_flags']))
        weight_dict['future_flags'].append(np.array(attn['future_flags']))
        weight_dict['decoder_self_attn'].append(np.array(attn['decoder_self_attn']))
        preds.append(pred)

    np_preds = np.concatenate(preds)

    return np_preds, weight_dict


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

    Nloc = data_manager.training[col_mappings['ID']].nunique().values[0]
    print('NLOC')
    print(Nloc)
    locs = data_manager.training[col_mappings['ID']].iloc[:Nloc, 0].values
    print('All locs')
    print(locs)
    # Get the predictions from restored model
    preds, attn_weights = getPredictions(restored_transformer, inf_batches, config=tft_params)

    np_inference = data_manager.np_inference

    PlotC = PlotResults(np_inference, attn_weights, Nloc, locs, preds, col_mappings, support_params, RunName, figPath)
    PlotC.makeSummedPlot()
    PlotC.makeIndividualPlots()

    PlotC.plotDecoderAttention()
    PlotC.plotObservedWeights()
    PlotC.plotFutureWeights()
    PlotC.plotStaticWeights()

    #make individual plots next

if __name__ == '__main__':
    main()