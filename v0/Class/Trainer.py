from turtle import position
import tensorflow as tf
from tqdm.auto import tqdm
import numpy as np
import gc, sys

from typing import Tuple

from time import process_time
from datetime import timedelta

from Class.ParameterManager import ParameterManager
from Class.TemporalFusionTransformer import TemporalFusionTransformer

### TF functions
class Trainer:
    def __init__(self, parameterManager:ParameterManager, disable_progress=True) -> None:
        self.batch_size = parameterManager.batch_size
        self.parameterManager = parameterManager
        self.disable_progress = disable_progress
        
        self.metric_object = tf.keras.metrics.MeanSquaredError()
        self.loss_object = tf.keras.losses.MeanSquaredError()
    
    def create_model(self):
        tft_params = self.parameterManager.tft_params
        attn_params = self.parameterManager.attn_params
        unknown_inputs = self.parameterManager.unknown_inputs
        
        model = TemporalFusionTransformer(input_seq_len=tft_params['input_sequence_length'],
            target_seq_len=tft_params['target_sequence_length'],
            output_size=tft_params['output_size'],
            static_inputs=tft_params['static_locs'],
            target_inputs=tft_params['target_loc'],
            future_inputs=tft_params['future_locs'],
            known_reg_inputs=tft_params['static_locs'] + tft_params['future_locs'],
            attn_hls=attn_params['hidden_layer_size'],
            num_heads=attn_params['num_heads'],
            final_mlp_hls=tft_params['final_mlp_hidden_layer'],
            unknown_inputs=unknown_inputs,
            cat_inputs=tft_params['categorical_loc'], rate=tft_params['dropout_rate']
        )
        return model

    @tf.function
    def train_step(self, model, input, target):
        with tf.GradientTape() as tape:
            prediction, _ = model([input, target], training=True)
            loss = self.loss_object(target, prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.metric_object.update_state(target, prediction)

    def run_train(self, model, data):
        self.metric_object.reset_states()
        progress_bar = tqdm(range(len(data)), desc=f'Train: ', file=sys.stdout, disable=self.disable_progress)
        
        for (_, (input, target)) in enumerate(data):
            self.train_step(model, input, target)

            progress_bar.update(1)
            progress_bar.set_postfix(train_loss='{:g}'.format(self.metric_object.result().numpy()))

        return self.metric_object.result().numpy()

    @tf.function
    def test_step(self, model: TemporalFusionTransformer, input, target):
        return model([input, target], training=False)

    def run_validation(self, model: TemporalFusionTransformer, data):
        self.metric_object.reset_states()

        progress_bar = tqdm(range(len(data)), desc=f'Validation: ', file=sys.stdout, disable=self.disable_progress)
            
        for (_, (input, target)) in enumerate(data):
            prediction, _ = self.test_step(model, input, target)
            self.metric_object.update_state(target, prediction)

            progress_bar.update(1)
            progress_bar.set_postfix(validation_loss='{:g}'.format(self.metric_object.result().numpy()))

        return self.metric_object.result().numpy()

    def fit(
        self, model: TemporalFusionTransformer, optimizer: tf.keras.optimizers.Optimizer, 
        train_batch:tf.data.Dataset, validation_batch:tf.data.Dataset, 
        checkpointManager:tf.train.CheckpointManager, early_stopping_patience:int=3
    ):
        self.optimizer = optimizer
        print(f'Running the model for {self.parameterManager.epochs} epochs.')

        # for saving the best model
        best_loss = np.inf
        # for early stopping if performance doesn't improve 
        patience_counter = 0

        history = {
            'train_loss':[],
            'validation_loss':[]
        }

        for epoch in range(self.parameterManager.epochs):
            print(f"Epoch {epoch+1}")
            gc.collect()

            train_start_time = process_time()
            train_loss = self.run_train(model, train_batch)
            train_end_time = process_time()

            validation_loss = self.run_validation(model, validation_batch)

            train_time, validation_time = timedelta(seconds=train_end_time - train_start_time), timedelta(seconds=process_time()-train_end_time)
            print(f'Train loss {train_loss:g}, time {train_time}. Validation loss {validation_loss:g}, time {validation_time}')

            history['train_loss'].append(train_loss)
            history['validation_loss'].append(validation_loss)

            if validation_loss < best_loss:
                ckpt_save_path = checkpointManager.save()
                print(f'Loss improved from {best_loss:g} to {validation_loss:g}')

                best_loss = validation_loss
                patience_counter = 0
                print(f'\nSaving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
            else:
                patience_counter +=1
                if patience_counter == early_stopping_patience:
                    print(f'\nPerformance did not improve for the last {patience_counter} epochs. Early stopping..')
                    break
                else:
                    print(f'Early stop counter {patience_counter}/{early_stopping_patience}')
            # break
        
        self.optimizer = None
        gc.collect()
        return history

    def load_from_checkpoint(self, checkpoint, checkpoint_path:str):
        try:
            # https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f'Checkpoint restored from {checkpoint_path}')
            return checkpoint.model
        except:
            ValueError(f'No valid checkpoint at {checkpoint_path}')
            return None

    def predict(self, model, data) -> Tuple[np.ndarray, np.ndarray, dict]:
        predictions = []
        actuals = []

        weight_dict = {'static_flags': [], 'historical_flags': [], 'future_flags': [], 'decoder_self_attn': []}
        progress_bar = tqdm(range(len(data)), file=sys.stdout, disable=self.disable_progress)

        for (_, (input, target)) in enumerate(data):
            prediction, attention_weights = self.test_step(model, input, target)

            for key in attention_weights.keys():
                weight_dict[key].append(np.array(attention_weights[key]))
            
            predictions.append(prediction)
            actuals.append(target)
            
            progress_bar.update(1)

        return np.concatenate(predictions), np.concatenate(actuals), weight_dict