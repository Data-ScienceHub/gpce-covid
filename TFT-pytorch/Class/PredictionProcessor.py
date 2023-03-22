import pandas as pd
import numpy as np

weekdays = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

class PredictionProcessor:
    """
    Converts the TFT output into plotable dataframe format
    """

    def __init__(
        self, time_idx, group_id, max_prediction_length, 
        targets, train_start, max_encoder_length
    ) -> None:
        self.time_idx = time_idx
        self.group_id = group_id
        self.max_prediction_length = max_prediction_length
        self.targets = targets
        self.train_start = train_start
        self.max_encoder_length = max_encoder_length

    def convert_prediction_to_dict(
            self, raw_predictions, index, target_time_step:int=None
        ):
        time_index = index[self.time_idx].values
        fips = index[self.group_id].values

        predictions = raw_predictions.copy()
        # set negative predictions to zero
        predictions[predictions<0] = 0

        results = {}

        # if you want result for only a specific number of days in the future
        if target_time_step is not None:
            assert 0 < target_time_step <= self.max_prediction_length,\
            f"Expects target time step within 1 and {self.max_prediction_length}, found {target_time_step}."

            # convert target day to index, as it starts from 0
            target_time_step -= 1
            for index in range(len(predictions)):
                # given time index is the time index of the first prediction
                # https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.base_model.BaseModel.html#pytorch_forecasting.models.base_model.BaseModel.predict
                current_time_index = time_index[index]
                current_fips = fips[index]

                item = (current_fips, current_time_index + target_time_step)
                predicted_value = predictions[index][target_time_step]
                results[item] = [predicted_value]

            return results

        # if you haven't specified a particular day, this returns all of them, 
        # so that you can take the average
        for index in range(len(predictions)):
            current_time_index = time_index[index]
            current_fips = fips[index]

            for time_step in range(self.max_prediction_length):
                item = (current_fips, current_time_index + time_step)
                predicted_value = predictions[index][time_step]

                if item in results:
                    results[item].append(predicted_value)
                else:
                    results[item] = [predicted_value]

        return results

    def convert_dict_to_dataframe(self, results:dict, feature_name:str):
        fips = []
        predictions = []
        time_index = []

        for key in results.keys():
            item = results[key]
            #TODO: more generalized
            fips.append(key[0])
            time_index.append(key[1])

            predictions.append(np.mean(item))
        
        result_df = pd.DataFrame({
            self.group_id: fips, self.time_idx: time_index,
            f'Predicted_{feature_name}': predictions  
        })
        return result_df

    def align_result_with_dataset(self, df, predictions, index, target_time_step:int=None):
        id_columns = list(index.columns)

        if type(predictions)==list:
            result_df = None
            for i, prediction in enumerate(predictions):
                prediction_df = self.convert_dict_to_dataframe(
                    self.convert_prediction_to_dict(prediction, index, target_time_step),
                    self.targets[i]
                )
                if result_df is None:
                    result_df = prediction_df
                else:
                    result_df = result_df.merge(prediction_df, on=id_columns, how='inner')
        else:
            # when prediction is on a single target, e.g. cases
            result_df = self.convert_dict_to_dataframe(
                self.convert_prediction_to_dict(predictions, index, target_time_step),
                self.targets[0]
            )

        merged_data = result_df.merge(
            df[self.targets + id_columns], on=id_columns, how='inner'
        ).reset_index(drop=True)
        merged_data = merged_data.sort_values(by=id_columns).reset_index(drop=True)
        merged_data['Date'] = self.train_start + pd.to_timedelta(merged_data[self.time_idx], unit='D') 

        # round the values
        predicted_columns = [col for col in merged_data.columns if 'Predicted' in col]
        merged_data[predicted_columns] = merged_data[predicted_columns].round()
        
        return merged_data

    @staticmethod
    def makeSummed(df, targets, columns=['Date']):
        predicted_columns = [col for col in df.columns if 'Predicted' in col]
        return df.groupby(columns)[predicted_columns + targets].aggregate('sum').reset_index()

    def get_attention(self, interpretation, index):
        attention = pd.DataFrame(interpretation['attention'].numpy())
        attention['Date'] = [
            self.train_start + pd.to_timedelta(day, 'D') 
            for day in index[self.time_idx].values.reshape(-1)
        ]
        attention[self.group_id] = index[self.group_id]
        return attention

    def get_mean_attention(
        self, interpretation, index, return_attention=False,
        trim:bool= True
    ) -> pd.DataFrame:
        attention = self.get_attention(interpretation, index)
        max_encoder_length = self.max_encoder_length
        attention_mean = attention.groupby('Date')[
            list(range(max_encoder_length))
        ].aggregate('mean').reset_index()

        # if you want to preserve early max_encoder_length days of attention. But not all days have attention within this time
        if not trim:
            df = pd.DataFrame(
            {encoder_length:[None]*max_encoder_length for encoder_length in range(max_encoder_length)}
            )
            df['Date'] = [
                attention_mean['Date'].min() - pd.to_timedelta(encoder_length, unit='day') 
                for encoder_length in range(1, max_encoder_length+1)
            ]
            attention_mean = pd.concat(
                [df, attention_mean]
            ).reset_index(drop=True).sort_values(by='Date')

        for i in range(max_encoder_length):
            attention_mean[i] = attention_mean[i].shift(
                periods=i-max_encoder_length, fill_value=0
            )

        # comment this out if you want to preserve the last max_encoder_length days of attention. 
        # But not all days have attention within this time
        if trim:
            attention_mean = attention_mean.drop(
                range(attention_mean.shape[0]-max_encoder_length-1, attention_mean.shape[0])
                , axis=0
            )

        attention_mean['mean'] = attention_mean.drop(columns='Date').mean(axis=1)
        attention_mean['median'] = attention_mean.drop(columns='Date').median(axis=1)

        if return_attention:
            return attention_mean, attention
        else:
            return attention_mean

    def get_attention_by_weekday(self, attention_mean, verbose=True):
        attention_mean['weekday'] = attention_mean.Date.dt.weekday.astype(str)
        attention_weekly = attention_mean.groupby('weekday')[
            list(range(self.max_encoder_length))
        ].aggregate('mean').reset_index(drop=True)
    
        attention_weekly['weekday'] = weekdays
        attention_weekly['mean'] = attention_weekly.drop(columns='weekday').mean(axis=1)
        attention_weekly['median'] = attention_weekly.drop(columns='weekday').median(axis=1)
        
        if verbose:
            max_index = np.argmax(attention_weekly[range(self.max_encoder_length)].to_numpy(), axis=0)
            max_days = [
                (-self.max_encoder_length+index, weekdays[day]) 
                for index, day in enumerate(max_index)
            ]
            print(f'Weekdays when these attentions are maximum: \n{max_days}')

        return attention_weekly