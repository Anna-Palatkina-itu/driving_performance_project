import pandas as pd
from datetime import timedelta, datetime, timezone
import numpy as np
import scipy as scp


def ts(x):
  return int(x.timestamp() * 1000)

def construct_timestamps(starting_timestamp):
  """ timestamp in milliseconds"""

  before_duration = timedelta(seconds=210)
  negative_duration = timedelta(seconds=272)
  positive_duration = timedelta(seconds=271)
  interim_duration = timedelta(seconds=49)
  after_duration = timedelta(minutes=15) - before_duration - negative_duration - positive_duration - interim_duration

  before_starting_timestamp = datetime.fromtimestamp(starting_timestamp / 1000, tz=timezone.utc) - before_duration

  before_audio_end = before_starting_timestamp + before_duration
  calm_audio_start = before_audio_end
  calm_audio_end = calm_audio_start + positive_duration
  interim_audio_start = calm_audio_end
  interim_audio_end = interim_audio_start + interim_duration
  intense_audio_start = interim_audio_end
  intense_audio_end = intense_audio_start + negative_duration
  after_audio_start = intense_audio_end
  after_audio_end = after_audio_start + after_duration



  sections = {'before':[starting_timestamp, ts(before_audio_end)], 'calm':[ts(calm_audio_start), ts(calm_audio_end)],
                'interim':[ts(interim_audio_start), ts(interim_audio_end)],
                'intense':[ts(intense_audio_start), ts(intense_audio_end)],
                'after':[ts(after_audio_start), ts(after_audio_end)]}
  return sections


def get_statistics(data, column='DistanceToTargetPosition',
                      xlabel='Distance to Correct Lane', sections=None):
  """
  Calculates mean, standard deviation, and third centered moment for calm and intense segments of the data.
  Returns list with [mean_calm, std_calm, moment_calm, mean_intense, std_intense, moment_intense]."""

  calm_start = sections['calm'][0]
  calm_end = sections['calm'][1]
  intense_start = sections['intense'][0]
  intense_end = sections['intense'][1]

  segment1 = data[(data['Timestamp'] > calm_start) &
                          (data['Timestamp'] < calm_end)][column].dropna()
  segment2 = data[(data['Timestamp'] > intense_start) &
                          (data['Timestamp'] < intense_end)][column].dropna()
#   print(segment1)

  return np.mean(segment1), np.std(segment1), scp.stats.moment(segment1, order=3, center=True),  np.mean(segment2), np.std(segment2), scp.stats.moment(segment2, order=3, center=True)


def load_participant(code='L0S1Z2I3',
                     path='participant data',
                     eye_tracking=False, remove_outliers=None):
  data = pd.read_csv(f'{path}{code}.csv', comment='#', low_memory=False)


  with open(f'{path}{code}.csv', 'r+') as f:
    for line in f.readlines():
      if line.startswith('#Recording time'):
        timestamp_str = line.replace('#Recording time,Date: ', '')
        timestamp_str = timestamp_str[:timestamp_str.find(',Unix time:')]
        break

  try:
    ts = pd.to_datetime(timestamp_str,
                        format="%d.%m.%Y,Time: %H:%M:%S.%f +02:00")
  except:
    ts = pd.to_datetime(timestamp_str,
                        format="%m/%d/%Y,Time: %H:%M:%S.%f +02:00")
  ts = ts - timedelta(hours=2)

  nanoseconds = int(ts.value)




  if eye_tracking:
    state_data = pd.read_csv(f'data/3d_eye_states_{code}.csv', comment='#')     
    state_data['timestamp [ns]'] = state_data['timestamp [ns]'].apply(lambda x: (x - nanoseconds)/ (10**6))

    state_data.rename(columns={'timestamp [ns]': 'Timestamp'}, inplace=True)
    state_data.rename(columns={'pupil diameter left [mm]': 'ET_PupilLeft'}, inplace=True)
    state_data.rename(columns={'pupil diameter right [mm]': 'ET_PupilRight'},
                  inplace=True)

    data['ET_PupilLeft'] = np.nan
    data['ET_PupilRight'] = np.nan

    subset = state_data[['Timestamp', 'ET_PupilLeft','ET_PupilRight']]

    # Append to df1
    combined = pd.concat([data, subset], ignore_index=True)

    # Sort by timestep
    combined_sorted = combined.sort_values(by='Timestamp').reset_index(drop=True)
    data = combined_sorted
    
  
  interpolated_data = data.ffill()
  
  try:
      trimmed_data = interpolated_data[['Timestamp', 'Channel 13 (Raw)','Channel 9 (Raw)', 'MarkerName','MarkerDescription','MarkerType', 'DistanceToNextSpeedSign', 'DistanceToNextOverheadSign', 'VelocityX', 'DistanceToTargetPosition',
              'DistanceToTargetSpeed', 'CarSpeed', 'ET_PupilLeft', 'ET_PupilRight']]
  except:
      trimmed_data = interpolated_data[['Timestamp', 'Channel 13 (ECG100C)','Channel 9 (EDA100C)', 'MarkerName','MarkerDescription','MarkerType', 'DistanceToNextSpeedSign', 'DistanceToNextOverheadSign', 'VelocityX', 'DistanceToTargetPosition',
              'DistanceToTargetSpeed', 'CarSpeed', 'ET_PupilLeft', 'ET_PupilRight']]
        
  try:      
    before_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='Experiment') & (trimmed_data['MarkerType']=='S')].iloc[0]
    before_audio_end = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='Experiment') & (trimmed_data['MarkerType']=='S')].iloc[-1]

    calm_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='CalmAudio') & (trimmed_data['MarkerType']=='S')].iloc[0]
    calm_audio_end = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='CalmAudio') & (trimmed_data['MarkerType']=='E')].iloc[0]

    interim_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='InterimAudio') & (trimmed_data['MarkerType']=='S')].iloc[0]
    interim_audio_end = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='InterimAudio') & (trimmed_data['MarkerType']=='E')].iloc[0]

    intense_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='IntenseAudio') & (trimmed_data['MarkerType']=='S')].iloc[0]
    intense_audio_end = trimmed_data[trimmed_data['MarkerName'].notna() & (trimmed_data['MarkerName']=='IntenseAudio') & (trimmed_data['MarkerType']=='E')].iloc[0]


    calmFirst = calm_audio_start['Timestamp'] < intense_audio_end['Timestamp']

    if calmFirst:
      after_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (
          trimmed_data['MarkerName'] == 'IntenseAudio') &
                                      (trimmed_data['MarkerType'] == 'E')].iloc[0]

      after_audio_end = trimmed_data[trimmed_data['MarkerName'].notna()
                                    & (trimmed_data['MarkerName'] == 'Experiment')
                                    & (trimmed_data['MarkerType'] == 'E')].iloc[0]

    else:
      after_audio_start = trimmed_data[trimmed_data['MarkerName'].notna() & (
          trimmed_data['MarkerName'] == 'CalmAudio') &
                                      (trimmed_data['MarkerType'] == 'E')].iloc[0]
      after_audio_end = trimmed_data[trimmed_data['MarkerName'].notna()
                                    & (trimmed_data['MarkerName'] == 'Experiment')
                                    & (trimmed_data['MarkerType'] == 'E')].iloc[0]

    sections = {'before':[before_audio_start['Timestamp'], before_audio_end['Timestamp']], 'calm':[calm_audio_start['Timestamp'], calm_audio_end['Timestamp']],
                  'interim':[interim_audio_start['Timestamp'], interim_audio_end['Timestamp']],
                  'intense':[intense_audio_start['Timestamp'], intense_audio_end['Timestamp']],
                  'after':[after_audio_start['Timestamp'], after_audio_end['Timestamp']]}
  except:
    sections = construct_timestamps(int(before_audio_start['Timestamp']))
    
  if remove_outliers:
    for column in remove_outliers:
      trimmed_data = replace_outliers_from_column(trimmed_data, column=column, threshold=3)

  return trimmed_data, sections



def replace_outliers_from_column(data, column='DistanceToTargetPosition', threshold=3):
    """
    Removes outliers from the specified column in the DataFrame based on a z-score threshold.
    Outliers are replaced by the column mean if their z-score exceeds the threshold.
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        column (str): The column from which to remove outliers.
        threshold (float): The z-score threshold for identifying outliers.
    """
    new_data = data.copy()
    z_scores = (new_data[column] - new_data[column].mean()) / new_data[column].std()
    outliers = np.abs(z_scores) > threshold
    new_data.loc[outliers, column] = new_data[column].mean()
    return new_data

def remove_outliers_from_column(data, column='DistanceToTargetPosition', threshold=3):
    """
    Removes outliers from the specified column in the DataFrame based on a z-score threshold.
    Outliers are dropped from the DataFrame.
    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        column (str): The column from which to remove outliers.
        threshold (float): The z-score threshold for identifying outliers.
    """
    new_data = data.copy()
    z_scores = (new_data[column] - new_data[column].mean()) / new_data[column].std()
    return new_data[~(np.abs(z_scores) > threshold)]