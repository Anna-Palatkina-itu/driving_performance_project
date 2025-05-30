import pandas as pd
from datetime import timedelta
import numpy as np

def load_participant(code='L0S1Z2I3',
                     path='participant data',
                     eye_tracking=False):
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
    
    sections = {'before':[before_audio_start, before_audio_end], 'calm':[calm_audio_start, calm_audio_end],
                'interim':[interim_audio_start, interim_audio_end],
                'intense':[intense_audio_start, intense_audio_end],
                'after':[after_audio_start, after_audio_end]}
    

    
    
    return trimmed_data, sections
