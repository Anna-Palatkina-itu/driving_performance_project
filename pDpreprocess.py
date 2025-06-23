import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')


output_path = "C:/Users/ArezooSedghi/OneDrive - IT U interdisciplinary transformation university austria/Desktop/Measuring_Drivers_Performance/Code/Eye Tracking/preprocessedData/"


class PupilPreprocessor:
    """
    Pupil data preprocessing module that preserves raw data and creates separate preprocessed columns
    """
    
    
        
    def __init__(self, data_path, baseline_method, interpolation_method, 
             apply_lowpass, lowpass_cutoff=6.0, sampling_rate=None):
        """Initialize the preprocessor"""

    
        self.data_path = data_path
        self.baseline_method = baseline_method
        self.interpolation_method = interpolation_method
        self.apply_lowpass = apply_lowpass
        self.lowpass_cutoff = lowpass_cutoff
        self.sampling_rate = sampling_rate

       
        self._parse_baseline_method()

     
        self.timestamp_col = "Timestamp"
        self.start_timestamp_col = "start timestamp [ms]"
        self.end_timestamp_col = "end timestamp [ms]"
        self.duration_col = "duration [ms]"
        self.pupil_left_col = "pupil diameter left [mm]"
        self.pupil_right_col = "pupil diameter right [mm]"

        # temporary processing columns
        self.pupil_left_working = f"{self.pupil_left_col}_working"
        self.pupil_right_working = f"{self.pupil_right_col}_working"

        
        self.pupil_left_corrected = f"{self.baseline_statistic} {self.baseline_operation} baseline corrected {self.pupil_left_col}"
        self.pupil_right_corrected = f"{self.baseline_statistic} {self.baseline_operation} baseline corrected {self.pupil_right_col}"

        
        self.final_left_col = self.pupil_left_corrected
        self.final_right_col = self.pupil_right_corrected

        
        self.data = None
        self.original_data = None
        self.blink_intervals = []
        self.estimated_sampling_rate = None



        

        
    def _parse_baseline_method(self):
        """Parse the baseline method and set baseline attributes"""
        if self.baseline_method == "mean_subtractive":
            self.baseline_statistic = "mean"
            self.baseline_operation = "subtractive"
        elif self.baseline_method == "median_subtractive":
            self.baseline_statistic = "median"
            self.baseline_operation = "subtractive"
        elif self.baseline_method == "mean_divisive":
            self.baseline_statistic = "mean"
            self.baseline_operation = "divisive" 
        elif self.baseline_method == "median_divisive":
            self.baseline_statistic = "median"
            self.baseline_operation = "divisive" 
        else:
            raise ValueError(f" baseline method is not valid it should ne one of mean_subtractive, median_subtractive, mean_divisive, median_divisive: {self.baseline_method}")



    
    def _estimate_sampling_rate(self):
        """
        Estimate sampling rate from timestamp data
        """
        # time differences in milliseconds
        time_diffs = np.diff(self.data[self.timestamp_col].dropna())
        
        #  outliers 
        time_diffs = time_diffs[time_diffs < np.percentile(time_diffs, 95)]
        
        # median time difference in milliseconds
        median_time_diff_ms = np.median(time_diffs)
        
        # Convert to sampling rate in Hz
        sampling_rate = 1000.0 / median_time_diff_ms  
        
        return sampling_rate
    
    def load_data(self):
        
        print("Step 1: Loading data...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            self.original_data = self.data.copy()
            
            print(f"Data loaded successfully!")
            print(f"Shape: {self.data.shape}")
            
            # required columns 
            required_columns = [
                self.timestamp_col,
                self.start_timestamp_col,
                self.end_timestamp_col,
                self.duration_col,
                self.pupil_left_col,
                self.pupil_right_col
            ]
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                # for some data that we dont have pupil diameter
                print(f"Missing required columns: {missing_columns}")
                return False
            
            print(f"required columns found!")
            
            # working copies 
            self.data[self.pupil_left_working] = self.data[self.pupil_left_col].copy()
            self.data[self.pupil_right_working] = self.data[self.pupil_right_col].copy()
            
            print(f"Working copies created - original data preserved")
            
            # Estimate sampling rate for filtering 
            if self.sampling_rate is None:
                self.estimated_sampling_rate = self._estimate_sampling_rate()
                self.sampling_rate = self.estimated_sampling_rate
                print(f"Estimated sampling rate: {self.sampling_rate:.2f} Hz")
            else:
                print(f"Using provided sampling rate: {self.sampling_rate:.2f} Hz")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_and_clean_blinks(self):
        """
        Step 2: Detect blinks and clean pupil diameter values in working copies
        """
        print("\nDetecting and cleaning blinks...")
        
        # Find blink rows (where start, end, and duration have values, even one of them is enough but ...)
        blink_mask = (
            self.data[self.start_timestamp_col].notna() & 
            self.data[self.end_timestamp_col].notna() & 
            self.data[self.duration_col].notna()
        )
        
        blink_rows = self.data[blink_mask].copy()
        print(f"Found {len(blink_rows)} blink events")
        
        if len(blink_rows) == 0:
            print("No blink events found - skipping blink cleaning")
            return True
        
        # Process the blink event
        total_cleaned_points = 0
        
        for idx, blink_row in blink_rows.iterrows():
            start_time = blink_row[self.start_timestamp_col]
            end_time = blink_row[self.end_timestamp_col]
            duration = blink_row[self.duration_col]
            
            # Find all data points within this blink interval during the timestamp range
            blink_interval_mask = (
                (self.data[self.timestamp_col] >= start_time) & 
                (self.data[self.timestamp_col] <= end_time)
            )
            
            affected_indices = self.data[blink_interval_mask].index
            
            # Cleaning pupil diameter values during blink intervals (ONLY in working copies)
            self.data.loc[affected_indices, self.pupil_left_working] = np.nan
            self.data.loc[affected_indices, self.pupil_right_working] = np.nan
            
            # keep blink information
            self.blink_intervals.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'affected_points': len(affected_indices)
            })
            
            total_cleaned_points += len(affected_indices)
        
        print(f"Cleaned {total_cleaned_points} data points affected by blinks")
        print(f"Percentage of data affected: {total_cleaned_points/len(self.data)*100:.2f}%")
        print(f"Original raw data preserved in: {self.pupil_left_col}, {self.pupil_right_col}")
        
        return True
    
    def interpolate_missing_data(self):
        """
        Interpolating missing dat
        """
        print(f"\nInterpolating missing data ({self.interpolation_method})...")
        
        # Count missing data before interpolation
        missing_left_before = self.data[self.pupil_left_working].isna().sum()
        missing_right_before = self.data[self.pupil_right_working].isna().sum()
        
        print(f"   Missing data before interpolation:")
        print(f"     Left eye: {missing_left_before} points ({missing_left_before/len(self.data)*100:.2f}%)")
        print(f"     Right eye: {missing_right_before} points ({missing_right_before/len(self.data)*100:.2f}%)")
        
        # linear or nearest interpolation
        # # linear 
        if self.interpolation_method == 'linear':
            self.data[self.pupil_left_working] = self.data[self.pupil_left_working].interpolate(method='linear')
            self.data[self.pupil_right_working] = self.data[self.pupil_right_working].interpolate(method='linear')
        else:  # nearest
            self.data[self.pupil_left_working] = self.data[self.pupil_left_working].interpolate(method='nearest')
            self.data[self.pupil_right_working] = self.data[self.pupil_right_working].interpolate(method='nearest')
        
        # Count missing data after interpolation
        missing_left_after = self.data[self.pupil_left_working].isna().sum()
        missing_right_after = self.data[self.pupil_right_working].isna().sum()
        
        print(f"Missing data after interpolation:")
        print(f"Left eye: {missing_left_after} points ({missing_left_after/len(self.data)*100:.2f}%)")
        print(f"Right eye: {missing_right_after} points ({missing_right_after/len(self.data)*100:.2f}%)")
        
        print(f"Interpolation completed using {self.interpolation_method} method")
        
        return True
    
    def baseline_correction(self):
        """
        Baseline correction 
        Uses 0.5 seconds from 10 seconds before experiment end (avoids end artifacts, not sure if this is the best approach)
        """
        print(f"\nBaseline correction ({self.baseline_statistic}_{self.baseline_operation})...")
    
        # Find clean baseline period: 0.5 seconds, starting 10 seconds before end
        max_timestamp = self.data[self.timestamp_col].max()
        baseline_start = max_timestamp - 10000  # 10 seconds before end ????
        baseline_end = baseline_start + 500     # 0.5 second window  ?????
    
        baseline_mask = ((self.data[self.timestamp_col] >= baseline_start) & 
                     (self.data[self.timestamp_col] <= baseline_end))
        baseline_data = self.data[baseline_mask]
    
        print(f"Baseline period: {baseline_start:.0f} - {baseline_end:.0f} ms")
        print(f"(10 seconds before end, 0.5 second duration)")
        print(f"Baseline data points: {len(baseline_data)}")
    
        # Validate baseline period has enough data
        if len(baseline_data) < 20:  # Minimum 10 data points
            print(f"Warning: Only {len(baseline_data)} baseline points found!")
            print(f"Expanding baseline window to 1 second...")
            baseline_end = baseline_start + 1000  # Expand to 1 second
            baseline_mask = ((self.data[self.timestamp_col] >= baseline_start) & 
                            (self.data[self.timestamp_col] <= baseline_end))
            baseline_data = self.data[baseline_mask]
            print(f"Updated baseline data points: {len(baseline_data)}")
    
        # Calculate baseline values for left and right eyes (from working copies)
        if self.baseline_statistic == 'mean':
            baseline_left = baseline_data[self.pupil_left_working].mean()
            baseline_right = baseline_data[self.pupil_right_working].mean()
        else:  # median
            baseline_left = baseline_data[self.pupil_left_working].median()
            baseline_right = baseline_data[self.pupil_right_working].median()
    
        print(f"Baseline {self.baseline_statistic} - Left: {baseline_left:.4f} mm")
        print(f"Baseline {self.baseline_statistic} - Right: {baseline_right:.4f} mm")
    
        # Check for valid baseline values
        if np.isnan(baseline_left) or np.isnan(baseline_right):
            print(f"Error: Baseline calculation resulted in NaN values!")
            print(f"This suggests the baseline period still contains missing data.")
            return False
    
        # Apply baseline correction to working copies
        if self.baseline_operation == 'subtractive':
            # Subtract baseline
            self.data[self.pupil_left_working] = self.data[self.pupil_left_working] - baseline_left
            self.data[self.pupil_right_working] = self.data[self.pupil_right_working] - baseline_right
            print(f"Applied subtractive baseline correction")
        
        else:  # divisive
            # Divide by baseline
            self.data[self.pupil_left_working] = self.data[self.pupil_left_working] / baseline_left
            self.data[self.pupil_right_working] = self.data[self.pupil_right_working] / baseline_right
            print(f"Applied divisive baseline correction")
    
        return True

    
    def apply_lowpass_filter(self):
        """
        Apply lowpass filter to working copies
        """
        if not self.apply_lowpass:
            print(f"\nStep 5: Lowpass filtering skipped (disabled)")
            return True
            
        print(f"\nStep 5: Applying lowpass filter...")
        print(f"Cutoff frequency: {self.lowpass_cutoff} Hz")
        print(f"Sampling rate: {self.sampling_rate:.2f} Hz")
        
        # Check if cutoff frequency is valid
        nyquist_freq = self.sampling_rate / 2
        if self.lowpass_cutoff >= nyquist_freq:
            print(f"Warning: Cutoff frequency ({self.lowpass_cutoff} Hz) is >= Nyquist frequency ({nyquist_freq:.2f} Hz)")
            print(f"Adjusting cutoff to {nyquist_freq * 0.9:.2f} Hz")
            self.lowpass_cutoff = nyquist_freq * 0.9
        
        try:
            # Butterworth lowpass filter (4th order) (not necessary but I put it to just have it!)
            b, a = butter(4, self.lowpass_cutoff, btype='low', fs=self.sampling_rate)
            
            # Apply filter to working copies
            left_data = self.data[self.pupil_left_working].values
            right_data = self.data[self.pupil_right_working].values
            
            # Remove any remaining NaN values for filtering
            left_valid = ~np.isnan(left_data)
            right_valid = ~np.isnan(right_data)
            
            if np.sum(left_valid) > 0:
                filtered_left = np.full_like(left_data, np.nan)
                filtered_left[left_valid] = filtfilt(b, a, left_data[left_valid])
                self.data[self.pupil_left_working] = filtered_left
            
            if np.sum(right_valid) > 0:
                filtered_right = np.full_like(right_data, np.nan)
                filtered_right[right_valid] = filtfilt(b, a, right_data[right_valid])
                self.data[self.pupil_right_working] = filtered_right
            
            print(f"Lowpass filter applied")
            print(f"Filter type: 4th-order Butterworth")
            print(f"Final cutoff frequency: {self.lowpass_cutoff:.2f} Hz")
            
            return True
            
        except Exception as e:
            print(f"Error applying lowpass filter: {e}")
            return False
    
    def create_final_columns(self):
        """
        Create final preprocessed columns with descriptive names
        """
        print(f"\nCreating final preprocessed columns...")
        
        # Copy working data to final columns
        self.data[self.final_left_col] = self.data[self.pupil_left_working].copy()
        self.data[self.final_right_col] = self.data[self.pupil_right_working].copy()
        
        # Remove working columns (cleanup)
        self.data.drop([self.pupil_left_working, self.pupil_right_working], axis=1, inplace=True)
        
        print(f"Final preprocessed columns created:")
        print(f"Left eye: {self.final_left_col}")
        print(f"Right eye: {self.final_right_col}")
        print(f"Working columns cleaned up")
        
        return True
    
    def run_preprocessing(self):
        """
        Run the complete preprocessing pipeline
        """
        print("Starting Pupil Data Preprocessing Pipeline")
        print("=" * 60)
        print(f"Settings:")
        print(f"Data path: {self.data_path}")
        print(f"Baseline method: {self.baseline_method}")
        print(f"Interpolation method: {self.interpolation_method}")
        print(f"Apply lowpass filter: {self.apply_lowpass}")
        if self.apply_lowpass:
            print(f"Lowpass cutoff: {self.lowpass_cutoff} Hz")
        print(f"Final columns will be: {self.final_left_col}, {self.final_right_col}")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Detect and clean blinks
        if not self.detect_and_clean_blinks():
            return False
        
        # Step 3: Interpolate missing data
        if not self.interpolate_missing_data():
            return False
        
        # Step 4: Baseline correction
        if not self.baseline_correction():
            return False
        
        # Step 5: Apply lowpass filter
        if not self.apply_lowpass_filter():
            return False
        
        # Step 6: Create final columns
        if not self.create_final_columns():
            return False
        
        print("\nPreprocessing completed successfully!")
        print(f"Final data shape: {self.data.shape}")
        
        # Show what we have
        print(f"\nAVAILABLE COLUMNS:")
        print(f"Original raw data: {self.pupil_left_col}, {self.pupil_right_col}")
        print(f"preprocessed data: {self.final_left_col}, {self.final_right_col}")
        
        return True
    
    def get_processed_data(self):
        """Get the processed data"""
        return self.data
    
    def get_preprocessed_column_names(self):
        """Get the names of the final preprocessed columns"""
        return {
            'left': self.final_left_col,
            'right': self.final_right_col
        }
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        try:
            self.data.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def save_processed_data_excel(self, output_path):
        """Save processed data to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                self.data.to_excel(writer, sheet_name='ProcessedData', index=False)
                
                # Summary sheet
                summary_data = {
                    'Setting': ['Baseline Method', 'Interpolation Method', 'Lowpass Filter', 
                               'Lowpass Cutoff (Hz)', 'Sampling Rate (Hz)', 'Blink Events',
                               'Left Preprocessed Column', 'Right Preprocessed Column'],
                    'Value': [self.baseline_method, self.interpolation_method, self.apply_lowpass,
                             self.lowpass_cutoff if self.apply_lowpass else 'N/A', 
                             f"{self.sampling_rate:.2f}", len(self.blink_intervals),
                             self.final_left_col, self.final_right_col]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='ProcessingSettings', index=False)
                
                # Blink events sheet
                if self.blink_intervals:
                    blink_df = pd.DataFrame(self.blink_intervals)
                    blink_df.to_excel(writer, sheet_name='BlinkEvents', index=False)
            
            print(f"Processed data saved to Excel: {output_path}")
            print(f"Sheets: ProcessedData, ProcessingSettings, BlinkEvents")
            return True
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            return False
    
    