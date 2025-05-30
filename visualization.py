import matplotlib.pyplot as plt
import seaborn as sns

def plot_column(data, column='VelocityX', unit='m', sections=None):

    if not sections:
        print('plot_column requires sections to equal the audio ranges')
        return
    
    before_audio_start = sections['before'][0]
    before_audio_end = sections['before'][1]

    calm_audio_start = sections['calm'][0]
    calm_audio_end = sections['calm'][1]

    interim_audio_start = sections['interim'][0]
    interim_audio_end = sections['interim'][1]

    intense_audio_start = sections['intense'][0]
    intense_audio_end = sections['intense'][1]

    after_audio_start = sections['after'][0]
    after_audio_end = sections['after'][1]


    
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    # Plot the fixation duration over time
    plt.plot(data['Timestamp'], data[column],
            color='#3498db', linewidth=1.5, alpha=0.7)

    colors = {
        'Before': '#D3D3D3',  # Light gray
        'Positive': '#90CAF9',    # Light blue
        'Interim': '#FFD54F', # Light amber
        'Negative': '#EF5350', # Light red
        'After': '#D3D3D3'    # Light gray
    }

    # Add colored background regions for each audio segment
    y_min, y_max = plt.ylim()
    plt.axvspan(before_audio_start['Timestamp'], before_audio_end['Timestamp'],
                alpha=0.2, color=colors['Before'], label='Before Audio')
    plt.axvspan(calm_audio_start['Timestamp'], calm_audio_end['Timestamp'],
                alpha=0.2, color=colors['Positive'], label='Positive Audio')
    plt.axvspan(interim_audio_start['Timestamp'], interim_audio_end['Timestamp'],
                alpha=0.2, color=colors['Interim'], label='Interim Audio')
    plt.axvspan(intense_audio_start['Timestamp'], intense_audio_end['Timestamp'],
                alpha=0.2, color=colors['Negative'], label='Negative Audio')
    plt.axvspan(after_audio_start['Timestamp'], after_audio_end['Timestamp'],
                alpha=0.2, color=colors['After'], label='After Audio')

    # Add vertical lines at transition points
    plt.axvline(x=before_audio_start['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=calm_audio_end['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=calm_audio_start['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=calm_audio_end['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=interim_audio_start['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=interim_audio_end['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=intense_audio_start['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=after_audio_start['Timestamp'], color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=after_audio_end['Timestamp'], color='black', linestyle='--', alpha=0.7)

    # Add titles and labels
    plt.title(f'{column} During Different Audio Segments', fontsize=16, pad=20)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel(f'{column} ({unit})', fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # Improve x-axis readability by converting to relative time
    relative_timestamps = []

    plt.xticks([t[1] for t in relative_timestamps],
            [f"{t[0]}\n{t[2]:.1f}s" for t in relative_timestamps],
            rotation=45)

    plt.tight_layout()