import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    # cell 1: imports
    import numpy as np
    import pandas as pd
    import scipy.signal as signal
    import marimo as mo

    import altair as alt
    import plotly.express as px
    return alt, mo, np, pd, signal


@app.cell
def _(mo):
    plot_library = mo.ui.radio(["Altair", "Plotly"], label="Plotting Library")
    plot_library
    return (plot_library,)


@app.cell
def _(c, d, np, order_high, order_low, pd, signal):
    def apply_filter(data, fs, filter_type, lowcut, highcut, order, order_low = 1, order_high = 1):
        nyq = 0.5 * fs
        if filter_type == "bandpass":
            b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        elif filter_type == "lowpass":
            b, a = signal.butter(order_low, highcut / nyq, btype='low')
        elif filter_type == "highpass":
            b, a = signal.butter(order_high, lowcut / nyq, btype='high')
        else:
            raise ValueError("Invalid filter type")
        return signal.filtfilt(b, a, data)


        # apply low-pass and high-pass butterworth filters to GSR and ST signals
    def lowpass_filter_signal(data, cutoff_freq, order = 1):
        """
        Takes pandas DataFrame containing GSR or ST measurements,
         applies high-pass and low-pass filters to GSR and ST signals
         and return filtered pandas DataFrame containing physiological signals
        :param data: pandas DF containing raw physiological data
        :param phys_signal: string representing the signal to apply the filters on ("GSR", "ST", etc.)

        :return: DataFrame containing filtered physiological signals
        """

        if order_low == order_high:
            order = order_low

            b, a = signal.butter(order, cutoff_freq, 'low', analog=False)


        # new version -- replace signal.filtfilt() with signal.lfilter()

        # old filter (causes drops in GSR signal before MOS) --> filters signal twice
        # z = signal.filtfilt(b, a, data.value_real)
        # filteredGSR = signal.filtfilt(c, d, z)

        # new filter --> filters signal only once
        z = signal.lfilter(b, a, data)
        lowpass_filtered_signal = signal.lfilter(c, d, z)
        # signal.filtfilt(b, a, data)

        #data['value_real'] = filteredGSR

        return lowpass_filtered_signal

    # apply low-pass and high-pass butterworth filters to signals

    def bandpass_filter_signal(data, lowcut, highcut, order_low = 1, order_high = 1, sampling_frequency = 4):

        """

        Takes pandas DataFrame containing GSR or ST measurements,

         applies high-pass and low-pass filters to GSR and ST signals

         and return filtered pandas DataFrame containing physiological signals

        :param data: pandas DF containing raw physiological data

        :param phys_signal: string representing the signal to apply the filters on ("GSR", "ST", etc.)

        :return: DataFrame containing filtered physiological signals

        """

        nyq = 0.5 * sampling_frequency

        if order_low == order_high:

            order = order_low


            b, a = signal.butter(order, lowcut / nyq, 'low', analog=False)

            c, d = signal.butter(order, highcut / nyq, "high", analog=False)

        else:

            b, a = signal.butter(order_low, lowcut / nyq, 'low', analog=False)

            c, d = signal.butter(order_high, highcut / nyq, "high", analog=False)

        # new version -- replace signal.filtfilt() with signal.lfilter()

        # old filter (causes drops in GSR signal before MOS) --> filters signal twice

        # z = signal.filtfilt(b, a, data.value_real)

        # filteredGSR = signal.filtfilt(c, d, z)

        # new filter --> filters signal only once

        z = signal.lfilter(b, a, data)

        filtered_signal = signal.lfilter(c, d, z)

        # signal.filtfilt(b, a, data)

        #data['value_real'] = filteredGSR

        return filtered_signal


    # frequency domain plot
    def compute_spectrum(sig, fs):
        freqs = np.fft.rfftfreq(len(sig), 1/fs)
        spectrum = np.abs(np.fft.rfft(sig))
        return pd.DataFrame({"Frequency": freqs, "Amplitude": spectrum})

    # frequency domain with windowing
    def apply_window(sig, kind):
        if kind == "Hamming":
            return sig * np.hamming(len(sig))
        elif kind == "Hanning":
            return sig * np.hanning(len(sig))
        elif kind == "Blackman":
            return sig * np.blackman(len(sig))
        else:
            return sig
    return apply_window, bandpass_filter_signal, compute_spectrum


@app.cell
def _(pd):
    file_path =  "/Users/annapalatkina/Desktop/drive/driving_performance_project/notebooks/data_full_biopac.csv"

    data = pd.read_csv(file_path)

    data
    return (data,)


@app.cell
def _(data, mo):

    participant_column = mo.ui.dropdown(data.columns, label = "Select Participant ID Column")

    participant_selection = mo.ui.dropdown(list(data["Participant"]), label="Select one of the participants")

    fs_input = mo.ui.number(200, label="Sampling Frequency (Hz)")

    column_select = mo.ui.dropdown(data.columns, label="Select Signal Column")

    time_column = mo.ui.dropdown(data.columns, label = "Select Time Column")

    filter_type = mo.ui.radio(["bandpass", "lowpass", "highpass"], label="Filter Type")

    low_cut = mo.ui.slider(0.00, 1.00, value=1.0, label="Low Cutoff Frequency (Hz)", step = 0.01)

    filter_order_low = mo.ui.slider(1, 10, value=1, label="Low Pass Filter Order")

    high_cut = mo.ui.slider(0.00, 1.00, value=0.05, label="High Cutoff Frequency (Hz)", step = 0.01)

    filter_order_high = mo.ui.slider(1, 10, value=1, label="High Pass Filter Order")

    # cell 2: extended UI
    window_fn = mo.ui.dropdown(
        ["None", "Hamming", "Hanning", "Blackman"], value="None", label="FFT Window"
    )

    downsample_factor = mo.ui.slider(1, 20, value=1, label="Downsampling Factor")

    return (
        column_select,
        downsample_factor,
        filter_order_high,
        filter_order_low,
        fs_input,
        high_cut,
        low_cut,
        participant_column,
        participant_selection,
        window_fn,
    )


@app.cell
def _(
    column_select,
    downsample_factor,
    filter_order_high,
    filter_order_low,
    fs_input,
    high_cut,
    low_cut,
    mo,
    participant_column,
    participant_selection,
    plot_library,
    window_fn,
):
    mo.sidebar(mo.vstack([
        participant_column,
        participant_selection,
        fs_input,
        column_select,
        #filter_type,
        low_cut,
        filter_order_low,
        high_cut, 
        filter_order_high, 
        #mo.hstack([low_cut, filter_order_low, high_cut, filter_order_high]),
        window_fn,
        downsample_factor,
        plot_library
    ]))
    return


@app.cell
def _(
    downsample_factor,
    filter_order_high,
    filter_order_low,
    high_cut,
    low_cut,
):
    print(f"Low Cutoff Frequency is set to {low_cut.value} Hz")
    print(f"Low pass filter order is set to {filter_order_low.value}")

    print(f"High Cutoff Frequency is set to {high_cut.value} Hz")
    print(f"High pass filter order is set to {filter_order_high.value}")

    print(f"Downsample factor is: {downsample_factor.value}")
    return


@app.cell
def _(column_select, data):
    print(f"Number of values before dropping NAs {len(data[column_select.value].values)}")
    print(f"Number of values after dropping NAs {len(data[column_select.value].dropna().values)}")
    data[column_select.value].dropna().values
    return


@app.cell
def _(column_select, data, participant_column, participant_selection):
    signal_data = data[data[participant_column.value] == participant_selection.value]
    signal_data = signal_data[column_select.value].dropna().values
    signal_data
    return (signal_data,)


@app.cell
def _(downsample_factor, fs_input, signal_data):
    # Downsampling
    factor = downsample_factor.value

    if factor > 1:
        signal_data[::factor]
        fs = fs_input.value / factor
    else: 
        fs = fs_input.value

    fs 
    return (fs,)


@app.cell
def _(fs):
    fs 
    return


@app.cell
def _(fs, np, signal_data):
    t = np.arange(len(signal_data)) / fs
    t
    return (t,)


@app.cell
def _(column_select, pd, signal_data, t):
    df = pd.DataFrame({"Time": t, column_select.value: signal_data})
    df
    return (df,)


@app.cell
def _(filter_order_high, filter_order_low):
    if filter_order_high == filter_order_low:
        order = filter_order_high.value
    return


@app.cell
def _(
    bandpass_filter_signal,
    df,
    filter_order_high,
    filter_order_low,
    fs_input,
    high_cut,
    low_cut,
    signal_data,
):
    filtered_signal = bandpass_filter_signal(
        data = signal_data,
        lowcut = low_cut.value,
        highcut = high_cut.value,
        order_low = filter_order_low.value,
        order_high = filter_order_high.value,
        sampling_frequency = fs_input.value
    )

    df["Filtered"] = filtered_signal
    df
    return (filtered_signal,)


@app.cell
def _(apply_window, signal_data, window_fn):
    sig_win = apply_window(signal_data, window_fn.value)
    sig_win
    return (sig_win,)


@app.cell
def _(apply_window, filtered_signal, window_fn):
    filt_win = apply_window(filtered_signal, window_fn.value)
    filt_win
    return (filt_win,)


@app.cell
def _(compute_spectrum, fs, sig_win):
    spec_orig = compute_spectrum(sig_win, fs)
    spec_orig
    return (spec_orig,)


@app.cell
def _(compute_spectrum, filt_win, fs):
    spec_filt = compute_spectrum(filt_win, fs)
    spec_filt
    return (spec_filt,)


@app.cell
def _(alt, mo, plot_library, spec_filt, spec_orig):
    if plot_library.value == "Altair":
        orig_plot = alt.Chart(spec_orig).mark_line(color='gray').encode(
            x="Frequency", y="Amplitude"
        ).properties(title="Original Spectrum").interactive()

        filt_plot = alt.Chart(spec_filt).mark_line(color='green').encode(
            x="Frequency", y="Amplitude"
        ).properties(title="Filtered Spectrum").interactive()

    mo.ui.altair_chart(orig_plot & filt_plot)

    return


@app.cell
def _(alt, column_select, df, mo, plot_library):
    if plot_library.value == "Altair":
        raw_plot = alt.Chart(df).mark_line(color='gray').encode(
            x="Time", y=column_select.value
        ).properties(title="Original Signal").interactive()

        filtered_plot = alt.Chart(df).mark_line(color='green').encode(
            x="Time", y="Filtered"
        ).properties(title="Filtered Signal").interactive()

    mo.ui.altair_chart(raw_plot & filtered_plot)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, participant_selection):
    df[['Filtered']].to_csv(f'/Users/annapalatkina/Desktop/drive/marimo_results/EDA_biopac/{participant_selection.value}.csv', index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
