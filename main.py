from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
from datetime import datetime

app = FastAPI()


def calculate_psd(signal, sampling_rate, freq_range):
    fft_result = np.fft.fft(signal)
    magnitudes = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    psd = np.mean(magnitudes[mask] ** 2)
    return psd


@app.post("/EEG")
async def process_EEG(chunks_json: str = Form(...), file: UploadFile = File(...)):
    print(f"Received chunks_json: {chunks_json}")
    print(f"Received file: {file.filename}")
    try:
        chunks = json.loads(chunks_json)["chunks"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing chunks_json: {str(e)}")
        return JSONResponse(content={"error": f"Invalid chunks_json: {str(e)}"}, status_code=400)

    try:
        df = pd.read_csv(file.file, sep=';')
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {str(e)}")
        return JSONResponse(content={"error": f"Error parsing CSV file: {str(e)}. Please check the file format and structure."}, status_code=400)

    timestamp_column = ' Timestamp'
    eeg_columns = ['EEG Channel1', 'EEG Channel2', 'EEG Channel3', 'EEG Channel4']

    if timestamp_column not in df.columns:
        print("Timestamp column not found in the CSV file.")
        return JSONResponse(content={"error": "Timestamp column not found in the CSV file."}, status_code=400)

    for column in eeg_columns:
        if column not in df.columns:
            print(f"EEG channel column {column} not found in the CSV file.")
            return JSONResponse(content={"error": f"EEG channel column {column} not found in the CSV file."}, status_code=400)

    try:
        df[timestamp_column] = df[timestamp_column].str.strip()  # Remove leading space from timestamp values
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='%H:%M:%S.%f')
    except ValueError as e:
        print(f"Error parsing timestamp column: {str(e)}")
        return JSONResponse(content={"error": f"Error parsing timestamp column: {str(e)}. Please check the timestamp format."}, status_code=400)

    results = []
    sampling_rate = 250  # Assuming a sampling rate of 250 Hz

    for chunk_data in chunks:
        start, end = chunk_data['timestamp']
        text = chunk_data['text']
        start_time = pd.to_datetime(start, unit='s').strftime('%H:%M:%S.%f')
        end_time = pd.to_datetime(end, unit='s').strftime('%H:%M:%S.%f')
        df_chunk = df[(df[timestamp_column] >= start_time) & (df[timestamp_column] <= end_time)]

        if len(df_chunk) == 0:
            print(f"No data found for timestamp range: {start_time} - {end_time}")
            results.append({
                "timestamp": [start, end],
                "text": text,
                "emotion": "neutral/calm"
            })
            continue

        try:
            alpha_psd = np.mean([calculate_psd(df_chunk[channel], sampling_rate, [8, 12]) for channel in eeg_columns])
            beta_psd = np.mean([calculate_psd(df_chunk[channel], sampling_rate, [12, 30]) for channel in eeg_columns])
            arousal = beta_psd / alpha_psd

            valence_alpha_psd = calculate_psd(df_chunk['EEG Channel3'], sampling_rate, [8, 12])
            valence_beta_psd = calculate_psd(df_chunk['EEG Channel4'], sampling_rate, [12, 30])
            valence = valence_alpha_psd - valence_beta_psd

            if arousal > 1 and valence > 0:
                emotion = 'Happy'
            elif arousal > 1 and valence < 0:
                emotion = 'Angry'
            elif arousal < 1 and valence > 0:
                emotion = 'Calm/Neutral'
            elif arousal < 1 and valence < 0:
                emotion = 'Sad'
            else:
                emotion = 'neutral/calm'
        except (IndexError, ValueError) as e:
            print(f"Error processing EEG data: {str(e)}")
            emotion = 'neutral/calm'

        results.append({
            "timestamp": [start, end],
            "text": text,
            "emotion": emotion
        })

    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)