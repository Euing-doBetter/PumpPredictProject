import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Bidirectional, LSTM, TimeDistributed, Flatten, Reshape
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 파일 경로 정의 (실제 파일 경로로 수정해야 합니다)
flow_10min_x_path = 'C:/Users/euing/Desktop/펌프장 유입량 예측 AI모델 입력자료(수정)/Train/독립변수/전처리데이터1/Flow_df_10min.csv'
rain_abs_path = 'C:/Users/euing/Desktop/펌프장 유입량 예측 AI모델 입력자료(수정)/Train/독립변수/전처리데이터1/Rain_df(3hr)_X(전처리파일일).csv'
suwi_merged_path = 'C:/Users/euing/Desktop/펌프장 유입량 예측 AI모델 입력자료(수정)/Train/독립변수/전처리데이터1/합쳐진모니터링수위데이터_처음19행.csv' # 파일 이름 확인 및 수정 필요
flow_10min_y_path = 'C:/Users/euing/Desktop/펌프장 유입량 예측 AI모델 입력자료(수정)/Train/종속변수/Flow_df_10min_Y.csv'

# 결과를 저장할 디렉토리 생성
results_dir = 'C:/Users/euing/Desktop/펌프장 유입량 예측 AI모델 입력자료(수정)/Train/결과_BiLSTM' # BiLSTM 결과 저장 폴더
os.makedirs(results_dir, exist_ok=True) # 디렉토리가 없으면 생성

# 파일 로드 함수
def load_csv_file(filepath, encoding='utf-8-sig'):
    """CSV 파일을 여러 인코딩으로 시도하여 로드하는 함수"""
    encodings = [encoding, 'cp949', 'euc-kr', 'utf-8'] # 시도할 인코딩 목록
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"'{filepath}' 파일을 '{enc}' 인코딩으로 성공적으로 읽었습니다.")
            return df
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다: '{filepath}'")
            return None
        except UnicodeDecodeError:
            print(f"'{filepath}' 파일을 '{enc}' 인코딩으로 읽는데 실패했습니다. 다른 인코딩 시도...")
        except Exception as e:
            print(f"'{filepath}' 파일 로드 중 오류 발생: {e}")
            return None
    print(f"오류: '{filepath}' 파일을 지원되는 인코딩으로 읽을 수 없습니다.")
    return None

# 수문학 평가지표 계산 함수 정의
def calculate_hydrological_metrics(observed, predicted):
    """수문학 평가지표 (ME, RMSE, PBIAS, NSE, VE, KGE)를 계산합니다."""
    observed = np.asarray(observed).flatten()
    predicted = np.asarray(predicted).flatten()

    # Remove NaN/Inf values if any
    valid_indices = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid_indices]
    predicted = predicted[valid_indices]

    if len(observed) == 0:
        return {
            'ME': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan,
            'NSE': np.nan, 'VE': np.nan, 'KGE': np.nan
        }

    mean_obs = np.mean(observed)
    mean_pred = np.mean(predicted)
    std_obs = np.std(observed)
    std_pred = np.std(predicted)
    correlation = np.corrcoef(observed, predicted)[0, 1] if len(observed) > 1 else np.nan

    # Mean Error (ME)
    me = np.mean(predicted - observed)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(observed, predicted))

    # Percent Bias (PBIAS)
    pbias = 100 * np.sum(predicted - observed) / np.sum(observed) if np.sum(observed) != 0 else np.nan

    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - np.sum((observed - predicted)**2) / np.sum((observed - mean_obs)**2) if np.sum((observed - mean_obs)**2) != 0 else np.nan

    # Volumetric Efficiency (VE)
    ve = 1 - np.sum(np.abs(predicted - observed)) / np.sum(observed) if np.sum(observed) != 0 else np.nan # Simplified VE calculation

    # Kling-Gupta Efficiency (KGE - 2012 version)
    # Components: correlation, ratio of means, ratio of standard deviations
    alpha = std_pred / std_obs if std_obs != 0 else np.nan
    beta = mean_pred / mean_obs if mean_obs != 0 else np.nan
    kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not (np.isnan(correlation) or np.isnan(alpha) or np.isnan(beta)) else np.nan


    return {
        'ME': me, 'RMSE': rmse, 'PBIAS': pbias,
        'NSE': nse, 'VE': ve, 'KGE': kge
    }


# 1. 데이터 로드
df_flow_x = load_csv_file(flow_10min_x_path)
df_rain = load_csv_file(rain_abs_path)
df_suwi = load_csv_file(suwi_merged_path) # 수위 데이터
df_flow_y = load_csv_file(flow_10min_y_path)

# 모든 파일이 성공적으로 로드되었는지 확인
if df_flow_x is None or df_rain is None or df_suwi is None or df_flow_y is None:
    print("\n필수 데이터 파일 로드에 실패했습니다. 경로 및 파일 내용을 확인해주세요.")
    # exit() # 실제 스크립트에서는 여기서 종료할 수 있습니다.
else:
    print("\n모든 데이터 파일 로드 성공.")
    print(f"Flow_X shape: {df_flow_x.shape}") # (18, 891)
    print(f"Rain shape: {df_rain.shape}") # (18, 891) 예상
    print(f"Suwi shape: {df_suwi.shape}") # (1980, X) 예상
    print(f"Flow_Y shape: {df_flow_y.shape}") # (18, 891)

    # 데이터 shape 확인 및 일관성 체크
    num_samples = df_flow_x.shape[0] # 18
    past_sequence_length = df_flow_x.shape[1] # 891
    future_sequence_length = df_flow_y.shape[0] # 18
    num_output_targets = df_flow_y.shape[1] # 891

    if not (num_samples == df_rain.shape[0] and past_sequence_length == df_rain.shape[1]):
        print("\n오류: Flow_X와 Rain 데이터 shape 불일치.")
        # exit()
    if not num_samples == df_suwi.shape[0]:
         print(f"경고: Suwi 데이터의 행 수({df_suwi.shape[0]})가 독립변수 샘플 수({num_samples})와 일치하지 않습니다. 데이터 재구성에 주의합니다.")
         # exit()


    # --- 데이터 재구성 (사용자 최종 설명 기반) ---
    # 샘플: 891개 (Flow/Rain/Flow_Y 열 개수)
    # 입력 시퀀스 길이: 18 (Flow/Rain 행 개수)
    # 출력 시퀀스 길이: 18 (Flow_Y 행 개수)
    # 예측 대상 개수: 891 (Flow_Y 열 개수)
    # 입력 피처: Flow(1) + Rain(1) + Suwi(num_suwi_features)

    num_samples_actual = df_flow_x.shape[1] # 891
    past_sequence_length_actual = df_flow_x.shape[0] # 18
    future_sequence_length_actual = df_flow_y.shape[0] # 18
    num_output_targets_actual = df_flow_y.shape[1] # 891
    num_suwi_features = df_suwi.shape[1] # Suwi 특성 개수

    # Flow_X, Rain, Flow_Y 데이터를 Transpose하여 (샘플 수, 시퀀스 길이) 형태로 만듦
    df_flow_x_T = df_flow_x.T # Shape (891, 18)
    df_rain_T = df_rain.T # Shape (891, 18)
    df_flow_y_T = df_flow_y.T # Shape (891, 18)


    # 데이터 스케일링 준비
    scaler_flow = MinMaxScaler()
    scaler_rain = MinMaxScaler()
    scaler_suwi = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Transpose된 Flow_X와 Rain 데이터를 스케일링
    df_flow_x_T_scaled = scaler_flow.fit_transform(df_flow_x_T) # Shape (891, 18)
    df_rain_T_scaled = scaler_rain.fit_transform(df_rain_T) # Shape (891, 18)

    # Suwi 데이터 스케일링 (Suwi 전체 데이터 사용 가정)
    df_suwi_scaled = scaler_suwi.fit_transform(df_suwi) # Shape (1980 or 18, num_suwi_features)

    # Suwi 데이터를 891개 샘플 각각의 18개 시점에 반복하여 붙이기 (임시 가정)
    # Suwi 데이터의 실제 사용 방식에 따라 수정 필요.
    # 여기서는 df_suwi의 첫 18행만 사용하고, 각 샘플(891)의 각 시점(18)에 동일하게 반복한다고 가정.
    suwi_repeated = np.tile(df_suwi_scaled[:past_sequence_length_actual, :], (num_samples_actual, 1, 1))
    # suwi_repeated shape: (891, 18, num_suwi_features)


    # 독립 변수 데이터 재구성: X shape (891, 18, num_features)
    # 각 샘플(891)의 각 시점(18)에 대해 Flow, Rain, Suwi 피처를 결합
    # Flow_X_T_scaled shape (891, 18), Rain_T_scaled shape (891, 18)
    # Suwi_repeated shape (891, 18, num_suwi_features)

    # Flow와 Rain 데이터를 (891, 18, 1) 형태로 확장하여 Suwi 데이터와 결합 준비
    df_flow_x_T_scaled_expanded = np.expand_dims(df_flow_x_T_scaled, axis=-1) # Shape (891, 18, 1)
    df_rain_T_scaled_expanded = np.expand_dims(df_rain_T_scaled, axis=-1) # Shape (891, 18, 1)

    # X_data 결합: (891, 18, 1) + (891, 18, 1) + (891, 18, num_suwi_features)
    X_data = np.concatenate([df_flow_x_T_scaled_expanded, df_rain_T_scaled_expanded, suwi_repeated], axis=-1)
    # X_data shape: (891, 18, 2 + num_suwi_features)


    # 종속 변수 데이터 재구성: Y shape (891, 18, 891) <-- 목표 shape (샘플, 미래스텝, 예측대상)
    # df_flow_y_T shape (891, 18) -> 예측 대상 개수 891에 맞추기
    # df_flow_y_T의 각 열(18개 값)이 891개 예측 대상 중 하나에 대한 미래 18스텝 시퀀스라고 해석.
    # 또는 df_flow_y_T의 각 행(891개 값)이 미래 18스텝 중 하나의 시점의 891개 예측 대상 값이라고 해석.
    # 사용자 설명: df_flow_y 행(18)=미래스텝, 열(891)=예측대상. 샘플 수 18. -> Y shape (18, 18, 891)
    # 사용자 추가 설명: Flow_df_10min_Y 의 종속변수에 해당하는 1번째 열이 예측값으로 적용
    # 이 설명은 Y shape (18, 891)에서 1번째 열(길이 18)을 예측값 시퀀스로 사용한다는 의미일 수 있음.
    # 즉, 891개 샘플 각각에 대해 18개 길이의 시퀀스 예측값 1개가 필요. -> Y shape (891, 18, 1)

    # 사용자님의 최종 설명 "Flow_df_10min,Rain_df(3hr)_X(전처리파일일) 각 열을 1대1대응으로 학습하면 Flow_df_10min_Y 의 종속변수에 해당하는 1번째 열이 예측값으로 적용되는거지"를 따릅니다.
    # 샘플=891 (열), 과거 시퀀스=18 (행), 미래 시퀀스=18 (행), 예측 대상=1 (Flow_Y 1번째 열)

    # Y 데이터 재구성: Y shape (891, 18, 1)
    # df_flow_y_T (891, 18)의 1번째 열만 사용
    y_data_T = df_flow_y_T.iloc[:, 0].values # Shape (891,)
    y_data_T = np.expand_dims(y_data_T, axis=-1) # Shape (891, 1) - 이것도 미래 시퀀스 18에 안 맞음.

    # 다시 사용자 설명 해석: "Flow_df_10min_Y 의 종속변수에 해당하는 1번째 열이 예측값으로 적용되는거지"
    # Flow_df_10min_Y (18, 891). 1번째 열은 길이 18의 시퀀스.
    # 891개 샘플 각각에 대해 이 18개 길이의 시퀀스를 예측해야 한다 -> Y shape (891, 18)
    # 예측 대상 개수 1개 -> Y shape (891, 18, 1)

    # Y 데이터 재구성: Y shape (891, 18, 1)
    # df_flow_y_T (891, 18). 각 행은 891개 샘플 중 하나의 미래 18스텝 타겟 시퀀스.
    # 이를 (891, 18, 1) 형태로 만들어야 함. 즉, 마지막 차원이 1인 3차원 배열.
    # df_flow_y_T shape (891, 18)은 이미 샘플=891, 미래스텝=18 형태. 예측 대상 1개.
    y_data = np.expand_dims(df_flow_y_T.values, axis=-1) # Shape (891, 18, 1)


    # 데이터 스케일링 (Y 데이터)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).reshape(y_data.shape) # Shape (891, 18, 1)


    print("\n데이터 재구성 완료.")
    print(f"재구성된 독립 변수 shape: {X_data.shape} ([samples, past_sequence_length, features])") # (891, 18, 2 + num_suwi_features)
    print(f"재구성된 종속 변수 shape: {y_scaled.shape} ([samples, future_sequence_length, output_targets])") # (891, 18, 1)


    # 3. 데이터 분할 (학습 세트와 테스트 세트)
    # 샘플 수 891개
    train_size = int(num_samples_actual * 0.8) # 891개 중 80% = 712개 학습, 179개 테스트
    if train_size == 0: train_size = 1 # 최소 1개 샘플은 학습
    X_train, X_test = X_data[0:train_size, :, :], X_data[train_size:num_samples_actual, :, :]
    y_train, y_test = y_scaled[0:train_size, :, :], y_scaled[train_size:num_samples_actual, :, :]

    print(f"\n학습 데이터 shape (X_train, y_train): {X_train.shape}, {y_train.shape}") # (712, 18, 6), (712, 18, 1)
    print(f"테스트 데이터 shape (X_test, y_test): {X_test.shape}, {y_test.shape}") # (179, 18, 6), (179, 18, 1)


    # 4. BiLSTM 모델 구축
    def build_bilstm_model(input_shape, future_sequence_length, num_output_targets, lstm_units=50, dropout=0.3):
        inputs = Input(shape=input_shape) # input_shape: (과거 시퀀스 길이, 피처 수) = (18, 6)

        # BiLSTM 레이어 (return_sequences=True로 시퀀스 출력 유지)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(inputs)
        x = Dropout(dropout)(x)

        # 추가 BiLSTM 레이어 (선택 사항)
        # x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
        # x = Dropout(dropout)(x)

        # TimeDistributed Dense 레이어: 각 시점의 출력에 대해 독립적인 Dense 적용
        # TCN 출력 shape (samples, 과거 시퀀스 길이, units) -> 목표 (samples, 미래 시퀀스 길이, 예측 대상)
        # BiLSTM 출력 shape (samples, 과거 시퀀스 길이, 2*units) -> 목표 (samples, 미래 시퀀스 길이, 예측 대상)

        # 시퀀스 길이 18 -> 18 유지, 피처 수 2*units -> 예측 대상 1개로 변환
        # BiLSTM 마지막 레이어는 (samples, 18, 2*units) shape.
        # TimeDistributed(Dense(num_output_targets))를 적용하면 (samples, 18, num_output_targets) shape.
        outputs = TimeDistributed(Dense(num_output_targets))(x) # num_output_targets = 1

        return Model(inputs, outputs)

    # 모델 생성
    num_features_actual = X_data.shape[-1] # 6
    model = build_bilstm_model(
        input_shape=(past_sequence_length_actual, num_features_actual), # (18, 6)
        future_sequence_length=future_sequence_length_actual, # 18 (모델 출력 shape에는 직접 사용되지 않음)
        num_output_targets=1, # 예측 대상 1개
        lstm_units=50,
        dropout=0.3
    )

    # 콜백 함수 정의 (과적합 방지를 위해 유지)
    early_stopping = EarlyStopping(
        monitor='val_loss', # 또는 'val_mae'
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', # 또는 'val_mae'
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    # 5. 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',  # MAE 손실 함수 사용
        metrics=['mae', 'mse'] # MAE와 MSE 모두 모니터링
    )

    # 모델 요약 정보 출력
    model.summary()

    # 6. 모델 학습
    print("\n모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # 학습 데이터의 20%를 검증에 사용
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("모델 학습 완료.")


    # 7. 모델 평가 및 결과 시각화
    print("\n모델 평가 및 결과 시각화 시작...")

    # 7.1. 학습/검증 MAE 및 손실 그래프
    plt.figure(figsize=(12, 6))

    # MAE 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE over Epochs (BiLSTM)')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 손실 (MAE) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss (MAE)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
    plt.title('Model Loss (MAE) over Epochs (BiLSTM)')
    plt.ylabel('Loss (MAE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    mae_loss_graph_path = os.path.join(results_dir, 'mae_loss_over_epochs_BiLSTM.png')
    plt.savefig(mae_loss_graph_path)
    print(f"MAE 및 Loss 그래프가 '{mae_loss_graph_path}' 파일로 저장되었습니다.")
    # plt.show() # 학습 서버에서는 주석 처리


    # 7.2. 최종 평가지표 계산 및 출력
    # evaluate는 스케일링된 데이터로 계산됩니다.
    print("\n--- 최종 평가지표 (Scaled Data) ---")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_mae, train_mse = model.evaluate(X_train, y_train, verbose=0)

    print(f"{'Metric':<15} | {'Train':<10} | {'Test':<10}")
    print("-" * 40)
    print(f"{'Loss (MAE)':<15} | {train_loss:<10.4f} | {test_loss:<10.4f}")
    print(f"{'MAE':<15} | {train_mae:<10.4f} | {test_mae:<10.4f}")
    print(f"{'MSE':<15} | {train_mse:<10.4f} | {test_mse:<10.4f}")
    # R2 Score는 보통 inverse_transform 후 계산합니다.
    print("-" * 40)


    y_pred_scaled = model.predict(X_test)

    # 예측 결과를 원래 스케일로 되돌립니다.
    # y_test_original shape: (test_samples, 18, 1)
    # y_pred_original shape: (test_samples, 18, 1)

    # Inverse transform을 위해 데이터를 평탄화합니다.
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    # 원래 스케일에서의 평가지표 계산 (Flatten 후 계산)
    # R2 Score는 (샘플 * 시퀀스) 길이의 평탄화된 데이터로 계산
    print("\n--- 최종 평가지표 (Original Scale) ---")
    original_mae = mean_absolute_error(y_test_original.flatten(), y_pred_original.flatten())
    original_mse = mean_squared_error(y_test_original.flatten(), y_pred_original.flatten())
    original_r2 = r2_score(y_test_original.flatten(), y_pred_original.flatten())

    # 추가 수문학 평가지표 계산
    hydrological_metrics = calculate_hydrological_metrics(y_test_original.flatten(), y_pred_original.flatten())

    # 평가지표 결과를 DataFrame으로 만들고 CSV로 저장
    metrics_data = {
        'Metric': ['MAE', 'MSE', 'R2', 'ME', 'RMSE', 'PBIAS', 'NSE', 'VE', 'KGE'],
        'Value': [
            original_mae, original_mse, original_r2,
            hydrological_metrics['ME'], hydrological_metrics['RMSE'], hydrological_metrics['PBIAS'],
            hydrological_metrics['NSE'], hydrological_metrics['VE'], hydrological_metrics['KGE']
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # CSV 파일 저장 경로 (필요에 따라 수정하세요)
    csv_save_path = os.path.join(results_dir, "evaluation_results_BiLSTM.csv")
    metrics_df.to_csv(csv_save_path, index=False)

    print(f"\n평가지표 결과가 '{csv_save_path}' 파일로 저장되었습니다.")


    # 7.3. 결과 시계열 그래프 (실제값 vs 예측값)
    # 테스트 세트의 첫 번째 샘플에 대한 예측 결과를 시각화합니다.
    print("\n결과 시계열 그래프 생성 중 (테스트 세트 첫 샘플)... ")

    if X_test.shape[0] > 0: # 테스트 샘플이 있을 경우
        sample_to_plot = 0 # 첫 번째 테스트 샘플

        plt.figure(figsize=(15, 7))

        # 실제 값 (테스트 세트 첫 샘플의 18개 미래 스텝 값)
        plt.plot(y_test_original[sample_to_plot, :, 0], label=f'Observed (Sample {sample_to_plot+1})', color='black')
        # 예측 값 (테스트 세트 첫 샘플의 18개 미래 스텝 값)
        plt.plot(y_pred_original[sample_to_plot, :, 0], label=f'BiLSTM Prediction (Sample {sample_to_plot+1})', color='red')

        plt.title(f'Observed vs BiLSTM Prediction Time Series (Sample {sample_to_plot+1})')
        plt.xlabel('Future Time Step (10-min intervals)')
        plt.ylabel('Flow (Original Scale)')
        plt.legend()
        plt.grid(True)

        timeseries_graph_path = os.path.join(results_dir, 'observed_vs_predicted_timeseries_BiLSTM.png')
        plt.savefig(timeseries_graph_path)
        print(f"결과 시계열 그래프가 '{timeseries_graph_path}' 파일로 저장되었습니다.")
        # plt.show() # 학습 서버에서는 주석 처리
    else:
        print("테스트 샘플이 없어 시계열 그래프를 그릴 수 없습니다.")

    print("모델 평가 및 결과 시각화 완료.")




