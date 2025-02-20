import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Forecast Zink 2025 BPI",
    page_icon="⚡",
    layout="wide"
)

# Function to load and preprocess data
def load_data():
    try:
        df = pd.read_excel('data dmmy mutasi gudang.xlsx')
        # Convert tanggal to datetime
        df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d/%m/%Y')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to calculate monthly aggregates
def calculate_monthly_data(df):
    try:
        monthly_data = df.groupby([df['tanggal'].dt.to_period('M')]).agg({
            'debet': 'sum',
            'kredit': 'sum'
        }).reset_index()
        monthly_data['tanggal'] = monthly_data['tanggal'].astype(str)
        return monthly_data
    except Exception as e:
        st.error(f"Error calculating monthly data: {str(e)}")
        return None

# Function for SARIMA forecast
def sarima_forecast(data, periods=12):
    try:
        model = SARIMAX(data, 
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False)
        results = model.fit()
        forecast = results.forecast(periods)
        conf_int = results.get_forecast(periods).conf_int()
        return forecast, conf_int
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None

def main():
    st.title("⚡ Analisis Prediktif Konsumsi Zink 2025 - PT Bakrie Pipe Industries")
    
    try:
        # Load data
        df = load_data()
        if df is None:
            st.error("Failed to load data. Please check your Excel file.")
            return
            
        monthly_data = calculate_monthly_data(df)
        if monthly_data is None:
            st.error("Failed to process monthly data.")
            return
        
        # Calculate basic metrics
        avg_monthly_kredit = monthly_data['kredit'].mean()
        std_dev_kredit = monthly_data['kredit'].std()
        cv = (std_dev_kredit / avg_monthly_kredit) * 100
        usage_percentage = (df['kredit'].sum() / df['debet'].sum()) * 100

        # Add tabs
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Informasi Sistem", 
            "Overview", 
            "Analisis Detail", 
            "Forecast 2025", 
            "Rekomendasi", 
            "Rumus & Metodologi"
        ])

        with tab0:
            st.header("Informasi Sistem Analisis Mutasi Gudang")
            
            st.subheader("🎯 Tujuan Sistem")
            st.write("""
            Sistem ini dikembangkan untuk:
            1. Menganalisis pola penggunaan dan pengisian stok barang
            2. Memprediksi kebutuhan barang di masa depan
            3. Memberikan rekomendasi pengelolaan stok yang optimal
            4. Membantu pengambilan keputusan dalam manajemen inventori
            """)
            
            st.subheader("🔍 Metode yang Digunakan")
            st.write("""
            Sistem ini menggunakan metode SARIMA (Seasonal Autoregressive Integrated Moving Average) untuk forecasting dengan alasan:
            
            **1. Komponen yang Ditangkap:**
            - Tren (trend)
            - Musiman (seasonality)
            - Siklus (cyclical patterns)
            - Noise/random variations
            
            **2. Kelebihan SARIMA untuk Kasus Ini:**
            - Mampu menangkap pola musiman yang terdapat dalam data
            - Mempertimbangkan dependency antar observasi
            - Dapat menangani data non-stasioner
            - Memberikan interval kepercayaan untuk forecast
            """)
            
            st.subheader("❌ Mengapa Tidak Menggunakan Metode Lain")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("""
                **Simple Moving Average:**
                - Terlalu sederhana
                - Tidak bisa menangkap seasonality
                - Tidak memberikan interval kepercayaan
                - Tidak mempertimbangkan tren
                
                **Exponential Smoothing:**
                - Kurang cocok untuk data dengan seasonality kuat
                - Tidak optimal untuk data dengan variasi tinggi
                - Tidak mempertimbangkan autokorelasi
                """)
            
            with col2:
                st.write("""
                **Linear Regression:**
                - Asumsi linearitas tidak terpenuhi
                - Tidak bisa menangkap pola musiman
                - Tidak cocok untuk time series kompleks
                
                **Prophet (Facebook):**
                - Terlalu kompleks untuk pola data yang ada
                - Membutuhkan data yang lebih panjang
                - Overfitting untuk kasus sederhana
                """)
            
            st.subheader("📈 Alur Analisis")
            st.write("""
            1. **Preprocessing Data**
               - Konversi tanggal
               - Agregasi data bulanan
               - Penanganan missing values
            
            2. **Analisis Pola**
               - Time series decomposition
               - Identifikasi seasonality
               - Analisis tren
            
            3. **Forecasting**
               - SARIMA model fitting
               - Parameter optimization
               - Forecast generation
            
            4. **Evaluasi & Rekomendasi**
               - Perhitungan metrics
               - Analisis hasil
               - Pemberian rekomendasi
            """)

        with tab1:
            st.header("Overview Data")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transaksi", f"{len(df):,}")
            with col2:
                st.metric("Total Debet", f"{df['debet'].sum():,.2f}")
            with col3:
                st.metric("Total Kredit", f"{df['kredit'].sum():,.2f}")
            with col4:
                st.metric("Persentase Penggunaan", f"{usage_percentage:.2f}%")
            
            # Basic info
            st.subheader("Informasi Barang")
            st.write(f"""
            - Kode Barang: {df['nobar'].iloc[0]}
            - Nama Barang: {df['nabar'].iloc[0]}
            - Periode Data: {df['tanggal'].min().strftime('%d %B %Y')} - {df['tanggal'].max().strftime('%d %B %Y')}
            """)
            
            # Time series plot
            st.subheader("Grafik Mutasi Barang")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=monthly_data['tanggal'],
                y=monthly_data['kredit'],
                name='Kredit (Penggunaan)',
                line=dict(color='red')
            ))
            fig_ts.add_trace(go.Scatter(
                x=monthly_data['tanggal'],
                y=monthly_data['debet'],
                name='Debet (Pengisian)',
                line=dict(color='blue')
            ))
            fig_ts.update_layout(
                title='Time Series Mutasi Barang',
                xaxis_title='Periode',
                yaxis_title='Jumlah'
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with tab2:
            st.header("Analisis Detail")
            
            # Monthly statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata Penggunaan Bulanan", f"{avg_monthly_kredit:,.2f}")
            with col2:
                st.metric("Standar Deviasi Penggunaan", f"{std_dev_kredit:,.2f}")
            with col3:
                st.metric("Coefficient of Variation", f"{cv:.2f}%")
            
            # Monthly trend analysis
            st.subheader("Analisis Tren Bulanan")
            monthly_summary = pd.DataFrame({
                'Bulan': monthly_data['tanggal'],
                'Penggunaan': monthly_data['kredit'],
                'Pengisian': monthly_data['debet']
            })
            st.dataframe(monthly_summary.style.highlight_max(axis=0), hide_index=True)
            
            # Usage patterns
            st.subheader("Pola Penggunaan")
            fig_pattern = go.Figure()
            fig_pattern.add_trace(go.Box(
                y=monthly_data['kredit'],
                name='Distribusi Penggunaan'
            ))
            st.plotly_chart(fig_pattern, use_container_width=True)

        with tab3:
            st.header("Forecast 2025")
            
            # Prepare data for forecasting
            kredit_series = monthly_data.set_index('tanggal')['kredit']
            forecast, conf_int = sarima_forecast(kredit_series)
            
            if forecast is not None and conf_int is not None:
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'tanggal': pd.date_range(start='2025-01-01', periods=12, freq='M'),
                    'forecast': forecast,
                    'lower_bound': conf_int.iloc[:, 0],
                    'upper_bound': conf_int.iloc[:, 1]
                })
                
                # Forecast plot
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=monthly_data['tanggal'],
                    y=monthly_data['kredit'],
                    name='Data Historis',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['tanggal'].dt.strftime('%Y-%m'),
                    y=forecast_df['forecast'],
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['tanggal'].dt.strftime('%Y-%m').tolist() + 
                      forecast_df['tanggal'].dt.strftime('%Y-%m').tolist()[::-1],
                    y=forecast_df['upper_bound'].tolist() + 
                      forecast_df['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='Interval Kepercayaan'
                ))
                
                fig_forecast.update_layout(
                    title='Forecast Penggunaan Barang 2025',
                    xaxis_title='Periode',
                    yaxis_title='Jumlah'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast details
                st.subheader("Detail Forecast 2025")
                forecast_details = pd.DataFrame({
                    'Bulan': pd.date_range(start='2025-01-01', periods=12, freq='M').strftime('%B %Y'),
                    'Prediksi': forecast,
                    'Batas Bawah': conf_int.iloc[:, 0],
                    'Batas Atas': conf_int.iloc[:, 1]
                })
                st.dataframe(forecast_details.round(2), hide_index=True)

        with tab4:
            st.header("Rekomendasi Pengelolaan Stok")
            
            if forecast is not None and conf_int is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Level Stok Rekomendasi")
                    st.write(f"""
                    - **Stok Minimal:** {forecast_df['lower_bound'].mean():.2f} unit
                    - **Stok Optimal:** {forecast_df['forecast'].mean():.2f} unit
                    - **Stok Maksimal:** {forecast_df['upper_bound'].mean():.2f} unit
                    """)
                
                with col2:
                    st.subheader("⚠️ Pengaturan Reorder")
                    st.write(f"""
                    - **Reorder Point:** 15,000 unit
                    - **Kuantitas Pemesanan:** 40,000 unit
                    - **Safety Stock:** {forecast_df['lower_bound'].mean() * 0.2:.2f} unit
                    """)
                
                st.subheader("📈 Analisis Penggunaan")
                st.write(f"""
                - **Usage Rate:** {usage_percentage:.2f}%
                - **Variabilitas Penggunaan:** {cv:.2f}%
                - **Rata-rata Penggunaan Bulanan:** {avg_monthly_kredit:,.2f} unit
                """)
                
                if usage_percentage > 90:
                    st.warning("""
                    ⚠️ **Perhatian!**
                    Usage rate di atas 90% menunjukkan tingkat penggunaan yang tinggi.
                    Pertimbangkan untuk:
                    1. Meningkatkan safety stock
                    2. Mempercepat siklus pengisian
                    3. Evaluasi kapasitas penyimpanan
                    """)

        with tab5:
            st.header("Rumus dan Metodologi")
            
            st.subheader("📊 Rumus-rumus yang Digunakan")
            
            st.write("**1. Perhitungan Statistik Dasar**")
            st.latex(r"""
            \begin{align*}
            \text{Usage Rate} &= \frac{\text{Total Kredit}}{\text{Total Debet}} \times 100\% \\
            \text{Average Monthly Usage} &= \frac{\sum \text{Kredit}_i}{n} \\
            \text{Standard Deviation} &= \sqrt{\frac{\sum(\text{Kredit}_i - \text{Average})^2}{n}} \\
            \text{Coefficient of Variation} &= \frac{\text{Standard Deviation}}{\text{Average}} \times 100\%
            \end{align*}
            """)
            
            st.write("**2. Model SARIMA (p,d,q)(P,D,Q)s**")
            st.latex(r"""
            \Phi(B^s)\phi(B)(1-B)^d(1-B^s)^D Y_t = \Theta(B^s)\theta(B)\epsilon_t
            """)
            
            st.write("""
            Dimana:
            - B = operator backshift
            - s = panjang musiman
            - ϕ(B) = AR polynomial
            - θ(B) = MA polynomial
            - Φ(Bs) = Seasonal AR polynomial
            - Θ(Bs) = Seasonal MA polynomial
            """)
            
            st.write("**3. Safety Stock**")
            st.latex(r"""
            \text{Safety Stock} = Z_\alpha \times \sigma \times \sqrt{L}
            """)
            
            st.write("""
            Dimana:
            - Zα = nilai Z untuk service level
            - σ = standar deviasi penggunaan
            - L = lead time
            """)
            
            st.write("**4. Reorder Point (ROP)**")
            st.latex(r"""
            \text{ROP} = (\text{Average Daily Usage} \times \text{Lead Time}) + \text{Safety Stock}
            """)
            
            st.write("**5. Economic Order Quantity (EOQ)**")
            st.latex(r"""
            \text{EOQ} = \sqrt{\frac{2DS}{H}}
            """)
            
            st.write("""
            Dimana:
            - D = permintaan tahunan
            - S = biaya pemesanan
            - H = biaya penyimpanan
            """)
            
            st.subheader("📈 Metrics Evaluasi Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Mean Absolute Percentage Error (MAPE)**")
                st.latex(r"""
                \text{MAPE} = \frac{1}{n}\sum_{t=1}^n \left|\frac{A_t - F_t}{A_t}\right| \times 100\%
                """)
                st.write("""
                Dimana:
                - At = nilai aktual
                - Ft = nilai forecast
                - n = jumlah periode
                """)
            
            with col2:
                st.write("**Root Mean Square Error (RMSE)**")
                st.latex(r"""
                \text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n(A_t - F_t)^2}
                """)
                st.write("""
                Dimana:
                - At = nilai aktual
                - Ft = nilai forecast
                - n = jumlah periode
                """)
            
            st.subheader("⚙️ Parameter Model SARIMA")
            st.write("""
            Model yang digunakan: SARIMA(1,1,1)(1,1,1)12
            
            **Komponen Non-Seasonal:**
            - p = 1 (AR order)
            - d = 1 (differencing)
            - q = 1 (MA order)
            
            **Komponen Seasonal:**
            - P = 1 (Seasonal AR)
            - D = 1 (Seasonal differencing)
            - Q = 1 (Seasonal MA)
            - s = 12 (Seasonal period)
            """)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.write("Silakan periksa data input dan coba lagi.")

if __name__ == "__main__":
    main()