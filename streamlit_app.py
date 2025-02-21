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
        # Load zinc consumption data
        df_zinc = pd.read_excel('data dmmy mutasi gudang.xlsx')
        df_zinc['tanggal'] = pd.to_datetime(df_zinc['tanggal'], format='%d/%m/%Y')
        
        # Load production data for usage percentage calculation
        df_prod = pd.read_excel('dmmy produksi galvaniz.xlsx')
        df_prod['tanggal'] = pd.to_datetime(df_prod['tanggal'], format='%d/%m/%y')
        
        # Ensure all dates are in the correct range (2022-2024)
        mask = df_prod['tanggal'].dt.year < 2000
        df_prod.loc[mask, 'tanggal'] = df_prod.loc[mask, 'tanggal'] + pd.DateOffset(years=100)
        
        return df_zinc, df_prod
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Function to calculate monthly data
def calculate_monthly_data(df_zinc, df_prod):
    try:
        # Calculate monthly zinc data
        monthly_zinc = df_zinc.groupby(df_zinc['tanggal'].dt.to_period('M')).agg({
            'debet': 'sum',
            'kredit': 'sum'
        }).reset_index()
        
        # Calculate monthly production weight
        monthly_prod = df_prod.groupby(df_prod['tanggal'].dt.to_period('M')).agg({
            'kg_shift1': 'sum',
            'kg_shift2': 'sum',
            'kg_shift3': 'sum'
        }).reset_index()
        
        # Calculate total production weight
        monthly_prod['total_production'] = (
            monthly_prod['kg_shift1'] + 
            monthly_prod['kg_shift2'] + 
            monthly_prod['kg_shift3']
        )
        
        # Merge zinc and production data
        monthly_data = pd.merge(
            monthly_zinc,
            monthly_prod[['tanggal', 'total_production']],
            on='tanggal',
            how='inner'
        )
        
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
        df_zinc, df_prod = load_data()
        if df_zinc is None or df_prod is None:
            st.error("Failed to load data. Please check your Excel files.")
            return
            
        monthly_data = calculate_monthly_data(df_zinc, df_prod)
        if monthly_data is None:
            st.error("Failed to process monthly data.")
            return
        
        # Calculate basic metrics
        avg_monthly_kredit = monthly_data['kredit'].mean()
        std_dev_kredit = monthly_data['kredit'].std()
        cv = (std_dev_kredit / avg_monthly_kredit) * 100

        # Add tabs
        tab0, tab1, tab1a, tab2, tab3, tab4, tab5 = st.tabs([
            "Informasi Sistem", 
            "Overview",
            "Analisis Persentase Penggunaan",
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transaksi", f"{len(df_zinc):,} transaksi")
            with col2:
                st.metric("Total Debet", f"{df_zinc['debet'].sum():,.2f} kg")
            with col3:
                st.metric("Total Kredit", f"{df_zinc['kredit'].sum():,.2f} kg")
            
            # Basic info
            st.subheader("Informasi Barang")
            st.write(f"""
            - Kode Barang: {df_zinc['nobar'].iloc[0]}
            - Nama Barang: {df_zinc['nabar'].iloc[0]}
            - Periode Data: {df_zinc['tanggal'].min().strftime('%d %B %Y')} - {df_zinc['tanggal'].max().strftime('%d %B %Y')}
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
                yaxis_title='Jumlah (kg)'
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with tab1a:
            st.header("Analisis Persentase Penggunaan Zinc")
            
            # Metrics overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata Persentase Penggunaan", f"{(monthly_data['kredit'] / monthly_data['total_production'] * 100).mean():.2f}%")
            with col2:
                st.metric("Persentase Tertinggi", f"{(monthly_data['kredit'] / monthly_data['total_production'] * 100).max():.2f}%")
            with col3:
                st.metric("Persentase Terendah", f"{(monthly_data['kredit'] / monthly_data['total_production'] * 100).min():.2f}%")

            # Detailed monthly table
            st.subheader("Detail Persentase Penggunaan per Bulan")
            monthly_usage = pd.DataFrame({
                'Periode': monthly_data['tanggal'],
                'Konsumsi Zinc (kg)': monthly_data['kredit'].round(2),
                'Berat Produksi (kg)': monthly_data['total_production'].round(2),
                'Persentase Penggunaan (%)': (monthly_data['kredit'] / monthly_data['total_production'] * 100).round(2)
            })
            st.dataframe(monthly_usage.style.highlight_max(axis=0, subset=['Persentase Penggunaan (%)']), hide_index=True)

            # Time series plot for usage percentage
            st.subheader("Grafik Persentase Penggunaan Zinc")
            fig_usage = go.Figure()
            fig_usage.add_trace(go.Scatter(
                x=monthly_data['tanggal'],
                y=(monthly_data['kredit'] / monthly_data['total_production'] * 100),
                name='Persentase Penggunaan',
                line=dict(color='green')
            ))
            fig_usage.update_layout(
                title='Tren Persentase Penggunaan Zinc terhadap Berat Produksi',
                xaxis_title='Periode',
                yaxis_title='Persentase (%)'
            )
            st.plotly_chart(fig_usage, use_container_width=True)

            # Box plot for usage percentage distribution
            st.subheader("Distribusi Persentase Penggunaan")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(
                y=(monthly_data['kredit'] / monthly_data['total_production'] * 100),
                name='Distribusi Persentase'
            ))
            fig_dist.update_layout(
                title='Distribusi Persentase Penggunaan Zinc',
                yaxis_title='Persentase (%)'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Statistical analysis
            st.subheader("Analisis Statistik Persentase Penggunaan")
            col1, col2 = st.columns(2)
            
            percentage_series = monthly_data['kredit'] / monthly_data['total_production'] * 100
            
            with col1:
                st.write("**Statistik Deskriptif:**")
                stats_df = pd.DataFrame({
                    'Metrik': ['Mean', 'Median', 'Standar Deviasi', 'Coefficient of Variation'],
                    'Nilai': [
                        f"{percentage_series.mean():.2f}%",
                        f"{percentage_series.median():.2f}%",
                        f"{percentage_series.std():.2f}%",
                        f"{(percentage_series.std() / percentage_series.mean() * 100):.2f}%"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                st.write("**Analisis Persentil:**")
                percentiles_df = pd.DataFrame({
                    'Persentil': ['25%', '50%', '75%', '90%'],
                    'Nilai': [
                        f"{percentage_series.quantile(0.25):.2f}%",
                        f"{percentage_series.quantile(0.50):.2f}%",
                        f"{percentage_series.quantile(0.75):.2f}%",
                        f"{percentage_series.quantile(0.90):.2f}%"
                    ]
                })
                st.dataframe(percentiles_df, hide_index=True)

            # Monthly comparison
            st.subheader("Perbandingan Bulanan")
            monthly_comparison = monthly_data.copy()
            monthly_comparison['Month'] = pd.to_datetime(monthly_comparison['tanggal']).dt.strftime('%B')
            monthly_comparison['usage_percentage'] = monthly_comparison['kredit'] / monthly_comparison['total_production'] * 100
            
            # Define month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
            
            # Calculate average by month and sort
            monthly_avg = monthly_comparison.groupby('Month')['usage_percentage'].mean().round(2)
            monthly_avg = monthly_avg.reindex(month_order)
            
            fig_monthly = go.Figure(data=[
                go.Bar(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    name='Rata-rata Persentase'
                )
            ])
            fig_monthly.update_layout(
                title='Rata-rata Persentase Penggunaan per Bulan',
                xaxis_title='Bulan',
                yaxis_title='Persentase (%)'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        with tab2:
            st.header("Analisis Detail")
            
            # Monthly statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata Penggunaan Bulanan", f"{avg_monthly_kredit:,.2f} kg")
            with col2:
                st.metric("Standar Deviasi Penggunaan", f"{std_dev_kredit:,.2f} kg")
            with col3:
                st.metric("Coefficient of Variation", f"{cv:.2f}%")
            
            # Monthly trend analysis
            st.subheader("Analisis Tren Bulanan")
            monthly_summary = pd.DataFrame({
                'Bulan': monthly_data['tanggal'],
                'Penggunaan (kg)': monthly_data['kredit'].round(2),
                'Pengisian (kg)': monthly_data['debet'].round(2)
            })
            st.dataframe(monthly_summary.style.highlight_max(axis=0), hide_index=True)
            
            # Usage patterns
            st.subheader("Pola Penggunaan")
            fig_pattern = go.Figure()
            fig_pattern.add_trace(go.Box(
                y=monthly_data['kredit'],
                name='Distribusi Penggunaan'
            ))
            fig_pattern.update_layout(
                yaxis_title='Jumlah (kg)'
            )
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
                    yaxis_title='Jumlah (kg)'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast details
                st.subheader("Detail Forecast 2025")
                forecast_details = pd.DataFrame({
                    'Bulan': forecast_df['tanggal'].dt.strftime('%B %Y'),
                    'Prediksi (kg)': forecast.round(2),
                    'Batas Bawah (kg)': conf_int.iloc[:, 0].round(2),
                    'Batas Atas (kg)': conf_int.iloc[:, 1].round(2)
                })
                st.dataframe(forecast_details, hide_index=True)

        with tab4:
            st.header("Rekomendasi Pengelolaan Stok")
            
            if forecast is not None and conf_int is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Level Stok Rekomendasi")
                    st.write(f"""
                    - **Stok Minimal:** {forecast_df['lower_bound'].mean():.2f} kg
                    - **Stok Optimal:** {forecast_df['forecast'].mean():.2f} kg
                    - **Stok Maksimal:** {forecast_df['upper_bound'].mean():.2f} kg
                    """)
                
                with col2:
                    st.subheader("⚠️ Pengaturan Reorder")
                    st.write(f"""
                    - **Reorder Point:** 15,000 kg
                    - **Kuantitas Pemesanan:** 40,000 kg
                    - **Safety Stock:** {forecast_df['lower_bound'].mean() * 0.2:.2f} kg
                    """)
                
                st.subheader("📈 Analisis Konsumsi")
                
                # Calculate correlation with production
                production_correlation = monthly_data['kredit'].corr(monthly_data['total_production'])
                
                st.write(f"""
                **1. Penggunaan Zinc:**
                - Rata-rata konsumsi bulanan: {avg_monthly_kredit:.2f} kg
                - Standar deviasi: {std_dev_kredit:.2f} kg
                - Koefisien variasi: {cv:.2f}%
                
                **2. Interpretasi Hasil:**
                - {'Konsumsi zinc stabil' if cv < 15 else 'Konsumsi zinc cukup variabel' if cv < 25 else 'Konsumsi zinc sangat variabel'}
                - {'Korelasi sangat baik dengan produksi' if production_correlation > 0.8 else 'Korelasi cukup baik dengan produksi' if production_correlation > 0.6 else 'Perlu evaluasi korelasi dengan produksi'}
                
                **3. Rekomendasi:**
                - {'Pertahankan pola konsumsi saat ini' if cv < 15 else 'Evaluasi variabilitas konsumsi' if cv < 25 else 'Perlu standardisasi proses'}
                - {'Monitor dan pertahankan korelasi' if production_correlation > 0.8 else 'Tingkatkan korelasi dengan produksi' if production_correlation > 0.6 else 'Evaluasi pola konsumsi vs produksi'}
                """)
                
                if cv > 25:
                    st.warning("""
                    ⚠️ **Perhatian!**
                    Teridentifikasi variabilitas konsumsi yang tinggi:
                    1. Evaluasi parameter proses galvanisasi
                    2. Periksa konsistensi ketebalan coating
                    3. Optimalkan suhu dan waktu pencelupan
                    4. Standardisasi prosedur operasi
                    """)

        with tab5:
            st.header("Rumus dan Metodologi")
            
            st.subheader("📊 Rumus-rumus yang Digunakan")
            
            st.write("**1. Perhitungan Statistik Dasar**")
            st.latex(r"""
            \begin{align*}
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