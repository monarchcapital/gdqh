import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import io
from datetime import timedelta, date
from pandas.tseries.offsets import BDay

# Set a consistent style for plots
sns.set_style("darkgrid")

# --- Helper Functions ---

def robust_date_parser(df, date_column):
    """
    Parses a date column using multiple known formats and returns the most successful result.
    This function avoids a generic parser fallback that can cause warnings and inconsistencies.
    """
    formats_to_try = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d %B %Y', '%d-%b-%y']
    best_result = None
    fewest_errors = float('inf')

    for fmt in formats_to_try:
        parsed_dates = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
        num_errors = parsed_dates.isnull().sum()

        if num_errors < fewest_errors:
            best_result = parsed_dates
            fewest_errors = num_errors

        if num_errors == 0:
            break  # found perfect format, exit early

    if best_result is not None:
        return best_result, fewest_errors
    
    fallback = pd.to_datetime(df[date_column], errors='coerce')
    return fallback, fallback.isnull().sum()

def np_busdays(start_d, end_d, holidays_np):
    """
    Calculates business days Mon-Fri excluding holidays. 
    start is exclusive, end inclusive-like via busday_count.
    """
    if pd.isna(start_d) or pd.isna(end_d):
        return 0
    s = np.datetime64(pd.Timestamp(start_d).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_d).date())
    if e <= s:
        return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_time_to_maturity(valuation_ts, expiry_ts, holidays_np):
    """
    Calculates the Time-to-Maturity (TTM) in years for a given contract.
    This TTM is based on the number of business days between the valuation and expiry dates.
    """
    BUSINESS_YEAR_DAYS = 252
    bd = np_busdays(valuation_ts, expiry_ts, holidays_np)
    return np.nan if bd <= 0 else bd / BUSINESS_YEAR_DAYS


def parse_expiry_date(maturity_str, expiry_df):
    """
    Looks up the expiry date for a given maturity code from the expiry dataframe.
    """
    try:
        expiry_df['MATURITY'] = expiry_df['MATURITY'].astype(str).str.strip().str.upper()
        expiry_date = expiry_df.loc[expiry_df['MATURITY'] == maturity_str, 'DATE'].iloc[0]
        return pd.to_datetime(expiry_date)
    except (IndexError, TypeError, ValueError) as e:
        # Silently fail, the main loop will handle the NaN
        return None

def interpolate_zero_curve(curve_data_df, standard_maturities_years, interpolation_method, log_messages):
    """
    Interpolates a zero-rate curve for a given date to a set of standard maturities.
    This version avoids extrapolation to prevent unrealistic values.
    """
    valid_data = curve_data_df.dropna().sort_index()
    if valid_data.empty or len(valid_data) < 2:
        return pd.Series(index=standard_maturities_years, dtype=float)

    x = valid_data.index.tolist()
    y = valid_data.values.tolist()
    
    # Define a new series for the results, pre-filled with NaNs
    interpolated_series = pd.Series(np.nan, index=standard_maturities_years, dtype=float)

    # Filter standard maturities to only those within the available data range
    min_ttm = x[0]
    max_ttm = x[-1]
    
    # Log the TTM range for debugging
    log_messages.append(f"Interpolating for TTM range: [{min_ttm:.2f} years, {max_ttm:.2f} years]")

    maturities_to_interpolate = [m for m in standard_maturities_years if min_ttm <= m <= max_ttm]
    
    # Provide a warning if any selected maturities were ignored
    ignored_maturities = [m for m in standard_maturities_years if m < min_ttm or m > max_ttm]
    if ignored_maturities:
        log_messages.append(
            f"Warning: The following maturity points were ignored because they are outside the range of your data's Time-to-Maturity (TTM): "
            f"{', '.join([f'{m:.2f}' for m in ignored_maturities])} years. "
            f"The available TTM range is from {min_ttm:.2f} to {max_ttm:.2f} years."
        )

    if not maturities_to_interpolate:
        return interpolated_series
        
    try:
        f = interp1d(x, y, kind=interpolation_method, bounds_error=True)
        interpolated_values = f(maturities_to_interpolate)
        interpolated_series.loc[maturities_to_interpolate] = interpolated_values
        return interpolated_series
    except Exception as e:
        log_messages.append(f"Warning: Interpolation failed for a curve. Error: {e}")
        return pd.Series(index=standard_maturities_years, dtype=float)

@st.cache_data
def run_analysis(yield_data, expiry_data, holiday_data,
                 interpolation_method, n_components,
                 start_date_filter, end_date_filter,
                 standard_maturities_years, use_smoothing):
    """
    DI-specific pipeline for PCA:
    1. Load and parse all data.
    2. Filter data by date range, weekends, and holidays.
    3. Calculate TTM.
    4. Interpolate rates to standard maturities.
    5. Mean-center and run PCA.
    """
    log_messages = []
    try:
        yield_df = pd.read_csv(io.StringIO(yield_data.getvalue().decode("utf-8")))
        yield_df.columns = [col.strip().upper() for col in yield_df.columns]

        expiry_df = pd.read_csv(io.StringIO(expiry_data.getvalue().decode("utf-8")))
        expiry_df.columns = [col.strip().upper() for col in expiry_df.columns]

        yield_df['DATE'], _ = robust_date_parser(yield_df, 'DATE')
        expiry_df['DATE'], _ = robust_date_parser(expiry_df, 'DATE')

        yield_df.dropna(subset=['DATE'], inplace=True)
        expiry_df.dropna(subset=['DATE'], inplace=True)

        yield_df.set_index('DATE', inplace=True)
        
        holidays_set = set()
        holidays_np = np.array([], dtype="datetime64[D]")
        if holiday_data:
            try:
                holiday_df = pd.read_csv(io.StringIO(holiday_data.getvalue().decode("utf-8")))
                holiday_df.columns = [c.strip().upper() for c in holiday_df.columns]
                holiday_parsed_dates, _ = robust_date_parser(holiday_df, 'DATE')
                holidays_set = set(holiday_parsed_dates.dropna().dt.date)
                holidays_np = np.array([np.datetime64(d) for d in holidays_set], dtype="datetime64[D]")
            except Exception:
                log_messages.append("Warning: Could not parse holidays file. Proceeding without holiday filtering.")

        if start_date_filter and end_date_filter:
            yield_df = yield_df.loc[(yield_df.index.date >= start_date_filter) & (yield_df.index.date <= end_date_filter)]

        if yield_df.empty:
            log_messages.append("Error: No yield data in the selected date range after parsing.")
            return (None,) * 7 + (log_messages,)

        is_business_day = yield_df.index.weekday < 5
        is_not_holiday = ~pd.Series(yield_df.index.date).isin(holidays_set).values
        yield_df = yield_df[is_business_day & is_not_holiday]

        if yield_df.empty:
            log_messages.append("Error: No business days left after holiday/weekend filtering.")
            return (None,) * 7 + (log_messages,)
        
        if use_smoothing:
            yield_df = yield_df.rolling(window=3, min_periods=1, center=True).mean()

        available_maturities = sorted(list(set(yield_df.columns) & set(expiry_df['MATURITY'].astype(str).str.strip().str.upper())))
        if not available_maturities:
            log_messages.append("Error: No maturity names match between yield file and expiry file.")
            return (None,) * 7 + (log_messages,)
        
        interpolated_curves_list = []
        for index, row in yield_df.iterrows():
            current_date = index
            ttm_and_zero_rates = {}
            
            for mat in available_maturities:
                yield_value = row.get(mat)
                if pd.notna(yield_value):
                    expiry_date = parse_expiry_date(mat, expiry_df)
                    if expiry_date and current_date.date() < expiry_date.date():
                        ttm = calculate_time_to_maturity(current_date, expiry_date, holidays_np)
                        if ttm is not np.nan and ttm > 0:
                            # CRITICAL FIX: The DI rate is already the zero-coupon rate for that TTM.
                            # No conversion is needed.
                            zero_rate = yield_value
                            ttm_and_zero_rates[ttm] = zero_rate
            
            if len(ttm_and_zero_rates) >= 2:
                curve_data_df = pd.Series(ttm_and_zero_rates).sort_index()
                # Use the new interpolation function that doesn't extrapolate
                interpolated_curve = interpolate_zero_curve(curve_data_df, standard_maturities_years, interpolation_method, log_messages)
                interpolated_curve.name = current_date
                interpolated_curves_list.append(interpolated_curve)
        
        if not interpolated_curves_list:
            log_messages.append("Error: Could not generate any interpolated curves. Please check your data and date range.")
            return (None,) * 7 + (log_messages,)

        interpolated_df = pd.concat(interpolated_curves_list, axis=1).T
        interpolated_df.index = pd.to_datetime(interpolated_df.index)
        
        # --- CRITICAL CHANGE: Use forward/backward fill instead of dropping rows ---
        interpolated_df.fillna(method='ffill', axis=1, inplace=True)
        interpolated_df.fillna(method='bfill', axis=1, inplace=True)
        interpolated_df.fillna(method='ffill', axis=0, inplace=True)
        interpolated_df.fillna(method='bfill', axis=0, inplace=True)
        
        # If there are still any missing values, drop them (this will be rare now)
        interpolated_df.dropna(how='any', inplace=True)

        if interpolated_df.empty:
            log_messages.append("Error: No complete interpolated curves could be generated.")
            return (None,) * 7 + (log_messages,)

        # --- PCA Core Logic ---
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(interpolated_df)
        
        n_components_actual = min(n_components, scaled_data.shape[1])
        pca = PCA(n_components=n_components_actual)
        principal_components = pca.fit_transform(scaled_data)

        log_messages.append("Success: Analysis completed successfully!")
        return (interpolated_df, pca, scaler, principal_components,
                yield_df, expiry_df, holidays_set, log_messages)

    except Exception as e:
        log_messages.append(f"Error: An error occurred during analysis: {e}")
        return (None,) * 7 + (log_messages,)

# --- Main Streamlit App ---

st.set_page_config(page_title="Yield Curve PCA Analysis", layout="wide")
st.title("Brazilian DI Futures Yield Curve PCA Analysis")

# Initialize session state variables
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'pca_model' not in st.session_state: st.session_state.pca_model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'principal_components' not in st.session_state: st.session_state.principal_components = None
if 'interpolated_df' not in st.session_state: st.session_state.interpolated_df = None
if 'standard_maturities_years' not in st.session_state: st.session_state.standard_maturities_years = np.linspace(0.25, 15.0, 60)
if 'raw_yield_df' not in st.session_state: st.session_state.raw_yield_df = None
if 'expiry_df' not in st.session_state: st.session_state.expiry_df = None
if 'holidays_set' not in st.session_state: st.session_state.holidays_set = set()
if 'log_messages' not in st.session_state: st.session_state.log_messages = []


# --- File Upload and Data Processing ---
st.sidebar.header("Upload Data")

yield_file = st.sidebar.file_uploader("Upload Yield Data (CSV)", type="csv")
expiry_file = st.sidebar.file_uploader("Upload Expiry Data (CSV)", type="csv")
holiday_file = st.sidebar.file_uploader("Upload Holidays (CSV, optional)", type="csv")

# Analysis parameters in sidebar
st.sidebar.header("Analysis Settings")
n_components = st.sidebar.slider("Number of PCA Components", 1, 5, 3)
interpolation_method = st.sidebar.selectbox("Interpolation Method", ["linear", "quadratic", "cubic"])
use_smoothing = st.sidebar.checkbox("Apply 3-day smoothing", value=False)
standard_maturities_years = st.sidebar.multiselect(
    "Standard Maturity Points (Years)",
    options=np.arange(0.25, 15.25, 0.25),
    default=np.arange(0.25, 10.25, 1.0)
)
standard_maturities_years = np.sort(standard_maturities_years)

st.sidebar.header("Date Range for PCA Training")
if yield_file:
    temp_df = pd.read_csv(io.StringIO(yield_file.getvalue().decode("utf-8")))
    temp_df.columns = [col.strip().upper() for col in temp_df.columns]
    temp_df['DATE'], _ = robust_date_parser(temp_df, 'DATE')
    temp_df.dropna(subset=['DATE'], inplace=True)
    temp_df.set_index('DATE', inplace=True)
    
    min_date_available = temp_df.index.min().date()
    max_date_available = temp_df.index.max().date()
    
    start_date_filter = st.sidebar.date_input("Start Date", value=min_date_available, min_value=min_date_available, max_value=max_date_available)
    end_date_filter = st.sidebar.date_input("End Date", value=max_date_available, min_value=min_date_available, max_value=max_date_available)

else:
    start_date_filter = None
    end_date_filter = None

if st.sidebar.button("Run Analysis"):
    if yield_file and expiry_file:
        with st.spinner("Running analysis... this may take a moment."):
            (st.session_state.interpolated_df, st.session_state.pca_model, 
             st.session_state.scaler, st.session_state.principal_components, 
             st.session_state.raw_yield_df, st.session_state.expiry_df, 
             st.session_state.holidays_set, st.session_state.log_messages) = run_analysis(
                yield_file, expiry_file, holiday_file, 
                interpolation_method, n_components,
                start_date_filter, end_date_filter,
                standard_maturities_years, use_smoothing
            )
        st.session_state.analysis_run = st.session_state.interpolated_df is not None
    else:
        st.session_state.log_messages.append("Warning: Please upload both yield and expiry data files to run the analysis.")
        st.session_state.analysis_run = False
    st.experimental_rerun()

if st.sidebar.button("Reset Analysis", help="Click to clear results and start over."):
    st.session_state.clear()
    st.experimental_rerun()


# --- Display Logs in a collapsible box ---
if st.session_state.log_messages:
    with st.expander("Show Analysis Logs", expanded=False):
        for message in st.session_state.log_messages:
            if message.startswith("Error:"):
                st.error(message)
            elif message.startswith("Warning:"):
                st.warning(message)
            else:
                st.info(message)


# --- Main page content based on analysis state ---
if st.session_state.analysis_run:
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["PCA Analysis", "Reconstruction Analysis", "Fair Value Curve"])

    with tab1:
        st.header("PCA Analysis")

        st.subheader("Explained Variance")
        variance_explained = st.session_state.pca_model.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, st.session_state.pca_model.n_components_ + 1), variance_explained, alpha=0.7, label='Individual explained variance')
        ax.step(range(1, st.session_state.pca_model.n_components_ + 1), np.cumsum(variance_explained), where='mid', label='Cumulative explained variance')
        ax.set_ylabel('Explained variance ratio')
        ax.set_xlabel('Principal component')
        ax.set_xticks(range(1, st.session_state.pca_model.n_components_ + 1))
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Principal Components (Factors) Heatmap")
        
        components_df_columns = st.session_state.interpolated_df.columns
        components_df = pd.DataFrame(st.session_state.pca_model.components_, 
                                     columns=components_df_columns, 
                                     index=[f'PC {i+1}' for i in range(st.session_state.pca_model.n_components_)])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(components_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_xlabel("Time to Maturity (Years)")
        ax.set_ylabel("Principal Component")
        ax.set_title("Principal Components (Factors) Heatmap")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Principal Components (Factors) Plot")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=components_df.T, ax=ax, marker='o')
        ax.set_xlabel("Time to Maturity (Years)")
        ax.set_ylabel("Factor Loading")
        ax.set_title("Principal Components (Factors) for Yield Curve Shifts")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Principal Components Data")
        st.dataframe(components_df)

    with tab2:
        st.header("Yield Curve Reconstruction")
        st.markdown(
            "This section demonstrates how well the PCA model can **reconstruct** a historical yield curve "
            "using its principal components. It's a diagnostic tool to understand the model's accuracy."
        )

        available_dates = st.session_state.interpolated_df.index.date
        unique_dates = sorted(list(set(available_dates)), reverse=True)
        
        reconstruction_date = st.selectbox(
            "Select a date for reconstruction",
            options=unique_dates,
            index=0
        )
        
        try:
            reconstruction_ts = pd.Timestamp(reconstruction_date)
            actual_interpolated_curve = st.session_state.interpolated_df.loc[reconstruction_ts]
            
            scaled_data_for_date = st.session_state.scaler.transform(
                actual_interpolated_curve.values.reshape(1, -1)
            )
            
            pca_scores = st.session_state.pca_model.transform(scaled_data_for_date)
            reconstructed_scaled_data = st.session_state.pca_model.inverse_transform(pca_scores)
            reconstructed_curve = st.session_state.scaler.inverse_transform(reconstructed_scaled_data).flatten()
            
            reconstruction_df = pd.DataFrame({
                'Maturity (Years)': st.session_state.interpolated_df.columns,
                'Actual Yield (%)': actual_interpolated_curve.values,
                'Reconstructed Yield (%)': reconstructed_curve
            })
            
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.lineplot(data=reconstruction_df, x='Maturity (Years)', y='Actual Yield (%)', marker='o', ax=ax, label='Actual Yield', color='dodgerblue')
            sns.lineplot(data=reconstruction_df, x='Maturity (Years)', y='Reconstructed Yield (%)', marker='x', ax=ax, label='Reconstructed Yield', color='tomato', linestyle='--')
            
            ax.set_title(f"Yield Curve Reconstruction on {reconstruction_date}")
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Yield (%)")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Reconstruction Data Table")
            st.dataframe(reconstruction_df.set_index('Maturity (Years)'), use_container_width=True)
            
        except KeyError:
            st.error(f"No data available for the selected date: {reconstruction_date}.")
        except Exception as e:
            st.error(f"Error during reconstruction: {e}")

    with tab3:
        st.header("Fair Value Curve Calculation")
        st.markdown(
            "This section calculates a **'fair value'** yield curve for the **most recent trading day** "
            "by using the PCA model to reconstruct the curve. This can be used to compare against "
            "the actual market curve for that day to identify potential mispricings."
        )

        try:
            latest_date_ts = st.session_state.interpolated_df.index.max()
            latest_date_str = latest_date_ts.strftime('%Y-%m-%d')
            
            latest_raw_yields = st.session_state.raw_yield_df.loc[latest_date_ts]

            latest_ttm_and_zero_rates = {}
            available_maturities = st.session_state.raw_yield_df.columns[st.session_state.raw_yield_df.columns != 'DATE']
            holidays_np = np.array([np.datetime64(d) for d in st.session_state.holidays_set], dtype="datetime64[D]")

            for mat in available_maturities:
                yield_value = latest_raw_yields.get(mat)
                if pd.notna(yield_value):
                    expiry_date = parse_expiry_date(mat, st.session_state.expiry_df)
                    if expiry_date and latest_date_ts.date() < expiry_date.date():
                        ttm = calculate_time_to_maturity(latest_date_ts, expiry_date, holidays_np)
                        if ttm is not np.nan and ttm > 0:
                            # CRITICAL FIX: The DI rate is already the zero-coupon rate.
                            zero_rate = yield_value
                            latest_ttm_and_zero_rates[ttm] = zero_rate

            curve_data_for_latest_day = pd.Series(latest_ttm_and_zero_rates).sort_index()
            # Use the new interpolation function that doesn't extrapolate
            actual_interpolated_curve_latest = interpolate_zero_curve(curve_data_for_latest_day, st.session_state.interpolated_df.columns, interpolation_method, st.session_state.log_messages)
            
            if actual_interpolated_curve_latest.isnull().all():
                st.warning("Could not generate an interpolated curve for the latest date. Please check your data.")
            else:
                # Fill NaN values after interpolation, which is now more common.
                actual_interpolated_curve_latest.fillna(method='ffill', inplace=True)
                
                latest_scaled_data = st.session_state.scaler.transform(actual_interpolated_curve_latest.values.reshape(1, -1))
                pca_scores_latest = st.session_state.pca_model.transform(latest_scaled_data)
                reconstructed_scaled_data = st.session_state.pca_model.inverse_transform(pca_scores_latest)
                fair_value_curve = st.session_state.scaler.inverse_transform(reconstructed_scaled_data).flatten()
            
                fair_value_df = pd.DataFrame({
                    'Maturity (Years)': st.session_state.interpolated_df.columns,
                    'Actual Yield (%)': actual_interpolated_curve_latest.values,
                    'Fair Value Yield (%)': fair_value_curve,
                    'Difference (Actual - Fair Value)': actual_interpolated_curve_latest.values - fair_value_curve
                })
                
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.lineplot(data=fair_value_df, x='Maturity (Years)', y='Actual Yield (%)', marker='o', ax=ax, label='Actual Market Curve', color='dodgerblue')
                sns.lineplot(data=fair_value_df, x='Maturity (Years)', y='Fair Value Yield (%)', marker='x', ax=ax, label='Fair Value Curve', color='tomato', linestyle='--')
                
                ax.set_title(f"Fair Value Curve vs. Actual Market Curve on {latest_date_str}")
                ax.set_xlabel("Maturity (Years)")
                ax.set_ylabel("Yield (%)")
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("Fair Value Data Table")
                st.dataframe(fair_value_df.set_index('Maturity (Years)'), use_container_width=True)

        except Exception as e:
            st.error(f"Error occurred while calculating Fair Value Curve: {e}")
