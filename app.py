# app.py (Spark-compatible)
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from config import Config
from process_data import ProcessData
from graphs import CreateGraphs
from dynamic_insights import CreateInsights

# Thư viện cần cho PySpark và xử lý dữ liệu
import pandas as pd
import polars as pl
import os

# ⚙️ Khai báo JAVA_HOME và PATH cho PySpark
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-17.0.16.8-hotspot"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")
from pyspark.sql import SparkSession, functions as F

# ⚙️ Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# ⚙️ Khởi tạo cấu hình và SparkSession
import os
from pyspark.sql import SparkSession
from config import Config

# 🧩 Đặt biến môi trường cho PySpark (bắt buộc với Windows)
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-17"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[*] pyspark-shell"

# ⚙️ Khởi tạo cấu hình
config = Config()

# ⚙️ Tạo SparkSession
spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("RealEstateDashboard")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.ui.showConsoleProgress", "false")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

print("✅ Spark started successfully:", spark.version)

# ⚙️ Khởi tạo các class xử lý
process = ProcessData(spark)
graphs = CreateGraphs()
insights = CreateInsights()

# ----------------------------
# Read cleaned data (Spark DataFrame)
main_df = process.get_clean_data()  # Spark DataFrame expected


# ----------------------------
# Helpers to get scalar/unique values from Spark DataFrame
def spark_min(col_name):
    val = main_df.select(F.min(F.col(col_name)).alias("min_val")).collect()[0]["min_val"]
    return val

def spark_max(col_name):
    val = main_df.select(F.max(F.col(col_name)).alias("max_val")).collect()[0]["max_val"]
    return val

def spark_unique_list(col_name):
    # returns Python list of unique values (excluding nulls)
    pd_vals = main_df.select(col_name).where(F.col(col_name).isNotNull()).distinct().toPandas()
    if col_name in pd_vals.columns:
        return pd_vals[col_name].tolist()
    return []

# ----------------------------
# Sidebar widgets for dynamic interaction
st.sidebar.header("Filters")

# Date Range Preparation (convert Spark min/max to python date)
earliest_date = spark_min("date")
latest_date = spark_max("date")

# If earliest/latest are Timestamps -> convert to date
try:
    if hasattr(earliest_date, "date"):
        earliest_date = earliest_date.date()
    if hasattr(latest_date, "date"):
        latest_date = latest_date.date()
except Exception:
    pass

# Location Filter Preparation
location_options = ["All"] + spark_unique_list("region")
location_options = sorted(location_options)

# Property Type Preparation
prop_types = ["All"] + [v for v in spark_unique_list("typeOfFlat") if v is not None]
prop_types = sorted(prop_types)

# Energy Efficiency Filter Preparation
energy_types = ["All"] + spark_unique_list("energyEfficiencyClass")
energy_types = sorted(energy_types)

# Date Filter
date_range = st.sidebar.date_input("Date Range", [earliest_date, latest_date])

# Location Filter
selected_location = st.sidebar.selectbox("Location", index=0, options=location_options)

# Property Type Filter
selected_prop_types = st.sidebar.multiselect("Select Property Type", options=tuple(prop_types))

# Energy Efficiency Class Filter
selected_efficiency_class = st.sidebar.selectbox("Select Energy Efficiency Class", options=["All"] + energy_types)

# Apply filters (process.filter_data expects Spark DataFrame and date_range-like)
filtered_data = process.filter_data(
    main_df,
    date_range,
    selected_location,
    selected_prop_types,
    selected_efficiency_class,
)

with st.sidebar:
    add_vertical_space(10)
    st.sidebar.markdown("<small style='margin-top:15px;'>made by SteinCode</small> ", unsafe_allow_html=True)

# Header & Intro
img_1_co, img_2_co = st.columns([2, 5])
with img_2_co:
    st.image(config.new_image)
left_co, cent_co, last_co = st.columns([2, 5, 1])

with cent_co:
    st.markdown("<h1 style='text-align: center; margin-top: -3%; font-size: 28px;'>Rental Property Analysis</h1>", unsafe_allow_html=True)

st.write(
    "Access detailed rental property data for analysis and controlling tasks with our app. "
    "It provides data exploration tools to analyze rent, property features, and geographic details. "
    "The application supports trend analysis and financial monitoring, offering insights for decision-making "
    "without complex terminology."
)

st.subheader("Data Overview")

# -------- TABLE ------------
@st.cache_data(show_spinner=True)
def split_frame(input_df: pd.DataFrame, rows: int):
    df_list = [input_df.loc[i: i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df_list

with st.expander("**Filter Table**"):
    slider_1, slider_2, slider_3 = st.columns(3)
    # For sliders we need numeric min/max from Spark (converted to python ints)
    min_rent = int(float(spark_min("baseRent"))) if spark_min("baseRent") is not None else 0
    max_rent = int(float(spark_max("baseRent"))) if spark_max("baseRent") is not None else 10000
    rent_range = slider_1.slider('Base Rent Range', min_value=min_rent, max_value=max_rent, value=(min_rent, max_rent))

    min_year = int(spark_min("yearConstructed")) if spark_min("yearConstructed") is not None else 1900
    max_year = int(spark_max("yearConstructed")) if spark_max("yearConstructed") is not None else 2020
    year_range = slider_2.slider('Year Constructed Range', min_value=min_year, max_value=max_year, value=(min_year, max_year))

    min_space = int(float(spark_min("livingSpace"))) if spark_min("livingSpace") is not None else 0
    max_space = int(float(spark_max("livingSpace"))) if spark_max("livingSpace") is not None else 300
    space_range = slider_3.slider('Living Space Range (sqm)', min_value=min_space, max_value=max_space, value=(min_space, max_space))

    sort_column = slider_1.selectbox('Select column to sort by:', [
        "None",
        "baseRent",
        "totalRent",
        "yearConstructed",
        "livingSpace",
        "heatingCosts",
        "noRooms"
    ])

# Prepare table: process.filter_table returns a Spark DataFrame (in PySpark version)
# Convert filtered Spark DF to pandas for table display (limit to top 100)
filtered_table_spark = process.filter_table(filtered_data, rent_range, year_range, space_range, sort_column)
# ensure it's Spark DF; convert to pandas and format
try:
    filtered_table_pd = filtered_table_spark.limit(100).toPandas()
except Exception:
    # if the function already returned pandas
    filtered_table_pd = filtered_table_spark if isinstance(filtered_table_spark, pd.DataFrame) else pd.DataFrame()

# Format using existing formatter if available (it expects pandas)
try:
    filtered_table_pd = process.format_pandas_dataframe(filtered_table_pd)
except Exception:
    pass

# Pagination UI
pagination = st.container()
bottom_menu = st.columns((4, 1, 1))
with bottom_menu[2]:
    batch_size = st.selectbox("Page Size", options=[25, 50, 100])
with bottom_menu[1]:
    total_pages = (int(len(filtered_table_pd) / batch_size) if int(len(filtered_table_pd) / batch_size) > 0 else 1)
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
with bottom_menu[0]:
    st.markdown(f"Page **{current_page}** of **{total_pages}** ")

pages = split_frame(filtered_table_pd, batch_size)
# guard index
page_to_show = pages[current_page - 1] if len(pages) >= current_page else pages[-1]
pagination.dataframe(data=page_to_show, column_order=config.column_order, use_container_width=True)

# ---- TABLE END ----------------

# Expanders with metadata
with st.expander("Location Details - Click for More Details"):
    for column, description in config.location_details.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Property Features - Click for More Details"):
    for column, description in config.property_features.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Amenities - Click for More Details"):
    for column, description in config.amenities.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Financial Details - Click for More Details"):
    for column, description in config.financial_details.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Construction and Efficiency - Click for More Details"):
    for column, description in config.construction_and_efficiency.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Additional Descriptions - Click for More Details"):
    for column, description in config.additional_descriptions.items():
        st.markdown(f"- **{column}:** {description}")

# GRAPHS SECTION
st.markdown("<h2 style='text-align: center;'>Explorative Data Analysis</h2><br>", unsafe_allow_html=True)

# Overview table (graphs.get_overview_info in PySpark version expects Spark DF)
table_md = graphs.get_overview_info(main_df=main_df)
st.markdown(table_md, unsafe_allow_html=True)

# Calculating Metrics (graphs.calculate_key_metrics accepts Spark DF)
median_rent, num_properties, avg_year_constructed = graphs.calculate_key_metrics(filtered_data)

metric_cols = st.columns((2, 3, 3, 3))
with metric_cols[1]:
    st.metric(label="Median Rent (€)", value=f"{median_rent:.2f}")
with metric_cols[2]:
    st.metric(label="Number of Properties", value=num_properties)
with metric_cols[3]:
    st.metric(label="Avg. Year Constructed", value=f"{avg_year_constructed:.0f}")

# Dynamic Map (graphs.create_dynamic_map expects Spark DF)
st.markdown("<strong>Median Base Rent based on Region</strong>", unsafe_allow_html=True)
map_fig, final_df_pd = graphs.create_dynamic_map(df=filtered_data)
st.plotly_chart(map_fig, use_container_width=True)
insight_text_region_map = insights.generate_map_insights(final_df=final_df_pd)
st.markdown(insights.insight_container_style, unsafe_allow_html=True)
st.markdown(f"<div class='insight-container'>{insight_text_region_map}</div>", unsafe_allow_html=True)

# Wordcloud & barchart (graphs.generate_wordcloud accepts Spark DF)
text_graph_col1, text_graph_col2 = st.columns(2)
# adjust slider range: max words
wordcloud_slider = text_graph_col1.slider("Max Number of Words", 50, 250, 100)
wordcloud_fig, word_barchart = graphs.generate_wordcloud(filtered_data, max_words=wordcloud_slider)
text_graph_col1.pyplot(wordcloud_fig, use_container_width=True)
text_graph_col2.plotly_chart(word_barchart, use_container_width=True)

# Distribution histograms (graphs.average_rent_year_distribution accepts Spark DF)
dist_col1, dist_col2 = st.columns(2)
dist_rent, dist_year = graphs.average_rent_year_distribution(df=main_df)
dist_col1.plotly_chart(dist_rent, use_container_width=True)
histogram_text_rental = insights.generate_histogram_insights(df=main_df, column_name="baseRent")
dist_col1.markdown(insights.insight_container_style, unsafe_allow_html=True)
dist_col1.markdown(f"<div class='insight-container'>{histogram_text_rental}</div>", unsafe_allow_html=True)

dist_col2.plotly_chart(dist_year, use_container_width=True)
histogram_text_year = insights.generate_histogram_insights(df=main_df, column_name="yearConstructed")
dist_col2.markdown(insights.insight_container_style, unsafe_allow_html=True)
dist_col2.markdown(f"<div class='insight-container'>{histogram_text_year}</div>", unsafe_allow_html=True)

st.markdown("""<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8" /> """, unsafe_allow_html=True)
col_1, col_2 = st.columns(2)

# Column 1
with col_1:
    rental_trend_fig, aggregated_df_average_rental = graphs.create_rent_trend_over_time(df=filtered_data)
    st.plotly_chart(rental_trend_fig, use_container_width=True)
    # aggregated_df_average_rental is pandas (graphs returned pandas agg), pass to insights
    insight_text_average = insights.extract_insights_average_rent_over_time(aggregated_df=aggregated_df_average_rental)
    col_1.markdown(insights.insight_container_style, unsafe_allow_html=True)
    col_1.markdown(f"<div class='insight-container'>{insight_text_average}</div>", unsafe_allow_html=True)

# Column 2
with col_2:
    rentals_per_season_fig, average_rent_by_season = graphs.get_comparison_rent_by_season(df=filtered_data)
    st.plotly_chart(rentals_per_season_fig, use_container_width=True)
    insight_text_season = insights.extract_insights_average_rent_by_season(monthly_avg_rent=average_rent_by_season)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(f"<div class='insight-container'>{insight_text_season}</div>", unsafe_allow_html=True)

st.markdown("""<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8" /> """, unsafe_allow_html=True)
col_3, col_4 = st.columns(2)
with col_3:
    comparison_rent_by_prop = graphs.get_comparison_rent_by_property(df=filtered_data)
    rental_trend_by_type_fig, aggregated_df_average_rental_by_type = graphs.create_rent_trend_over_time_by_type(df=filtered_data)
    st.plotly_chart(comparison_rent_by_prop, use_container_width=True)

with col_4:
    impact_prop_fig, impact_prop_df = graphs.get_impact_prop_features_on_rent(df=filtered_data)
    st.plotly_chart(impact_prop_fig, use_container_width=True)
    insight_impact_prop_text = insights.generate_dynamic_insights_impact_prop_on_rental(plot_data=impact_prop_df)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(f"<div class='insight-container'>{insight_impact_prop_text}</div>", unsafe_allow_html=True)

st.markdown("""<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8;" /> """, unsafe_allow_html=True)
col_5, col_6 = st.columns(2)
with col_5:
    st.markdown("<strong>Heatmap of Property Age vs. Rent Price Distribution</strong>", unsafe_allow_html=True)
    age_property_fig, age_df = graphs.generate_age_rent_heatmap(df=filtered_data)
    st.plotly_chart(age_property_fig, use_container_width=True)
    age_text = insights.generate_age_heatmap_insights(pivot_df=age_df)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(f"<div class='insight-container'>{age_text}</div>", unsafe_allow_html=True)

with col_6:
    energy_fig = graphs.plot_energy_efficiency_impact(df=filtered_data)
    st.plotly_chart(energy_fig, use_container_width=True)

st.markdown("""<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8;" /> """, unsafe_allow_html=True)
col_7, col_8 = st.columns(2)
with col_7:
    affordability_fig, avg_ratio_by_region_df = graphs.generate_affordability_graph(df=filtered_data)
    st.plotly_chart(affordability_fig, use_container_width=True)
    st.markdown(
        "<small>The average income data utilized in this analysis is derived from "
        "the <a href='https://en.wikipedia.org/wiki/List_of_German_states_by_household_income'>"
        "Wikipedia</a> page on household income in Germany, reflecting the "
        "household income per capita across the German states. </small>",
        unsafe_allow_html=True,
    )
    insight_affordability_text = insights.extract_insights_affordability(df=avg_ratio_by_region_df)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(f"<div class='insight-container'>{insight_affordability_text}</div>", unsafe_allow_html=True)

with col_8:
    st.plotly_chart(rental_trend_by_type_fig, use_container_width=True)
    insight_text_by_prop = insights.extract_insights_average_rent_over_time_by_type(aggregated_df=aggregated_df_average_rental_by_type)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(f"<div class='insight-container'>{insight_text_by_prop}</div>", unsafe_allow_html=True)
    # ----------------------------
# Đọc dữ liệu đã xử lý (Spark DataFrame)
main_df = process.get_clean_data()  # Spark DataFrame expected

# Chuyển sang pandas để hiển thị hoặc vẽ biểu đồ
main_pdf = main_df.toPandas()

st.subheader("📊 Sample of Cleaned Real Estate Data")
st.dataframe(main_pdf.head())

# Ví dụ: vẽ biểu đồ nhanh
graphs.plot_rent_distribution(main_pdf)

# Hoặc tạo insight tự động
insights.display_market_insights(main_pdf)