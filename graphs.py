import calendar
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
from typing import Tuple

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Import từ project gốc
from process_data import ProcessData, join_descriptions
from config import Config

# Tải stopwords
nltk.download("stopwords")


class CreateGraphs:
    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.process = ProcessData()
        self.config = Config()
        self.table_style = """
            <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            th {
                background-color: #f0f0f0;
            }
            </style>
        """
        self.german_stopwords = stopwords.words("german")

    # --------------------------------------------------------------------------------
    # ✅ 1. Tổng quan dữ liệu
    # --------------------------------------------------------------------------------
    def get_overview_info(self, main_df):
        total_rows = main_df.count()
        total_cols = len(main_df.columns)
        missing_cells = main_df.select(
            [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in main_df.columns]
        ).toPandas().sum(axis=1)[0]
        duplicate_rows = total_rows - main_df.dropDuplicates().count()
        total_size_in_memory = None  # Spark không lưu trong RAM như pandas

        data_overview = [
            {"Metric": "Number of variables", "Value": total_cols},
            {"Metric": "Number of observations", "Value": total_rows},
            {"Metric": "Missing cells", "Value": missing_cells},
            {"Metric": "Duplicate rows", "Value": duplicate_rows},
        ]

        table_html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for item in data_overview:
            table_html += f"<tr><td>{item['Metric']}</td><td>{item['Value']}</td></tr>"
        table_html += "</table><br>"
        return self.table_style + table_html

    # --------------------------------------------------------------------------------
    # ✅ 2. Tính các chỉ số cơ bản
    # --------------------------------------------------------------------------------
    @staticmethod
    def calculate_key_metrics(df):
        stats = df.agg(
            F.expr("percentile(baseRent, 0.5)").alias("median_rent"),
            F.expr("percentile(yearConstructed, 0.5)").alias("avg_year_constructed"),
            F.count("*").alias("num_properties")
        ).collect()[0]

        median_rent = stats["median_rent"]
        avg_year_constructed = stats["avg_year_constructed"]
        num_properties = stats["num_properties"]

        return median_rent, num_properties, avg_year_constructed

    # --------------------------------------------------------------------------------
    # ✅ 3. Phân phối giá thuê và năm xây dựng
    # --------------------------------------------------------------------------------
    @staticmethod
    def average_rent_year_distribution(df):
        df_pd = df.select("baseRent", "yearConstructed").toPandas()

        fig_rent = px.histogram(df_pd, x="baseRent", nbins=50, title="Distribution of Base Rent (€)")
        fig_year = px.histogram(df_pd, x="yearConstructed", nbins=30, title="Distribution of Year Constructed")
        return fig_rent, fig_year

    # --------------------------------------------------------------------------------
    # ✅ 4. Xu hướng giá thuê theo thời gian
    # --------------------------------------------------------------------------------
    @staticmethod
    def create_rent_trend_over_time(df):
        agg_df = df.groupBy("date").agg(
            F.expr("percentile(baseRent, 0.5)").alias("Average Base Rent"),
            F.expr("percentile(totalRent, 0.5)").alias("Average Total Rent")
        ).orderBy("date")

        agg_pd = agg_df.toPandas()
        fig = px.line(
            agg_pd,
            x="date",
            y=["Average Base Rent", "Average Total Rent"],
            title="Rent Trend Analysis Over Time",
            labels={"value": "Average Rent (€)", "variable": "Rent Type"},
            markers=True
        )
        return fig, agg_pd

    # --------------------------------------------------------------------------------
    # ✅ 5. So sánh giá thuê theo loại căn hộ
    # --------------------------------------------------------------------------------
    @staticmethod
    def get_comparison_rent_by_property(df):
        agg_df = df.groupBy("typeOfFlat").agg(
            F.expr("percentile(baseRent, 0.5)").alias("Average Rent")
        ).orderBy("Average Rent")

        agg_pd = agg_df.toPandas()
        fig = px.bar(
            agg_pd,
            x="typeOfFlat",
            y="Average Rent",
            title="Average Rent by Property Type",
            color="Average Rent",
            text_auto=True
        )
        return fig

    # --------------------------------------------------------------------------------
    # ✅ 6. So sánh giá thuê theo mùa
    # --------------------------------------------------------------------------------
    @staticmethod
    def get_comparison_rent_by_season(df):
        df = df.withColumn("month", F.month("date"))
        agg_df = df.groupBy("month").agg(
            F.expr("percentile(baseRent, 0.5)").alias("average_baseRent")
        ).orderBy("month")

        agg_pd = agg_df.toPandas()
        month_names = {i: name for i, name in enumerate(calendar.month_abbr) if i}
        agg_pd["month_name"] = agg_pd["month"].map(month_names)

        fig = px.line(
            agg_pd,
            x="month_name",
            y="average_baseRent",
            title="Seasonal Variations in Average Rent Prices",
            markers=True
        )
        return fig, agg_pd

    # --------------------------------------------------------------------------------
    # ✅ 7. Ảnh hưởng của đặc điểm nhà lên giá thuê
    # --------------------------------------------------------------------------------
    @staticmethod
    def get_impact_prop_features_on_rent(df):
        features = ["balcony", "garden", "noParkSpaces", "cellar", "lift"]
        results = []

        df_pd = df.select(features + ["baseRent"]).toPandas()

        for f in features:
            group_yes = df_pd[df_pd[f] == True]["baseRent"].dropna()
            group_no = df_pd[df_pd[f] == False]["baseRent"].dropna()

            if len(group_yes) > 2 and len(group_no) > 2:
                _, p_value = ttest_ind(group_yes, group_no, equal_var=False)
                results.append({"Feature": f, "With": group_yes.median(), "Without": group_no.median(), "P-Value": p_value})

        plot_df = pd.DataFrame(results)
        fig = px.bar(
            plot_df.melt(id_vars=["Feature", "P-Value"], var_name="Condition", value_name="Average Base Rent"),
            x="Feature", y="Average Base Rent", color="Condition",
            title="Impact of Property Features on Average Base Rent",
            hover_data=["P-Value"]
        )
        return fig, plot_df

    # --------------------------------------------------------------------------------
    # ✅ 8. Heatmap tuổi nhà vs giá thuê
    # --------------------------------------------------------------------------------
    def generate_age_rent_heatmap(self, df):
        current_year = datetime.now().year
        df = df.withColumn("Age", F.lit(current_year) - F.col("yearConstructed"))

        df_pd = df.select("Age", "baseRent").toPandas()
        df_pd["Age Group"] = df_pd["Age"].apply(lambda x: self.process.categorize_age(x, [0,5,10,20,30,50,100,150]))
        df_pd["Rent Range"] = df_pd["baseRent"].apply(lambda x: self.process.categorize_rent(x, range(0, int(df_pd["baseRent"].max()), 200)))

        pivot = df_pd.pivot_table(index="Rent Range", columns="Age Group", values="baseRent", aggfunc="count").fillna(0)
        fig = px.imshow(pivot, labels=dict(x="Age Group", y="Rent Range", color="Count"))
        fig.update_xaxes(side="top")
        return fig, pivot

    # --------------------------------------------------------------------------------
    # ✅ 9. WordCloud mô tả
    # --------------------------------------------------------------------------------
    def generate_wordcloud(self, df, max_words):
        desc_pd = df.select("description").dropna().toPandas()
        descriptions = desc_pd["description"].tolist()

        processed_text, most_occur = join_descriptions(
            descriptions=descriptions,
            stopwords=self.german_stopwords + self.config.additional_stopwords
        )

        wordcloud = WordCloud(
            stopwords=self.german_stopwords + self.config.additional_stopwords,
            max_words=max_words, background_color="white",
            width=800, height=600
        ).generate(processed_text)

        fig_wc, ax = plt.subplots(figsize=(10, 7), dpi=120)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        df_most = pd.DataFrame(most_occur, columns=["Word", "Frequency"]).sort_values("Frequency", ascending=False)
        fig_bar = px.bar(df_most.head(20), x="Frequency", y="Word", orientation="h", title="Most Frequent Words")
        return fig_wc, fig_bar
