# dynamic_insights.py
import numpy as np
import calendar
import pandas as pd
from typing import Union
from process_data import ProcessData
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame


class CreateInsights:
    def __init__(self):
        self.process = ProcessData()
        self.insight_container_style = """
                                        <style>
                                        .insight-container {
                                            padding: 10px;
                                            margin: 5px 0;
                                            border-left: 5px solid #4CAF50;
                                            background-color: #f8f9fa;
                                            animation: fadeIn 1s linear;
                                        }
                                        @keyframes fadeIn {
                                                from { opacity: 0; }
                                                to { opacity: 1; }
                                            }
                                        </style>
                                        """

    # ---------------- helper ----------------
    @staticmethod
    def _ensure_pandas(df: Union[SparkDataFrame, pd.DataFrame]):
        """
        Nếu đầu vào là Spark DataFrame -> chuyển sang pandas.
        Nếu là pandas -> trả lại nguyên dạng.
        """
        if df is None:
            return pd.DataFrame()
        if isinstance(df, SparkDataFrame):
            return df.toPandas()
        if isinstance(df, pd.DataFrame):
            return df
        # Nếu là các dạng khác (like polars), cố gắng chuyển qua pandas nếu có to_pandas/toPandas
        try:
            if hasattr(df, "to_pandas"):
                return df.to_pandas()
            if hasattr(df, "toPandas"):
                return df.toPandas()
        except Exception:
            pass
        # fallback
        return pd.DataFrame(df)

    # ---------------- Insights helpers ----------------
    @staticmethod
    def generate_histogram_insights(df: Union[SparkDataFrame, pd.DataFrame], column_name: str, currency: bool = False) -> str:
        """
        Sinh insight cơ bản cho cột numeric (mode, mean, median, max, min).
        Hỗ trợ cả Spark DataFrame và pandas DataFrame.
        """
        df_pd = CreateInsights._ensure_pandas(df)

        # Nếu dataframe rỗng
        if df_pd.empty or column_name not in df_pd.columns:
            return f"No data available for column {column_name}."

        # Most common (mode) - chọn giá trị có frequency cao nhất
        try:
            most_common_bin = df_pd[column_name].mode().iat[0]
        except Exception:
            # If mode fails, fallback to value_counts
            most_common_bin = df_pd[column_name].value_counts().idxmax()

        average_value = int(df_pd[column_name].mean())
        median_value = int(df_pd[column_name].median())
        max_value = int(df_pd[column_name].max())
        min_value = int(df_pd[column_name].min())

        currency_symbol = "€" if currency else ""

        column_map = {
            "baseRent": "Base Rent",
            "yearConstructed": "Year of Construction",
        }
        col_label = column_map.get(column_name, column_name)

        insights_text = (
            f"The most common {col_label} value is around {currency_symbol}{most_common_bin}. "
            f"The average {col_label} is {currency_symbol}{average_value}, "
            f"with a median of {currency_symbol}{median_value}. "
            f"The maximum {col_label} found in the dataset is {currency_symbol}{max_value}, "
            f"while the minimum is {currency_symbol}{min_value}."
        )

        return insights_text

    def extract_insights_average_rent_over_time(self, aggregated_df: Union[SparkDataFrame, pd.DataFrame]) -> str:
        """
        aggregated_df expected to have columns:
        'date', 'Average Base Rent', 'Average Total Rent' and be sorted by date.
        """
        df_pd = CreateInsights._ensure_pandas(aggregated_df)

        if df_pd.empty:
            return "No data available for the selected period."

        # Ensure sorted by date
        if "date" in df_pd.columns:
            try:
                df_pd = df_pd.sort_values("date")
            except Exception:
                pass

        first_base = df_pd["Average Base Rent"].iloc[0]
        last_base = df_pd["Average Base Rent"].iloc[-1]
        first_total = df_pd["Average Total Rent"].iloc[0]
        last_total = df_pd["Average Total Rent"].iloc[-1]

        rent_increase_base = last_base - first_base
        rent_increase_total = last_total - first_total

        percentage_increase_base = (rent_increase_base / first_base) * 100 if first_base != 0 else 0.0
        percentage_increase_total = (rent_increase_total / first_total) * 100 if first_total != 0 else 0.0

        # Period
        period_start = df_pd["date"].iloc[0]
        period_end = df_pd["date"].iloc[-1]

        # Format dates (if Timestamp or datetime)
        try:
            period_start_str = pd.to_datetime(period_start).strftime("%Y-%m-%d")
        except Exception:
            period_start_str = str(period_start)
        try:
            period_end_str = pd.to_datetime(period_end).strftime("%Y-%m-%d")
        except Exception:
            period_end_str = str(period_end)

        insight_text = (
            f"<strong>Period covered:</strong> {period_start_str} to {period_end_str} <br>"
            f"Average base rent changed by {float(rent_increase_base):.2f}€ ({float(percentage_increase_base):.2f}%) over the period. "
            f"Average total rent changed by {float(rent_increase_total):.2f}€ ({float(percentage_increase_total):.2f}%) over the period. "
        )

        if percentage_increase_base > 10:
            insight_text += (
                "<br><strong>Significant Increase in Base Rent:</strong> "
                "The base rent has seen a significant increase, suggesting a tightening rental market."
            )
        elif percentage_increase_base < -10:
            insight_text += (
                "<br><strong>Significant Decrease in Base Rent:</strong> "
                "A notable decrease in base rent could indicate a shift towards a renter's market."
            )
        else:
            insight_text += (
                "<br><strong>🔄 Stable Rent Prices:</strong> "
                "Rent prices have remained relatively stable, indicating a balanced market condition."
            )

        return insight_text

    def extract_insights_average_rent_over_time_by_type(self, aggregated_df: Union[SparkDataFrame, pd.DataFrame]) -> str:
        """
        aggregated_df expected to contain columns: 'date', 'typeOfFlat',
        'Average Base Rent', 'Average Total Rent'
        """
        df_pd = CreateInsights._ensure_pandas(aggregated_df)

        if df_pd.empty:
            return "No data available."

        # Sort by date and type
        if "date" in df_pd.columns:
            df_pd = df_pd.sort_values(["date", "typeOfFlat"])

        # Unique property types (use process helper to remove None)
        property_types = self.process.remove_none_from_list(df_pd["typeOfFlat"].dropna().unique().tolist())

        insight_text = ""

        for property_type in property_types:
            df_filtered = df_pd[df_pd["typeOfFlat"] == property_type]
            if df_filtered.empty:
                insight_text += f"<br>No data available for property type '{property_type}'."
                continue

            first_base = df_filtered["Average Base Rent"].iloc[0]
            last_base = df_filtered["Average Base Rent"].iloc[-1]
            first_total = df_filtered["Average Total Rent"].iloc[0]
            last_total = df_filtered["Average Total Rent"].iloc[-1]

            rent_increase_base = last_base - first_base
            rent_increase_total = last_total - first_total

            percentage_increase_base = (rent_increase_base / first_base) * 100 if first_base != 0 else 0.0
            percentage_increase_total = (rent_increase_total / first_total) * 100 if first_total != 0 else 0.0

            period_start = df_filtered["date"].iloc[0]
            period_end = df_filtered["date"].iloc[-1]
            try:
                period_start_str = pd.to_datetime(period_start).strftime("%Y-%m-%d")
            except Exception:
                period_start_str = str(period_start)
            try:
                period_end_str = pd.to_datetime(period_end).strftime("%Y-%m-%d")
            except Exception:
                period_end_str = str(period_end)

            insight_text += (
                f"<br><strong>Insights for {property_type}</strong><br>"
                f"Period covered: {period_start_str} to {period_end_str}. "
                f"Average base rent changed by {float(rent_increase_base):.2f}€ ({float(percentage_increase_base):.2f}%) over the period. "
                f"Average total rent changed by {float(rent_increase_total):.2f}€ ({float(percentage_increase_total):.2f}%) over the period."
            )

        return insight_text

    @staticmethod
    def extract_insights_average_rent_by_season(monthly_avg_rent: Union[SparkDataFrame, pd.DataFrame]) -> str:
        """
        monthly_avg_rent: can be Spark DF with columns 'month' and 'average_baseRent' or pandas DF
        """
        df_pd = CreateInsights._ensure_pandas(monthly_avg_rent)

        if df_pd.empty or "month" not in df_pd.columns or "average_baseRent" not in df_pd.columns:
            return "No data available for the selected period."

        # Find max/min month
        max_row = df_pd.loc[df_pd["average_baseRent"].idxmax()]
        min_row = df_pd.loc[df_pd["average_baseRent"].idxmin()]

        max_rent_month = int(max_row["month"])
        min_rent_month = int(min_row["month"])
        max_rent_value = float(max_row["average_baseRent"])
        min_rent_value = float(min_row["average_baseRent"])

        max_rent_month_name = calendar.month_name[max_rent_month]
        min_rent_month_name = calendar.month_name[min_rent_month]

        insight_text = (
            f"The highest average base rent was observed in {max_rent_month_name} at €{max_rent_value:.2f}, "
            f"while the lowest was in {min_rent_month_name} at €{min_rent_value:.2f}. "
        )

        if max_rent_month in [6, 7, 8]:
            insight_text += "This peak during the summer months could suggest higher demand for rentals. "
        if min_rent_month in [12, 1, 2]:
            insight_text += "The decrease in rent prices during winter may indicate a lower demand. "

        return insight_text

    @staticmethod
    def generate_dynamic_insights_impact_prop_on_rental(plot_data: Union[SparkDataFrame, pd.DataFrame]) -> str:
        """
        plot_data expected to have columns:
        'Feature', 'Average Base Rent', 'P-Value' and be structured with pairs (With/Without)
        """
        df_pd = CreateInsights._ensure_pandas(plot_data)

        if df_pd.empty:
            return "No data available for property feature impact analysis."

        insights_text = ""
        # iterate by pairs (assumes ordering in plot_data as in original)
        # find unique features and summarise
        features = df_pd["Feature"].unique().tolist()
        for feature in features:
            sub = df_pd[df_pd["Feature"] == feature].reset_index(drop=True)
            if sub.shape[0] < 2:
                continue
            p_value = sub["P-Value"].iloc[0]
            significant = "is statistically significant" if p_value < 0.05 else "is not statistically significant"
            insights_text += f"\n The difference in rent for properties with and without **{feature}** {significant} (p-value: {p_value:.3f})."

            if p_value < 0.05:
                # determine which condition has higher average rent
                # sub assumed to have two rows: With and Without (or similar)
                idx_max = sub["Average Base Rent"].idxmax()
                condition_with = sub.loc[idx_max, "Condition"] if "Condition" in sub.columns else ("With" if sub["Average Base Rent"].iloc[0] > sub["Average Base Rent"].iloc[1] else "Without")
                difference = abs(sub["Average Base Rent"].iloc[0] - sub["Average Base Rent"].iloc[1])
                insights_text += (
                    f"\n Properties {condition_with} **{feature}** have an average rent difference of €{difference:.2f} compared to those without.\n"
                )

        return insights_text

    def generate_age_heatmap_insights(self, pivot_df: pd.DataFrame) -> str:
        """
        pivot_df is expected to be a pandas pivot table (rent range x age group) with counts.
        """
        if pivot_df is None or pivot_df.size == 0:
            return "No data available to generate heatmap insights."

        # find location of max
        max_loc = np.unravel_index(np.argmax(pivot_df.values, axis=None), pivot_df.shape)
        max_rent_range = pivot_df.index[max_loc[0]]
        max_age_group = pivot_df.columns[max_loc[1]]

        insight_text = (
            f"The highest concentration of properties is found within the rent range {max_rent_range} € "
            f"for properties in the age group of {max_age_group} years. "
            f"This suggests a strong preference or availability in this segment."
        )

        general_trends = (
            " The heatmap indicates that newer properties (lower age groups) tend to have higher rent ranges, "
            "while older properties show a wider distribution of rent prices."
        )

        return insight_text + general_trends

    @staticmethod
    def extract_insights_affordability(df: Union[SparkDataFrame, pd.DataFrame]) -> str:
        df_pd = CreateInsights._ensure_pandas(df)

        if df_pd.empty or "Average Rent-to-Income Ratio" not in df_pd.columns:
            return "No affordability data available."

        high_ratio_regions = df_pd[df_pd["Average Rent-to-Income Ratio"] > 30]["region"].unique().tolist()
        low_ratio_regions = df_pd[df_pd["Average Rent-to-Income Ratio"] <= 30]["region"].unique().tolist()

        high_ratio_text = ", ".join(high_ratio_regions) if high_ratio_regions else "None"
        low_ratio_text = ", ".join(low_ratio_regions) if low_ratio_regions else "None"

        insights_text = (
            f"In particular, regions such as {high_ratio_text} exhibit rent-to-income ratios that might pose affordability challenges. "
            f"On the other hand, regions like {low_ratio_text} appear to offer a more balanced economic environment for renters."
        )

        return insights_text

    @staticmethod
    def generate_map_insights(final_df: Union[SparkDataFrame, pd.DataFrame]) -> str:
        final_pd = CreateInsights._ensure_pandas(final_df)

        if final_pd.empty:
            return "No data available for the selected filters."

        # require columns: region, MedianRent, CountOfProperties
        if not all(col in final_pd.columns for col in ["region", "MedianRent", "CountOfProperties"]):
            return "Insufficient data to generate map insights."

        highest_rent_region = final_pd.loc[final_pd["MedianRent"].idxmax(), "region"]
        lowest_rent_region = final_pd.loc[final_pd["MedianRent"].idxmin(), "region"]
        highest_rent_value = final_pd["MedianRent"].max()
        lowest_rent_value = final_pd["MedianRent"].min()

        most_properties_region = final_pd.loc[final_pd["CountOfProperties"].idxmax(), "region"]
        least_properties_region = final_pd.loc[final_pd["CountOfProperties"].idxmin(), "region"]
        most_properties_count = final_pd["CountOfProperties"].max()
        least_properties_count = final_pd["CountOfProperties"].min()

        insights_text = (
            f"Regions with the highest and lowest median rents are <strong>{highest_rent_region}</strong> (€{highest_rent_value:.2f}) "
            f"and <strong>{lowest_rent_region}</strong> (€{lowest_rent_value:.2f}), respectively. "
            f"<strong>{most_properties_region}</strong> stands out with the highest number of properties available for rent ({most_properties_count}), "
            f"while <strong>{least_properties_region}</strong> has the least ({least_properties_count})."
        )

        return insights_text