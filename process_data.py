# ======================================================
# process_data.py — sử dụng PySpark thay cho Polars
# ======================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, DateType
from pyspark.sql.window import Window

import re, string
from collections import Counter
import warnings
from config import Config

warnings.filterwarnings('ignore')


# ======================================================
# 1️⃣ Class chính: Xử lý dữ liệu bằng PySpark
# ======================================================
class ProcessDataSpark:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = Config()

    # ------------------------------------------------------
    # 1. Đọc dữ liệu
    # ------------------------------------------------------
    def read_data(self):
        """
        Đọc file CSV bằng Spark, bỏ các cột không cần, xử lý giá trị null.
        """
        csv_path = str(self.config.input_path.joinpath(self.config.csv_file_name))

        df = (
            self.spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("nullValue", "NA")
            .csv(csv_path)
        )

        # Bỏ các cột không cần
        drop_cols = [c for c in self.config.ignore_columns if c in df.columns]
        if drop_cols:
            df = df.drop(*drop_cols)

        return df

    # ------------------------------------------------------
    # 2. Ép kiểu và xử lý cột
    # ------------------------------------------------------
    def convert_columns(self, df):
        """
        Chuyển kiểu dữ liệu của các cột sang dạng phù hợp.
        """
        if "yearConstructed" in df.columns:
            df = df.withColumn("yearConstructed", F.col("yearConstructed").cast(IntegerType()))

        if "totalRent" in df.columns:
            df = df.withColumn("totalRent", F.regexp_replace("totalRent", ",", ".").cast(FloatType()))

        if "date" in df.columns:
            df = df.withColumn("date", F.to_date(F.col("date"), "MMMYY"))  # ví dụ: 'Jan20' → ngày

        return df

    # ------------------------------------------------------
    # 3. Chuẩn hóa giá trị text
    # ------------------------------------------------------
    def replace_values(self, df):
        """
        Chuẩn hóa tên cột và giá trị text.
        """
        text_cols = [
            "heatingType", "regio1", "firingTypes",
            "condition", "typeOfFlat", "energyEfficiencyClass"
        ]

        for col in text_cols:
            if col in df.columns:
                df = df.withColumn(col, F.regexp_replace(F.col(col), "_", " "))

        # Đổi tên cột regio1 -> region
        if "regio1" in df.columns:
            df = df.withColumnRenamed("regio1", "region")

        return df

    # ------------------------------------------------------
    # 4. Pipeline làm sạch dữ liệu
    # ------------------------------------------------------
    def get_clean_data(self):
        """
        Pipeline chính: đọc, chuẩn hóa, ép kiểu, lọc dữ liệu.
        """
        df = self.read_data()
        df = self.replace_values(df)
        df = self.convert_columns(df)

        # Tính mean và std của baseRent
        rent_stats = df.select(
            F.mean("baseRent").alias("mean_rent"),
            F.stddev("baseRent").alias("std_rent")
        ).collect()[0]

        mean_rent = rent_stats["mean_rent"]
        std_rent = rent_stats["std_rent"]

        # Lọc dữ liệu hợp lệ
        df = (
            df.filter((F.col("yearConstructed") <= 2020) & (F.col("yearConstructed") > 1930))
              .filter((F.col("baseRent") < (mean_rent + 0.095 * std_rent)) & (F.col("baseRent") > 100))
              .dropDuplicates()
        )

        return df

    # ------------------------------------------------------
    # 5. Lọc dữ liệu theo điều kiện
    # ------------------------------------------------------
    def filter_data(self, df, date_range=None, location=None, property_type=None, energy_type=None):
        """
        Lọc dữ liệu tương tác theo input từ người dùng.
        """
        if date_range:
            if len(date_range) == 1:
                start_date, end_date = date_range[0], date_range[0]
            elif len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = None, None

            if start_date and end_date:
                df = df.filter(F.col("date").between(start_date, end_date))

        if location and location != "All":
            df = df.filter(F.col("region") == location)

        if energy_type and energy_type != "All":
            df = df.filter(F.col("energyEfficiencyClass") == energy_type)

        if property_type and property_type != "All":
            if isinstance(property_type, list):
                df = df.filter(F.col("typeOfFlat").isin(property_type))
            else:
                df = df.filter(F.col("typeOfFlat") == property_type)

        return df

    # ------------------------------------------------------
    # 6. Lọc bảng chi tiết
    # ------------------------------------------------------
    def filter_table(self, df, rent_range, year_range, space_range, sort_column=None):
        """
        Lọc dữ liệu cho bảng chi tiết.
        """
        df = (
            df.filter((F.col("baseRent") >= rent_range[0]) & (F.col("baseRent") <= rent_range[1]))
              .filter((F.col("yearConstructed") >= year_range[0]) & (F.col("yearConstructed") <= year_range[1]))
              .filter((F.col("livingSpace") >= space_range[0]) & (F.col("livingSpace") <= space_range[1]))
        )
        if sort_column and sort_column != "None":
            df = df.orderBy(sort_column)

        return df

    # ------------------------------------------------------
    # 7. WordCloud mô tả
    # ------------------------------------------------------
    def join_descriptions(self, descriptions, stopwords, most_common_n=1000):
        """
        Gộp các mô tả văn bản để tạo WordCloud.
        """
        config = self.config
        pattern = r'[0-9]'
        all_words = []

        for desc in descriptions:
            clean_text = re.sub(pattern, '', desc.translate(str.maketrans('', '', string.punctuation)))
            tokens = clean_text.split()
            for token in tokens:
                word = token.strip().lower()
                word = config.replace_dict.get(word, word)
                if word and word not in stopwords:
                    all_words.append(word)

        counter = Counter(all_words)
        most_common_words = counter.most_common(most_common_n)
        text = ' '.join([w for w, _ in most_common_words])
        return text, most_common_words

    # ------------------------------------------------------
    # 8. Hàm hỗ trợ
    # ------------------------------------------------------
    def categorize_age(self, age, bins):
        for i, b in enumerate(bins):
            if age <= b:
                return f"{bins[i-1] if i>0 else 0}-{b}"
        return f"{bins[-1]}+"

    def categorize_rent(self, rent, bins):
        for i, b in enumerate(bins):
            if rent <= b:
                return f"{bins[i-1] if i>0 else 0}-{b}"
        return f"{bins[-1]}+"

    def get_latitude(self, region):
        return self.config.region_coordinates.get(region, {}).get("lat", None)

    def get_longitude(self, region):
        return self.config.region_coordinates.get(region, {}).get("lon", None)


# ======================================================
# 2️⃣ Hàm join_descriptions để dùng trong graphs.py
# ======================================================
def join_descriptions(df, column_name="description"):
    """
    Gộp toàn bộ mô tả văn bản trong DataFrame (pandas hoặc polars)
    để tạo WordCloud.
    """
    return " ".join(df[column_name].dropna().astype(str))


# ======================================================
# 3️⃣ Alias cho import từ app.py
# ======================================================
ProcessData = ProcessDataSpark
