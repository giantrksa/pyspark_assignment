# customer_analysis.py

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, when, avg, sum as spark_sum, count
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def initialize_spark():
    """Initialize and return a Spark session."""
    spark = SparkSession.builder \
        .appName("AirlineCustomerAnalysis") \
        .getOrCreate()
    return spark

def define_paths(base_dir):
    """Define and return file paths based on the base directory."""
    data_dir = os.path.join(base_dir, "data")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    loyalty_history_path = os.path.join(data_dir, "customer_loyalty_history.csv")
    flight_activity_path = os.path.join(data_dir, "customer_flight_activity.csv")
    
    output_dir = os.path.join(base_dir, "output")
    graphs_dir = os.path.join(output_dir, "graphs")
    data_output_dir = os.path.join(output_dir, "data")
    
    # Create output directories if they don't exist
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    
    return calendar_path, loyalty_history_path, flight_activity_path, graphs_dir, data_output_dir

def load_dataframes(spark, calendar_path, loyalty_history_path, flight_activity_path):
    """Load CSV files into Spark DataFrames."""
    calendar_df = spark.read.csv(calendar_path, header=True, inferSchema=True)
    loyalty_history_df = spark.read.csv(loyalty_history_path, header=True, inferSchema=True)
    flight_activity_df = spark.read.csv(flight_activity_path, header=True, inferSchema=True)
    return calendar_df, loyalty_history_df, flight_activity_df

def clean_data(calendar_df, loyalty_history_df, flight_activity_df):
    """Clean and transform the DataFrames."""
    # Clean calendar_df
    calendar_df = calendar_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
                             .withColumn("start_of_the_year", to_date(col("start_of_the_year"), "yyyy-MM-dd")) \
                             .withColumn("start_of_the_quarter", to_date(col("start_of_the_quarter"), "yyyy-MM-dd")) \
                             .withColumn("start_of_the_month", to_date(col("start_of_the_month"), "yyyy-MM-dd"))
    
    # Clean loyalty_history_df
    loyalty_history_df = loyalty_history_df.withColumn("salary", col("salary").cast("double")) \
                                           .withColumn("enrollment_year", col("enrollment_year").cast("integer")) \
                                           .withColumn("enrollment_month", col("enrollment_month").cast("integer")) \
                                           .withColumn("cancellation_year", col("cancellation_year").cast("integer")) \
                                           .withColumn("cancellation_month", col("cancellation_month").cast("integer"))
    
    # Handle missing CLV by filling with 0
    loyalty_history_df = loyalty_history_df.withColumn("customer_lifetime_value", 
                                                       when(col("customer_lifetime_value").isNull(), 0)
                                                       .otherwise(col("customer_lifetime_value")))
    
    # Clean flight_activity_df
    flight_activity_df = flight_activity_df.withColumn("year", col("year").cast("integer")) \
                                           .withColumn("month", col("month").cast("integer")) \
                                           .withColumn("total_flights", col("total_flights").cast("integer")) \
                                           .withColumn("distance", col("distance").cast("double")) \
                                           .withColumn("points_accumulated", col("points_accumulated").cast("double")) \
                                           .withColumn("points_redeemed", col("points_redeemed").cast("double")) \
                                           .withColumn("dollar_cost_points_redeemed", col("dollar_cost_points_redeemed").cast("double"))
    
    return calendar_df, loyalty_history_df, flight_activity_df

def join_dataframes(loyalty_history_df, flight_activity_df):
    """Join loyalty history and flight activity DataFrames."""
    customer_df = loyalty_history_df.join(flight_activity_df, on="loyalty_number", how="left")
    return customer_df

def register_temp_views(customer_df, calendar_df):
    """Register Spark DataFrames as temporary views for SQL queries."""
    customer_df.createOrReplaceTempView("customer")
    calendar_df.createOrReplaceTempView("calendar")

def plot_bar(df_pd, x, y, title, xlabel, ylabel, filename, colors=None, rotation=0):
    """Plot and save a bar chart."""
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_pd, x=x, y=y, palette=colors)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_line(df_pd, x, y, title, xlabel, ylabel, filename):
    """Plot and save a line chart."""
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_pd, x=x, y=y, marker='o')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_scatter(df_pd, x, y, title, xlabel, ylabel, filename):
    """Plot and save a scatter plot."""
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df_pd, x=x, y=y, alpha=0.6)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_heatmap(df_pd, title, filename):
    """Plot and save a correlation heatmap."""
    plt.figure(figsize=(12,10))
    corr = df_pd.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def perform_analysis(spark, graphs_dir, data_output_dir):
    """Perform various analyses and generate visualizations."""
    # Analysis 1: Average Customer Lifetime Value (CLV) by Gender
    clv_gender = spark.sql("""
        SELECT gender, AVG(customer_lifetime_value) as avg_clv
        FROM customer
        GROUP BY gender
    """)
    clv_gender_pd = clv_gender.toPandas()
    plot_bar(
        df_pd=clv_gender_pd,
        x='gender',
        y='avg_clv',
        title='Average Customer Lifetime Value by Gender',
        xlabel='Gender',
        ylabel='Average CLV (CAD)',
        filename=os.path.join(graphs_dir, 'avg_clv_by_gender.png'),
        colors='Blues'
    )
    clv_gender.write.csv(os.path.join(data_output_dir, "avg_clv_by_gender.csv"), header=True, mode='overwrite')
    
    # Analysis 2: CLV by Education Level
    clv_education = spark.sql("""
        SELECT education, AVG(customer_lifetime_value) as avg_clv
        FROM customer
        GROUP BY education
        ORDER BY avg_clv DESC
    """)
    clv_education_pd = clv_education.toPandas()
    plot_bar(
        df_pd=clv_education_pd,
        x='education',
        y='avg_clv',
        title='Average Customer Lifetime Value by Education Level',
        xlabel='Education Level',
        ylabel='Average CLV (CAD)',
        filename=os.path.join(graphs_dir, 'avg_clv_by_education.png'),
        colors='Greens',
        rotation=45
    )
    clv_education.write.csv(os.path.join(data_output_dir, "avg_clv_by_education.csv"), header=True, mode='overwrite')
    
    # Analysis 3: CLV Over Enrollment Years
    clv_year = spark.sql("""
        SELECT enrollment_year, AVG(customer_lifetime_value) as avg_clv
        FROM customer
        WHERE enrollment_year IS NOT NULL
        GROUP BY enrollment_year
        ORDER BY enrollment_year
    """)
    clv_year_pd = clv_year.toPandas()
    plot_line(
        df_pd=clv_year_pd,
        x='enrollment_year',
        y='avg_clv',
        title='Average CLV Over Enrollment Years',
        xlabel='Enrollment Year',
        ylabel='Average CLV (CAD)',
        filename=os.path.join(graphs_dir, 'avg_clv_over_years.png')
    )
    clv_year.write.csv(os.path.join(data_output_dir, "avg_clv_over_years.csv"), header=True, mode='overwrite')
    
    # Analysis 4: Salary vs. CLV
    salary_clv = spark.sql("""
        SELECT salary, customer_lifetime_value
        FROM customer
        WHERE salary IS NOT NULL AND customer_lifetime_value > 0
    """)
    salary_clv_pd = salary_clv.toPandas()
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=salary_clv_pd, x='salary', y='customer_lifetime_value', alpha=0.6)
    plt.title('Salary vs. Customer Lifetime Value', fontsize=16)
    plt.xlabel('Salary (CAD)', fontsize=14)
    plt.ylabel('Customer Lifetime Value (CAD)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'salary_vs_clv.png'))
    plt.close()
    salary_clv.write.csv(os.path.join(data_output_dir, "salary_vs_clv.csv"), header=True, mode='overwrite')
    
    # Analysis 5: Marital Status vs. CLV
    clv_marital = spark.sql("""
        SELECT marital_status, AVG(customer_lifetime_value) as avg_clv
        FROM customer
        GROUP BY marital_status
    """)
    clv_marital_pd = clv_marital.toPandas()
    plot_bar(
        df_pd=clv_marital_pd,
        x='marital_status',
        y='avg_clv',
        title='Average Customer Lifetime Value by Marital Status',
        xlabel='Marital Status',
        ylabel='Average CLV (CAD)',
        filename=os.path.join(graphs_dir, 'avg_clv_by_marital_status.png'),
        colors='Purples'
    )
    clv_marital.write.csv(os.path.join(data_output_dir, "avg_clv_by_marital_status.csv"), header=True, mode='overwrite')
    
    # Analysis 6: Points Accumulated and Redeemed Over Time
    points_time = spark.sql("""
        SELECT year, month, SUM(points_accumulated) as total_points_accumulated,
               SUM(points_redeemed) as total_points_redeemed
        FROM customer
        GROUP BY year, month
        ORDER BY year, month
    """)
    points_time_pd = points_time.toPandas()
    # Create a datetime column for better plotting
    points_time_pd['date'] = pd.to_datetime(points_time_pd[['year', 'month']].assign(DAY=1))
    
    plt.figure(figsize=(12,6))
    sns.lineplot(data=points_time_pd, x='date', y='total_points_accumulated', label='Points Accumulated', marker='o')
    sns.lineplot(data=points_time_pd, x='date', y='total_points_redeemed', label='Points Redeemed', marker='o')
    plt.title('Points Accumulated and Redeemed Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Points', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'points_over_time.png'))
    plt.close()
    points_time.write.csv(os.path.join(data_output_dir, "points_over_time.csv"), header=True, mode='overwrite')
    
    # Analysis 7: Correlation Heatmap of Numerical Variables
    numerical_cols = ['salary', 'customer_lifetime_value', 'total_flights', 'distance',
                      'points_accumulated', 'points_redeemed', 'dollar_cost_points_redeemed']
    correlation_df = spark.sql("""
        SELECT salary, customer_lifetime_value, total_flights, distance,
               points_accumulated, points_redeemed, dollar_cost_points_redeemed
        FROM customer
    """).select([col(c) for c in numerical_cols])
    
    # Drop rows with nulls to compute correlation
    correlation_df = correlation_df.na.drop()
    correlation_pd = correlation_df.toPandas()
    plot_heatmap(
        df_pd=correlation_pd,
        title='Correlation Heatmap of Numerical Variables',
        filename=os.path.join(graphs_dir, 'correlation_heatmap.png')
    )
    # Save correlation matrix as CSV
    corr_matrix = correlation_pd.corr()
    corr_matrix.to_csv(os.path.join(data_output_dir, "correlation_matrix.csv"), index=True)
    
    # Analysis 8: Customer Distribution by Province (Existing Analysis Improved)
    customer_province = spark.sql("""
        SELECT province, COUNT(*) as customer_count
        FROM customer
        GROUP BY province
        ORDER BY customer_count DESC
    """)
    customer_province_pd = customer_province.toPandas()
    plot_bar(
        df_pd=customer_province_pd,
        x='province',
        y='customer_count',
        title='Customer Distribution by Province',
        xlabel='Province',
        ylabel='Number of Customers',
        filename=os.path.join(graphs_dir, 'customer_distribution_by_province.png'),
        colors='Oranges',
        rotation=45
    )
    customer_province.write.csv(os.path.join(data_output_dir, "customer_distribution_by_province.csv"), header=True, mode='overwrite')
    
    # Analysis 9: Total Flights per Year (Existing Analysis Improved)
    total_flights_year = spark.sql("""
        SELECT year, SUM(total_flights) as total_flights
        FROM customer
        GROUP BY year
        ORDER BY year
    """)
    total_flights_year_pd = total_flights_year.toPandas()
    plot_line(
        df_pd=total_flights_year_pd,
        x='year',
        y='total_flights',
        title='Total Flights per Year',
        xlabel='Year',
        ylabel='Total Flights',
        filename=os.path.join(graphs_dir, 'total_flights_per_year.png')
    )
    total_flights_year.write.csv(os.path.join(data_output_dir, "total_flights_per_year.csv"), header=True, mode='overwrite')
    
    # Analysis 10: Points Redeemed vs Points Accumulated (Existing Analysis Improved)
    points = spark.sql("""
        SELECT loyalty_number, SUM(points_accumulated) as total_points_accumulated,
               SUM(points_redeemed) as total_points_redeemed
        FROM customer
        GROUP BY loyalty_number
    """)
    points_pd = points.toPandas()
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=points_pd, x='total_points_accumulated', y='total_points_redeemed', alpha=0.6)
    plt.title('Points Redeemed vs Points Accumulated', fontsize=16)
    plt.xlabel('Total Points Accumulated', fontsize=14)
    plt.ylabel('Total Points Redeemed', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'points_redeemed_vs_accumulated.png'))
    plt.close()
    points.write.csv(os.path.join(data_output_dir, "points_redeemed_vs_accumulated.csv"), header=True, mode='overwrite')
    
def main():
    # Initialize Spark Session
    spark = initialize_spark()
    
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calendar_path, loyalty_history_path, flight_activity_path, graphs_dir, data_output_dir = define_paths(base_dir)
    
    # Load DataFrames
    calendar_df, loyalty_history_df, flight_activity_df = load_dataframes(spark, calendar_path, loyalty_history_path, flight_activity_path)
    
    # Clean Data
    calendar_df, loyalty_history_df, flight_activity_df = clean_data(calendar_df, loyalty_history_df, flight_activity_df)
    
    # Join DataFrames
    customer_df = join_dataframes(loyalty_history_df, flight_activity_df)
    
    # Register Temp Views
    register_temp_views(customer_df, calendar_df)
    
    # Perform Analysis and Generate Visualizations
    perform_analysis(spark, graphs_dir, data_output_dir)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
