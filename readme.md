Airline Customer Activity Analysis
===================================
---

### Overview

This project analyzes an airline's customer activity data using PySpark. The analysis includes data cleaning, transformation, and generating insightful visualizations to understand customer behavior and business performance.

### Dataset

The analysis uses three datasets:

1. **calendar.csv**: Contains date-related information.
2. **customer_loyalty_history.csv**: Details customer demographics and loyalty program history.
3. **customer_flight_activity.csv**: Records flight activities per customer over time.

### Tools & Libraries

- **PySpark**: For data processing and analysis.
- **Matplotlib**: For generating visualizations.
- **Pandas**: For handling dataframes during visualization.

### Script Overview (`customer_analysis.py`)

1. **Initialization**:
   - Initializes a Spark session.
   - Defines file paths for input data and output directories.

2. **Data Loading**:
   - Reads the three CSV files into Spark DataFrames with appropriate schema inference.

3. **Data Cleaning & Transformation**:
   - Converts date strings to `date` type.
   - Casts numerical fields to appropriate data types.
   - Handles missing values, specifically filling `customer_lifetime_value` with 0 where missing.
   - Joins the `customer_loyalty_history` and `customer_flight_activity` DataFrames on `loyalty_number`.

4. **Analysis Using Spark SQL**:
   - **Average Customer Lifetime Value (CLV) by Gender**:
     - Calculates the average CLV for each gender.
     - Generates a bar chart visualization.
   - **Total Flights per Year**:
     - Sums the total number of flights each year.
     - Generates a line chart visualization.
   - **Points Redeemed vs Points Accumulated**:
     - Aggregates total points accumulated and redeemed per customer.
     - Generates a scatter plot to show the relationship.
   - **Customer Distribution by Province**:
     - Counts the number of customers in each province.
     - Generates a bar chart visualization.

5. **Output**:
   - Saves the generated visualizations in the `output/graphs/` directory.
   - Exports the analysis results as CSV files in the `output/data/` directory.

6. **Termination**:
   - Stops the Spark session after completing the analysis.

### How to Run the Script

1. **Prerequisites**:
   - Ensure Python is installed (version 3.6 or higher recommended).
   - Install PySpark:
     ```
     pip install pyspark
     ```
   - Install Matplotlib:
     ```
     pip install matplotlib
     ```
   - Ensure the dataset CSV files are placed in the `data/` directory as per the folder structure.

2. **Execution**:
   - Navigate to the project directory in the terminal.
   - Run the script using Python:
     ```
     python customer_analysis.py
     ```
   - The script will process the data, perform analyses, generate graphs, and save the outputs in the `output/` directory.

### Analysis Explanation

1. **Average Customer Lifetime Value by Gender**:
   - **Insight**: Determines if there's a significant difference in the lifetime value between male and female customers.
   - **Graph**: A bar chart showing average CLV for each gender.

2. **Total Flights per Year**:
   - **Insight**: Observes the trend in flight bookings over the years to identify growth or decline patterns.
   - **Graph**: A line chart depicting the total number of flights booked each year.

3. **Points Redeemed vs Points Accumulated**:
   - **Insight**: Analyzes the relationship between points earned and points redeemed by customers, indicating engagement levels.
   - **Graph**: A scatter plot illustrating the correlation between points accumulated and redeemed.

4. **Customer Distribution by Province**:
   - **Insight**: Identifies the geographical distribution of customers to aid in targeted marketing strategies.
   - **Graph**: A bar chart showing the number of customers in each province.

### Conclusion

The analysis provides valuable insights into customer behavior, flight activity trends, and geographic distribution. These insights can inform strategic decisions in marketing, customer relationship management, and service offerings to enhance customer satisfaction and business performance.

---


