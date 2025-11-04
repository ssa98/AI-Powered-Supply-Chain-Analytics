import numpy as np
import pandas as pd

df = pd.read_csv('add-your-data.csv')

# ✅ Clean up the existing columns first
df['Order_Demand'] = df['Order_Demand'].str.replace('[^0-9-]', '', regex=True)
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce').abs()  # convert to number, remove negatives
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with missing dates or demand
df = df.dropna(subset=['Date', 'Order_Demand'])

# ✅ Add new enhanced columns

# 1️⃣ Cost_Price between 10 and 50
df['Cost_Price'] = np.round(np.random.uniform(10, 50, len(df)), 2)

# 2️⃣ Selling_Price (20%–40% higher)
df['Selling_Price'] = np.round(df['Cost_Price'] * np.random.uniform(1.2, 1.4, len(df)), 2)

# 3️⃣ Supplier_Name (random 5 suppliers)
suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D', 'Supplier_E']
df['Supplier_Name'] = np.random.choice(suppliers, len(df))

# 4️⃣ Lead_Time_Days (3 to 15 days)
df['Lead_Time_Days'] = np.random.randint(3, 16, len(df))

# 5️⃣ Region mapped from Warehouse
warehouse_to_region = {
    'Whse_A': 'North',
    'Whse_B': 'South',
    'Whse_C': 'East',
    'Whse_D': 'West',
    'Whse_E': 'North',
    'Whse_F': 'South',
    'Whse_G': 'East',
    'Whse_H': 'West',
    'Whse_I': 'North',
    'Whse_J': 'South'
}
df['Region'] = df['Warehouse'].map(warehouse_to_region).fillna('Other')

# ✅ Derive extra business features 
df['Revenue'] = df['Order_Demand'] * df['Selling_Price']
df['Profit'] = df['Revenue'] - (df['Order_Demand'] * df['Cost_Price'])

# 6) Time columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week_Start'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)  # week start (Monday)

# ✅ Save the enhanced dataset
df.to_csv("enhanced_supplychain_data.csv", index=False)

# 7) Weekly aggregation
weekly_df = (
    df.groupby(['Product_Code', 'Warehouse', 'Week_Start'], as_index=False)
      .agg(
          Total_Demand = ('Order_Demand', 'sum'),
          Total_Revenue = ('Revenue', 'sum'),
          Total_Profit = ('Profit', 'sum'),
          Avg_Lead_Time = ('Lead_Time_Days', 'mean')
      )
)

# 8) Monthly aggregation
monthly_df = (
    df.groupby(['Product_Code', 'Warehouse', 'Year', 'Month'], as_index=False)
      .agg(
          Total_Demand = ('Order_Demand', 'sum'),
          Total_Revenue = ('Revenue', 'sum'),
          Total_Profit = ('Profit', 'sum'),
          Avg_Lead_Time = ('Lead_Time_Days', 'mean')
      )
)

weekly_df.to_csv("weekly_demand.csv", index=False)
monthly_df.to_csv("monthly_demand.csv", index=False)

monthly_df['Date'] = pd.to_datetime(
    monthly_df[['Year', 'Month']].assign(DAY=1)
)

forecast_list = []

for (product, whse), group in monthly_df.groupby(["Product_Code", "Warehouse"]):

    df_p = group[["Date", "Total_Demand"]].rename(
        columns={"Date": "ds", "Total_Demand": "y"}
    )

    # Initialize model
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=150
    )

    # Fit the model
    model.fit(df_p, freq="M")

    # Forecast next 3 months
    future = model.make_future_dataframe(df_p, periods=3)
    forecast = model.predict(future)

    # Add product & warehouse info
    forecast["Product_Code"] = product
    forecast["Warehouse"] = whse

    forecast_list.append(forecast)

# Combine all forecasts
final_forecast = pd.concat(forecast_list).reset_index(drop=True)

# Save results
final_forecast.to_csv("NeuralProphet_Forecast.csv", index=False)

print("Forecast completed & saved as NeuralProphet_Forecast.csv")



