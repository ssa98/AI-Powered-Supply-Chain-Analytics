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

# ✅ Derive extra business features (optional but great for dashboard)
df['Revenue'] = df['Order_Demand'] * df['Selling_Price']
df['Profit'] = df['Revenue'] - (df['Order_Demand'] * df['Cost_Price'])

# ✅ Save the enhanced dataset
df.to_csv("enhanced_supplychain_data.csv", index=False)

df.head(10)
