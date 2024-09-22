import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

players_data = pd.read_csv("players_20.csv")
print(f"First 6 rows: {players_data.head(6)}")
print(f"players_data.shape: {players_data.shape}")
print(f"list from players_data.columns: {list(players_data.columns)}")
useless_columns = ['dob',
                   'sofifa_id',
                   'player_url',
                   'long_name',
                   'body_type',
                   'real_face',
                   'loaned_from',
                   'nation_position',
                   'nation_jersey_number'
                   ]
df_dropped = players_data.drop(columns=useless_columns, axis=1)
print(f"filtered players_data: {df_dropped}")
print(f"index weight_kg\n{players_data['weight_kg'].tail()}")
print(f"{players_data[['short_name', 'weight_kg']].head()}")
df_players_with_bmi = df_dropped.assign(BMI=lambda x: x.weight_kg / ((x.height_cm * 0.01) ** 2))
print(f"{df_players_with_bmi[['short_name','BMI']].head(10)}")

plt.figure(figsize=(8, 5))
plt.hist(df_players_with_bmi['BMI'], bins=5, edgecolor='black', color='skyblue')
plt.title('Гістограма розподілу індексу маси тіла (BMI)')
plt.xlabel('Індекс маси тіла (BMI)')
plt.ylabel('Кількість гравців')
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(players_data['age'], bins=5, edgecolor='black', color='lightgreen')
plt.title('Гістограма розподілу гравців за віком')
plt.xlabel('Вік гравців')
plt.ylabel('Кількість гравців')
plt.show()

players_above_30 = players_data[players_data['age'] > 30]
mean_weight = players_above_30['weight_kg'].mean()
median_weight = players_above_30['weight_kg'].median()
print(f"median: {median_weight}")
print(f"mean: {mean_weight}")
players_above_30.tail(10)
nationality_counts = players_data['nationality'].value_counts()
top_7_nationalities = nationality_counts.sort_values(ascending=False).head(7)
print(f"top_7_nationalities: {top_7_nationalities}")


