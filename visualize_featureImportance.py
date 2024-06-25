import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data into a DataFrame
FILE_PATH = r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\Models\model__20240610_163644_featureimportance.csv'
data = pd.read_csv(FILE_PATH)

# Identify feature columns (all columns except 'epoch')
feature_columns = [col for col in data.columns if col != 'epoch']

# Melting the DataFrame to long format
melted_data = data.melt(id_vars=['epoch'], 
                        value_vars=feature_columns,
                        var_name='feature', 
                        value_name='importance')

# Exclude features that contain 'before', 'after', or 'feature' in their names
filtered_melted_data = melted_data[~melted_data['feature'].str.contains('before|after|feature')]

# Identify the top 10 features with the highest importance in the final epoch
final_epoch = data['epoch'].max()
final_importances = filtered_melted_data[filtered_melted_data['epoch'] == final_epoch]
top_10_features = final_importances.nlargest(10, 'importance')['feature']

# Filter the melted_data to include only the top 10 features
top_10_data = filtered_melted_data[filtered_melted_data['feature'].isin(top_10_features)]

# Plotting the feature importance over time (by epoch)
plt.figure(figsize=(12, 6))
lines = {}
for feature in top_10_features:
    subset = top_10_data[top_10_data['feature'] == feature]
    line, = plt.plot(subset['epoch'], subset['importance'], label=feature)
    lines[feature] = line
    
plt.xlabel('Epoch')
plt.ylabel('Average Mean Square Error (Log Scale)')
plt.yscale('log')
plt.title('Top 10 Feature Importance Over Time (by Epoch)')
# Sort legend by importance in the final epoch
sorted_handles = sorted(lines.items(), key=lambda item: final_importances[final_importances['feature'] == item[0]]['importance'].values[0], reverse=True)
plt.legend([item[1] for item in sorted_handles], [item[0] for item in sorted_handles], title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Bar plot for all features in the final epoch excluding 'before', 'after', and 'feature' with a logarithmic x-axis
plt.figure(figsize=(12, 20))
final_importances_all = final_importances.set_index('feature').sort_values(by='importance', ascending=False)
final_importances_all_filtered = final_importances_all[~final_importances_all.index.str.contains('before|after|feature')]

# Create the horizontal bar plot with seaborn to avoid overlapping
sns.barplot(x='importance', y=final_importances_all_filtered.index, data=final_importances_all_filtered.reset_index(), color='#1f77b4')  # Nice blue color
plt.xscale('log')
plt.xlabel('Average Mean Square Error (Log Scale)')
plt.ylabel('Features')
plt.title('Feature Importance of final model (excluding time before/after)')
plt.tight_layout()
plt.show()
