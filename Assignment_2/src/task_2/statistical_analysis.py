import pandas as pd
import matplotlib.pyplot as plt

# List of files and language names
import pandas as pd

# List of files
files = [
    "data/features/Bengali_mfcc_features.csv",
    "data/features/Gujarati_mfcc_features.csv",
    "data/features/Hindi_mfcc_features.csv",
    "data/features/Kannada_mfcc_features.csv",
    "data/features/Malayalam_mfcc_features.csv",
    "data/features/Marathi_mfcc_features.csv",
    "data/features/Punjabi_mfcc_features.csv",
    "data/features/Tamil_mfcc_features.csv",
    "data/features/Telugu_mfcc_features.csv",
    "data/features/Urdu_mfcc_features.csv",
]

# Extract language names from file paths
languages = [f.split("/")[-1].split("_")[0] for f in files]

# Store results
data = []

for file, lang in zip(files, languages):
    df = pd.read_csv(file)
    # Extract mean and variance columns
    mfcc_mean_cols = [col for col in df.columns if "mfcc_" in col and "_mean" in col]
    mfcc_var_cols = [col for col in df.columns if "mfcc_" in col and "_var" in col]
    
    mean_values = df[mfcc_mean_cols].mean()
    var_values = df[mfcc_var_cols].mean()
    
    # Combine into one row
    row = {"language": lang}
    row.update(mean_values.to_dict())
    row.update(var_values.to_dict())
    
    data.append(row)

# Convert to DataFrame and save
summary_df = pd.DataFrame(data)
summary_df.to_csv("./data/features/language_mfcc_summary.csv", index=False)

print("Saved to language_mfcc_summary.csv")


# # Create plot
# plt.figure(figsize=(16, 9))  # Bigger plot

# for lang, values in mean_values.items():
#     plt.plot(values.index, values.values, label=lang, marker='o')  # Circles at each point

# plt.xlabel("MFCC Feature")
# plt.ylabel("Mean Value")
# plt.title("Mean MFCC Features by Language")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.5)

# # Adjust y-axis scale to zoom in (you can change these limits based on your actual data)
# all_values = pd.DataFrame(mean_values).values.flatten()
# y_min = all_values.min()
# y_max = all_values.max()
# y_margin = (y_max - y_min) * 0.1
# plt.ylim(y_min - y_margin, y_max + y_margin)

# plt.legend()
# plt.tight_layout()
# plt.show()


# # Normalize across features (columns)
# scaler = StandardScaler()
# df_means_scaled = pd.DataFrame(scaler.fit_transform(df_means), columns=df_means.columns, index=df_means.index)

# # Plot
# plt.figure(figsize=(16, 9))
# for lang in df_means_scaled.index:
#     plt.plot(df_means_scaled.columns, df_means_scaled.loc[lang], marker='o', label=lang)

# plt.xlabel("MFCC Feature")
# plt.ylabel("Standardized Mean Value")
# plt.title("Standardized MFCC Means by Language")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

