import itertools
import pandas as pd
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
print("libraries loaded")

file_paththreshold = '/home/felipe.castanogonzal/ENEL645/finalproject/principal_data.csv'
file_paththreshold1 = '/home/felipe.castanogonzal/ENEL645/finalproject/unique_data.csv'
file_paththreshold2 = '/home/felipe.castanogonzal/ENEL645/finalproject/duplicated_data.csv'
base_data = pd.read_csv(file_paththreshold)
original_data = pd.read_csv(file_paththreshold1)
deduplicated_data = pd.read_csv(file_paththreshold2)
print("\nStart of the threshold")

# Sample 3000 records from the base data
sampled_data = base_data.sample(n=3000, random_state=42)

# Filter the sampled data to include only rows where IsDuplicated == 0
sampled_unique_data = sampled_data[sampled_data['IsDuplicated'] == 0]

# Filter the sampled data to include only rows where IsDuplicated == 0
sampled_duplicate_data = sampled_data[sampled_data['IsDuplicated'] == 1]

# Generate all possible comparisons and create a table without indices
comparisons = itertools.combinations(sampled_data['BloomFilter32Bit'], 2)

# Create a list to store comparison results
comparison_table = []

# Populate the table with comparisons
for bloom1, bloom2 in comparisons:
    comparison_table.append({
        "BloomFilter1": bloom1,
        "BloomFilter2": bloom2
    })
print("\nStart with 3000")
# Convert to a DataFrame
comparison_df = pd.DataFrame(comparison_table)

# Function to calculate the Dice coefficient between two integers
def dice_coefficient(bf1, bf2):
    # Convert integers to binary strings
    bin1 = bin(bf1)[2:]
    bin2 = bin(bf2)[2:]
    # Pad binary strings to ensure they are the same length
    max_length = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_length)
    bin2 = bin2.zfill(max_length)
    # Count matching bits and total bits
    intersection = sum(1 for b1, b2 in zip(bin1, bin2) if b1 == b2 == '1')
    total_bits = len(bin1) + len(bin2)
    # Dice coefficient calculation
    return 2 * intersection / total_bits

# Generate all possible comparisons and calculate the Dice coefficient
comparisons = itertools.combinations(sampled_data['BloomFilter32Bit'], 2)

# Create a list to store comparison results with Dice coefficient
comparison_dice_table = []

# Populate the table with comparisons and Dice coefficients
for bloom1, bloom2 in comparisons:
    dice_value = dice_coefficient(bloom1, bloom2)
    comparison_dice_table.append({
        "BloomFilter1": bloom1,
        "BloomFilter2": bloom2,
        "DiceCoefficient": dice_value
    })

# Convert to a DataFrame
comparison_dice_df = pd.DataFrame(comparison_dice_table)
print("\nstart the 04,03,02,01 threshold")
# Add a new column with the threshold condition
comparison_dice_df['Threshold_0.4'] = comparison_dice_df['DiceCoefficient'].apply(lambda x: 0 if x > 0.4 else 1)
# Add a new column with the threshold condition
comparison_dice_df['Threshold_0.3'] = comparison_dice_df['DiceCoefficient'].apply(lambda x: 0 if x > 0.3 else 1)
# Add a new column with the threshold condition
comparison_dice_df['Threshold_0.2'] = comparison_dice_df['DiceCoefficient'].apply(lambda x: 0 if x > 0.2 else 1)
# Add a new column with the threshold condition
comparison_dice_df['Threshold_0.1'] = comparison_dice_df['DiceCoefficient'].apply(lambda x: 0 if x > 0.1 else 1)

# Extract the BloomFilter32Bit column from the unique sampled data
unique_bloom_filters = sampled_unique_data['BloomFilter32Bit'].drop_duplicates().values

# Generate all possible pairwise comparisons for the unique Bloom filters
unique_comparisons = itertools.combinations(unique_bloom_filters, 2)

# Create a list to store unique comparison results
unique_comparison_table = []

# Populate the table with comparisons
for bloom1, bloom2 in unique_comparisons:
    unique_comparison_table.append({
        "BloomFilter1": bloom1,
        "BloomFilter2": bloom2
    })

# Convert to a DataFrame
unique_comparison_df = pd.DataFrame(unique_comparison_table)

# Define the Dice coefficient function
def dice_coefficient(bf1, bf2):
    # Convert integers to binary strings
    bin1 = bin(bf1)[2:]
    bin2 = bin(bf2)[2:]
    # Pad binary strings to ensure they are the same length
    max_length = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_length)
    bin2 = bin2.zfill(max_length)
    # Count matching bits and total bits
    intersection = sum(1 for b1, b2 in zip(bin1, bin2) if b1 == b2 == '1')
    total_bits = len(bin1) + len(bin2)
    # Dice coefficient calculation
    return 2 * intersection / total_bits

# Add a DiceCoefficient column to the unique_comparison_df
unique_comparison_df['DiceCoefficient'] = unique_comparison_df.apply(
    lambda row: dice_coefficient(row['BloomFilter1'], row['BloomFilter2']),
    axis=1
)

# Add a new column with all zeros
unique_comparison_df['uniquecolumn'] = 0

# Compare the tables and add a new column to comparison_dice_df
comparison_dice_df['uniquecolumn'] = comparison_dice_df.apply(
    lambda row: unique_comparison_df[
        (unique_comparison_df['BloomFilter1'] == row['BloomFilter1']) &
        (unique_comparison_df['BloomFilter2'] == row['BloomFilter2']) &
        (unique_comparison_df['DiceCoefficient'] == row['DiceCoefficient'])
    ]['uniquecolumn'].values[0]
    if not unique_comparison_df[
        (unique_comparison_df['BloomFilter1'] == row['BloomFilter1']) &
        (unique_comparison_df['BloomFilter2'] == row['BloomFilter2']) &
        (unique_comparison_df['DiceCoefficient'] == row['DiceCoefficient'])
    ].empty else None,
    axis=1
)

# Count the number of zeros in the 'uniquecolumn' of unique_comparison_df
zero_count = (comparison_dice_df['uniquecolumn'] == 0).sum()

# Use all rows, including duplicates, for comparisons
duplicate_bloom_filters = sampled_duplicate_data['BloomFilter32Bit'].values

# Generate all possible pairwise comparisons for all rows
duplicate_comparisons = itertools.combinations(duplicate_bloom_filters, 2)

# Create a list to store all comparison results
duplicate_comparison_table = [
    {"BloomFilter1": bloom1, "BloomFilter2": bloom2} for bloom1, bloom2 in duplicate_comparisons
]

# Convert to a DataFrame
duplicate_comparison_df = pd.DataFrame(duplicate_comparison_table)

# Add a DiceCoefficient column to the duplicate_comparison_df
duplicate_comparison_df['DiceCoefficient'] = duplicate_comparison_df.apply(
    lambda row: dice_coefficient(row['BloomFilter1'], row['BloomFilter2']),
    axis=1
)

# Add a new column with all zeros
duplicate_comparison_df['uniquecolumn'] = 1

# Update the `uniquecolumn` in comparison_dice_df based on matches with unique_comparison_df
comparison_dice_df['uniquecolumn'] = comparison_dice_df.apply(
    lambda row: 1 if not duplicate_comparison_df[
        (duplicate_comparison_df['BloomFilter1'] == row['BloomFilter1']) &
        (duplicate_comparison_df['BloomFilter2'] == row['BloomFilter2']) &
        (duplicate_comparison_df['DiceCoefficient'] == row['DiceCoefficient'])
    ].empty else row['uniquecolumn'],
    axis=1
)

# Count the number of zeros in the 'uniquecolumn' of unique_comparison_df
one_count = (comparison_dice_df['uniquecolumn'] == 1).sum()

# Calculate true positives and true negatives for threshold_0.3
true_positives = comparison_dice_df[
    (comparison_dice_df['Threshold_0.3'] == 1) & (comparison_dice_df['uniquecolumn'] == 1)
].shape[0]

true_negatives = comparison_dice_df[
    (comparison_dice_df['Threshold_0.3'] == 0) & (comparison_dice_df['uniquecolumn'] == 0)
].shape[0]

# Calculate precision, recall, and accuracy
total_positives = comparison_dice_df[comparison_dice_df['uniquecolumn'] == 1].shape[0]
total_negatives = comparison_dice_df[comparison_dice_df['uniquecolumn'] == 0].shape[0]

precision = true_positives / total_positives if total_positives > 0 else 0
recall = true_positives / (true_positives + total_positives - true_positives) if total_positives > 0 else 0
accuracy = (true_positives + true_negatives) / comparison_dice_df.shape[0] if comparison_dice_df.shape[0] > 0 else 0

# Display the results
{
    "True Positives": true_positives,
    "True Negatives": true_negatives,
    "Precision": precision,
    "Recall": recall,
    "Accuracy": accuracy,
}
print("\nprinting the threshold")
# Reinitialize thresholds and calculate false positives (FP) and false negatives (FN)
thresholds = ['Threshold_0.1', 'Threshold_0.2', 'Threshold_0.3', 'Threshold_0.4']

# Initialize a list to store the updated metrics
updated_results1 = []

for threshold in thresholds:
    true_positives = comparison_dice_df[
        (comparison_dice_df[threshold] == 1) & (comparison_dice_df['uniquecolumn'] == 1)
    ].shape[0]

    true_negatives = comparison_dice_df[
        (comparison_dice_df[threshold] == 0) & (comparison_dice_df['uniquecolumn'] == 0)
    ].shape[0]

    total_positives = comparison_dice_df[comparison_dice_df['uniquecolumn'] == 1].shape[0]
    total_negatives = comparison_dice_df[comparison_dice_df['uniquecolumn'] == 0].shape[0]

    false_positives = total_negatives - true_negatives
    false_negatives = total_positives - true_positives

    precision = true_positives / (true_positives + false_positives) 
    recall = true_positives / (true_positives + false_negatives) 
    accuracy = (true_positives + true_negatives) / ( true_positives + true_negatives + false_positives + false_negatives)

    updated_results1.append({
        "Threshold": threshold,
        "True Positives": true_positives,
        "True Negatives": true_negatives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "Total Positives": total_positives,
        "Total Negatives": total_negatives,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
    })

# Convert updated results to a DataFrame
updated_metrics_df1 = pd.DataFrame(updated_results1)
updated_metrics_df1.to_csv('output_updated_metrics_3000threshold.csv', index=False)
print(updated_metrics_df1)
