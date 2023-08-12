import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Define the dataset size
dataset_size = 1000

# Generate the spam and ham emails
spam_emails = ["Buy our products now!", "Get rich quick!", "Claim your prize!", "Limited time offer!"]
ham_emails = ["Hello, how are you?", "Meeting tomorrow at 2 PM", "Please find attached the report"]

# Create the dataset
dataset = []
for _ in range(dataset_size):
    if np.random.rand() < 0.2:  # 20% spam emails
        email = np.random.choice(spam_emails)
        label = 'spam'
    else:  # 80% ham emails
        email = np.random.choice(ham_emails)
        label = 'ham'
    dataset.append((email, label))

# Create a DataFrame from the dataset
df = pd.DataFrame(dataset, columns=['text', 'label'])

# Save the dataset to a CSV file
df.to_csv("spam_dataset.csv", index=False)
