import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model # Changed from Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout # Added Input
from tensorflow.keras.utils import to_categorical

# --- 1. Data Loading and Preprocessing ---

def one_hot_encode(sequence, max_len=600):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = np.zeros((max_len, len(mapping)))
    padded_seq = sequence[:max_len].upper().ljust(max_len, 'N')
    for i, base in enumerate(padded_seq):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    return encoded

print("Loading labeled dataset...")
df = pd.read_csv('fungi_labeled_data.csv')
df.dropna(subset=['family'], inplace=True)
df = df[df['family'] != 'Unknown_Family'].copy()

print("Filtering out families with only one sample...")
family_counts = df['family'].value_counts()
families_to_keep = family_counts[family_counts >= 2].index.tolist()
df_filtered = df[df['family'].isin(families_to_keep)].copy()

print("Preparing features (X) and labels (y)...")
X = np.array(df_filtered['sequence'].apply(one_hot_encode).tolist())
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_filtered['family'])
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)


# --- 2. Building the Model with the Functional API ---

print("Building the 1D CNN using the Functional API...")

# Define the input shape
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Define the network layers and connect them
x = Conv1D(filters=64, kernel_size=12, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=4)(x)
x = Dropout(0.3)(x)

x = Conv1D(filters=128, kernel_size=8, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=4)(x)
x = Dropout(0.3)(x)

x = Flatten()(x)
x = Dense(256, activation='relu', name="embedding_layer")(x) # We still name our embedding layer
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

# Create the final model by specifying its inputs and outputs
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 3. Training the Model ---

print("\n--- Starting Model Training ---")
EPOCHS = 15
BATCH_SIZE = 32
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test)
)

print("\n--- Evaluating Final Model Performance ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Model Accuracy on Test Data: {accuracy * 100:.2f}%")

model.save('fungi_family_classifier.h5')
print("Trained model saved as 'fungi_family_classifier.h5'")