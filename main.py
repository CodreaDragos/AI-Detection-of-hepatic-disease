import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set up enhanced data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
validation_generator = test_datagen.flow_from_directory(
    'valid',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Save the class indices mapping
class_indices = train_generator.class_indices
print("Class mapping:", class_indices)

# Define liver condition classes mapping 
# This mapping will be used to convert from your current labels to the desired 4 categories
# Each key is a current class index, and value is the new liver condition category index
LIVER_CLASSES = [
    'Homogeneous Liver',   # Normal liver
    'Liver Tumor',         # Secondary determinations
    'Liver Hemangioma',    
    'Liver Cyst'
]

'''
# Option 1: Create a new model with 4 outputs for liver conditions
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes for liver conditions
])
'''

# Option 2: Load the existing model with 6 outputs and modify it
# This is better if you already have a well-performing model
try:
    # Check if the high-accuracy model exists
    if os.path.exists('trained_model.h5'):
        print("Loading pre-trained model...")
        base_model = tf.keras.models.load_model('trained_model.h5')
        
        # Remove the last layer
        x = base_model.layers[-2].output  # Get the output of the second-to-last layer
        
        # Add a new classification layer for 4 classes
        new_outputs = Dense(4, activation='softmax', name='liver_condition_classifier')(x)
        
        # Create a new model
        model = Model(inputs=base_model.input, outputs=new_outputs)
        
        # Freeze the early layers to preserve the learned features
        for layer in model.layers[:-4]:  # Freeze all except the last few layers
            layer.trainable = False
            
        print("Modified model to classify into 4 liver conditions")
    else:
        # Create a new model if no existing model
        print("No pre-trained model found. Creating new model...")
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # 4 classes for liver conditions
        ])
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Creating new model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes for liver conditions
    ])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Print model summary
model.summary()

print("\nIMPORTANT: Before training, you need to:")
print("1. Reorganize your data into 4 folders matching the liver conditions")
print("2. Each folder should contain the corresponding liver condition images")
print("3. Update the 'train', 'valid', and 'test' directories with the new structure")
print("\nAfter organizing the data, run this script again to train the model.")

# Ask user if they want to proceed with training
proceed = input("\nHave you organized your data and want to proceed with training? (y/n): ")

if proceed.lower() != 'y':
    print("Exiting without training. Please organize your data first.")
    exit()

# Train the model
print("Starting model training...")

# Define callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

# Define learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch % 10 == 0):
        return lr * 0.8
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,  # Increase epochs with early stopping
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model
model.save('liver_model.h5')
print("Model saved as 'liver_model.h5'")

# Save the class indices mapping for visualization
np.save('class_mapping.npy', {'indices': class_indices, 'liver_classes': LIVER_CLASSES})

# Save the training history as a numpy file for later visualization
np.savez('training_history.npz', history=history.history)
print("Training history saved to 'training_history.npz'")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')

# Print final metrics
print("\nFinal Training Metrics:")
print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Training AUC: {history.history['auc'][-1]:.4f}")

print("\nFinal Validation Metrics:")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

# Evaluate on the test set
print("\nEvaluating on test set...")
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test AUC: {test_auc:.4f}") 