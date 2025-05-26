import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Set environment variable for better memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Configure GPU memory usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True first
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Then set memory limit to 60% of available GPU memory
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.6)]
            )
        print("GPU memory limit set to 60% with memory growth enabled")
    except RuntimeError as e:
        print(f"Error setting GPU memory limit: {e}")
        # Fallback to memory growth if setting memory limit fails
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Falling back to GPU memory growth mode")

# Enable mixed precision training
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision training enabled")

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Reduced batch size to prevent memory issues
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 40

# Focal Loss implementation
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + tf.keras.backend.epsilon()) + \
                      (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

# Only preprocessing for validation/test
val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = val_test_datagen.flow_from_directory(
    'test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:", class_weight_dict)

def create_model():
    # Load ResNet50 with pre-trained weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Enhanced architecture with regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    predictions = Dense(4, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, base_model

# Create model
model, base_model = create_model()

# Compile model with focal loss and label smoothing
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2.),
    metrics=['accuracy', 'AUC']
)

# Enhanced callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7
)

# Phase 1: Train only the top layers
print("\nPhase 1: Training top layers")
history1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS_PHASE1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Phase 2: Fine-tuning
print("\nPhase 2: Fine-tuning")
# Unfreeze more layers gradually
for layer in base_model.layers[-120:]:  # Unfreeze more layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss=focal_loss(gamma=2.),
    metrics=['accuracy', 'AUC']
)

history2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS_PHASE2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Combine histories
history = {}
for k in history1.history.keys():
    history[k] = history1.history[k] + history2.history[k]

# Create directory for results
os.makedirs('evaluation_results', exist_ok=True)

# Save the model with better error handling and feedback
print("\nSaving model...")
model_saved = False

# Save in TensorFlow format (more reliable than h5)
try:
    model_path = 'evaluation_results/trained_model_resnet50'
    model.save(model_path, save_format='tf')
    print(f"Model successfully saved in TensorFlow format at: {model_path}")
    model_saved = True
except Exception as e:
    print(f"Error: Could not save model in TensorFlow format: {e}")
    print("Attempting to save in h5 format...")
    
    # Try saving in h5 format as backup
    try:
        model_path = 'evaluation_results/trained_model_resnet50.h5'
        model.save(model_path, save_format='h5')
        print(f"Model successfully saved in h5 format at: {model_path}")
        model_saved = True
    except Exception as e:
        print(f"Error: Could not save model in h5 format: {e}")

if not model_saved:
    print("\nWARNING: Model could not be saved! Visualization script may not work.")
else:
    print("\nModel saved successfully. You can now run the visualization script.")

# Save training history
try:
    history_path = 'evaluation_results/training_history.npy'
    np.save(history_path, history)
    print(f"Training history saved at: {history_path}")
except Exception as e:
    print(f"Error saving training history: {e}")

# Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Get predictions and evaluate
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Calculate and display test metrics
test_loss, test_accuracy, test_auc = model.evaluate(test_generator, verbose=1)
print("\nTest Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Print classification report
class_names = list(test_generator.class_indices.keys())
print("\nDetailed Classification Report:")
print("-"*50)
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('evaluation_results/confusion_matrix.png')
plt.close()

# Plot and save training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.axvline(x=EPOCHS_PHASE1, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.axvline(x=EPOCHS_PHASE1, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history['auc'], label='Training')
plt.plot(history['val_auc'], label='Validation')
plt.axvline(x=EPOCHS_PHASE1, color='r', linestyle='--', label='Start Fine-tuning')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('evaluation_results/training_history.png')
plt.close()

print("\nEvaluation Results:")
print("-"*50)
print("All results have been saved in the 'evaluation_results' directory:")
print("1. confusion_matrix.png - Shows the confusion matrix")
print("2. training_history.png - Shows accuracy, loss, and AUC curves")
print("3. training_history.npy - Contains raw training history data")
print("\nFinal Training Metrics:")
print(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"Training AUC: {history['auc'][-1]:.4f}")
print(f"Validation AUC: {history['val_auc'][-1]:.4f}") 