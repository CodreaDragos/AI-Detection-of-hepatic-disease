import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print GPU details if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("\nGPU Details:")
        print("Name:", gpu.name)
        print("Type:", gpu.device_type)
        
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    # Test GPU with a simple operation
    print("\nTesting GPU with a simple operation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("Matrix multiplication completed successfully on GPU!")
else:
    print("\nNo GPU devices found. TensorFlow is using CPU.") 