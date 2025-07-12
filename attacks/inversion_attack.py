import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def run_inversion(model_path, input_dim):
    model = tf.keras.models.load_model(model_path)

    target_output = tf.constant([[0., 1.]], dtype=tf.float32)
    reconstructed_input = tf.Variable(tf.random.normal((1, input_dim)))

    optimizer = tf.keras.optimizers.Adam(0.1)

    for i in range(1000):
        with tf.GradientTape() as tape:
            pred = model(reconstructed_input)
            loss = tf.reduce_mean(tf.square(pred - target_output))
        grads = tape.gradient(loss, [reconstructed_input])
        optimizer.apply_gradients(zip(grads, [reconstructed_input]))
        if i % 100 == 0:
            print(f"[Iteration {i}] Loss: {loss.numpy():.4f}")

    np.save("evaluation/reconstructed_input.npy", reconstructed_input.numpy())
    print("[Inversion] Reconstruction complete. Saved to evaluation/.")

if __name__ == "__main__":
    run_inversion("evaluation/fl_model.h5", input_dim=19)  # Replace 19 with actual input feature size
