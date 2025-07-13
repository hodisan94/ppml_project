import tensorflow as tf
import numpy as np

def run_inversion(model_path, input_dim, target_label=1):
    model = tf.keras.models.load_model(model_path)

    target_output = tf.constant([[0., 1.]] if target_label == 1 else [[1., 0.]], dtype=tf.float32)
    reconstructed_input = tf.Variable(tf.random.normal((1, input_dim)))

    optimizer = tf.keras.optimizers.Adam(0.1)

    for i in range(500):
        with tf.GradientTape() as tape:
            pred = model(reconstructed_input)
            loss = tf.reduce_mean(tf.square(pred - target_output))
        grads = tape.gradient(loss, [reconstructed_input])
        optimizer.apply_gradients(zip(grads, [reconstructed_input]))

    print(f"[INVERSION] Done - saved to results/reconstructed_input.npy")
    np.save("results/reconstructed_input.npy", reconstructed_input.numpy())

if __name__ == "__main__":
    run_inversion("results/fl_dp_model.h5", input_dim=19)
