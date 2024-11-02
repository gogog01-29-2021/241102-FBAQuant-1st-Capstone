import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import optuna
from sklearn.metrics import r2_score
import seaborn as sns
import scipy.stats as stats
import os
from datetime import datetime

# Set parameters for stock standard deviation following a normal random walk
r = 0.00
sig = 0.2
T = 30 / 365
M = 300
N = 30
dt = T / N
rdt = r * dt
sigsdt = sig * np.sqrt(dt)

# Generate stock prices
S0 = 100
np.random.seed(20220617)
S = np.empty([M, N + 1])
rv = np.random.normal(r * dt, sigsdt, [M, N])

for i in range(M):
    S[i, 0] = S0
    for j in range(N):
        S[i, j + 1] = S[i, j] * (1 + rv[i, j])

# Visualization of stock prices
for i in range(M):
    plt.plot(S[i, :])
plt.show()

# KDE Plot for random variables
rv_d1 = rv.reshape(M * N, 1)
sns.kdeplot(data=rv_d1, color="red", fill=True)  # Fixed: Replaced `shade` with `fill`
plt.show()

# Shapiro-Wilk Test
test_stat, p_val = stats.shapiro(rv_d1)
print(f"Test statistics: {test_stat}, p-value: {p_val}")

# Option Pricing
K = 95
Savg = np.empty([M, N + 1])
for i in range(M):
    for j in range(N + 1):
        Savg[i, j] = np.mean(S[i, :j + 1])

call_payoff = np.where(Savg - K < 0, 0, Savg - K)
put_payoff = np.where(K - Savg < 0, 0, K - Savg)

# Reinforcement Learning Model and Training Loop
class R2Environment:
    def __init__(self, x_train, y_train, x_val, y_val, episodes=10):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.episodes = episodes
        self.current_episode = 0
        self.state = self.reset()

    def reset(self):
        self.state = np.random.uniform(low=0.01, high=0.05, size=(1,))
        self.current_episode = 0
        return self.state

    def step(self, action):
        new_parameters = self.state + action
        model = self._build_model(new_parameters)
        model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        r2_value = self._evaluate_model(model)
        reward = -r2_value
        self.current_episode += 1
        done = self.current_episode >= self.episodes or r2_value < 0.01
        return new_parameters, reward, done, r2_value  # Include r2_value in the output

    def _build_model(self, parameters):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(1, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters[0]), loss='mse')
        return model

    def _evaluate_model(self, model):
        predictions = model.predict(self.x_val)
        ss_res = np.sum((self.y_val - predictions) ** 2)
        ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
        r2_value = 1 - (ss_res / ss_tot)
        return r2_value

# Data Preparation for RL
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Training Parameters
gamma = 0.7
epsilon = 0.5
epsilon_min = 0.1
epsilon_decay = 0.90
episodes = 10

# Q-Learning Model
def create_q_model(input_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    action = tf.keras.layers.Dense(num_actions, activation="linear")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model

q_model = create_q_model(input_shape=1, num_actions=1)

# Timestamped model path
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"rl_model_{timestamp}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Helper functions for Q-learning
@tf.function
def get_action(state):
    if tf.random.uniform(()) < epsilon:
        return 0  # Exploration
    else:
        return tf.argmax(q_model(tf.expand_dims(state, axis=0))[0]).numpy()  # Exploitation

@tf.function
def update_q_values(state, action, reward, next_state):
    q_values = q_model(tf.expand_dims(state, axis=0))
    next_q_values = q_model(tf.expand_dims(next_state, axis=0))
    target = reward + gamma * tf.reduce_max(next_q_values)
    q_values = q_values.numpy()
    q_values[0][action] = target
    q_model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

# Initialize lists to store rewards and R² scores per episode
episode_rewards = []
episode_r2_scores = []

# Initialize environment for training
env = R2Environment(x_train, y_train, x_val, y_val, episodes)

# Training with interruption handling
try:
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_r2_score = None  # Variable to hold the R² score for the episode

        while not done:
            action = get_action(state)
            next_state, reward, done, r2_score = env.step(action)  # Capture r²_score from step function
            total_reward += reward
            episode_r2_score = r2_score  # Store R² score for the episode

            update_q_values(state, action, reward, next_state)
            state = next_state

            if done:
                print(f"Episode {episode}, Total Reward: {total_reward}, R² Score: {episode_r2_score}")
                episode_rewards.append(total_reward)
                episode_r2_scores.append(episode_r2_score)
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    save_model(q_model)

# Final model save after training completion
save_model(q_model)

# Generate Dynamic Report with Actual Values
average_reward = np.mean(episode_rewards)
final_reward = episode_rewards[-1]
average_r2_score = np.mean(episode_r2_scores)
final_r2_score = episode_r2_scores[-1]

print("\n### Training Report ###")
print("Input Data:")
print(f"  - Generated Stock Prices for {M} paths, {N} time points each")
print("  - Reinforcement Learning Environment with regression-based generated data (100 samples)")
print("Actions Taken:")
print("  - Exploration (Random actions) and Exploitation (Best learned actions) during training")
print("Model Performance:")
print(f"  - Total Episodes: {episodes}")
print(f"  - Average Reward: {average_reward:.4f}")
print(f"  - Final Episode Reward: {final_reward:.4f}")
print(f"  - Average R² Score: {average_r2_score:.4f}")
print(f"  - Final R² Score: {final_r2_score:.4f}")
