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

# Visualize stock prices
for i in range(M):
    plt.plot(S[i, :])
plt.show()

# KDE Plot for random variables
rv_d1 = rv.reshape(M * N, 1)
sns.kdeplot(data=rv_d1, color="red", shade=True)
plt.show()

# Shapiro-Wilk Test
test_stat, p_val = stats.shapiro(rv_d1)
print(f"Test statistics: {test_stat}, p-value: {p_val}")

# Asian option price calculation
K = 95
Savg = np.empty([M, N + 1])
for i in range(M):
    for j in range(N + 1):
        Savg[i, j] = np.mean(S[i, :j + 1])

call_payoff = np.where(Savg - K < 0, 0, Savg - K)
put_payoff = np.where(K - Savg < 0, 0, K - Savg)

# Histogram of average stock price
plt.hist(Savg)
plt.show()

# Call option pricing function
def bocall(Savg, K, M, N):
    payoff = Savg[:, N] - K
    payoff = 0.5 * (abs(payoff) + payoff)
    res = np.mean(payoff)
    return res

bo_call = bocall(Savg, K, M, N)
print("Call option price:", bo_call)

# Put option pricing based on put-call parity
def boput(C, K, S):
    put = C + K - S
    return put

bo_put = boput(bo_call, K, S0)
print("Put option price:", bo_put)

# Custom Reinforcement Learning Environment
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
        return new_parameters, reward, done

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

# Dataset for RL model
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Reinforcement Learning Hyperparameters
gamma = 0.7
epsilon = 0.5
epsilon_min = 0.1
epsilon_decay = 0.90
episodes = 10

# Initialize environment
env = R2Environment(x_train, y_train, x_val, y_val, episodes)

# Q-Learning Model
def create_q_model(input_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    action = tf.keras.layers.Dense(num_actions, activation="linear")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model

q_model = create_q_model(input_shape=1, num_actions=1)

# Training loop
@tf.function
def predict_action(state):
    return q_model(tf.expand_dims(state, axis=0))

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() < epsilon:
            action = 0
        else:
            action = int(np.argmax(predict_action(state)[0]))

        next_state, reward, done = env.step(action)
        total_reward += reward

        q_values = q_model(tf.expand_dims(state, axis=0))
        next_q_values = q_model(tf.expand_dims(next_state, axis=0))
        target = reward + gamma * tf.reduce_max(next_q_values)
        q_values = q_values.numpy()
        q_values[0][action] = target
        q_model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

        state = next_state
        if done:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Rewards and R² Scores Visualization
rewards, r2_scores = [], []
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() < epsilon:
            action = 0
        else:
            action = int(np.argmax(predict_action(state)[0]))

        next_state, reward, done = env.step(action)
        total_reward += reward

        q_values = q_model(tf.expand_dims(state, axis=0))
        next_q_values = q_model(tf.expand_dims(next_state, axis=0))
        target = reward + gamma * tf.reduce_max(next_q_values)
        q_values = q_values.numpy()
        q_values[0][action] = target
        q_model.fit(tf.expand_dims(state, axis=0), q_values, verbose=0)

        state = next_state

        if done:
            y_pred = q_model.predict(x_val).flatten()
            r2 = r2_score(y_val, y_pred)
            rewards.append(total_reward)
            r2_scores.append(r2)
            print(f"Episode {episode}, Total Reward: {total_reward}, R² Score: {r2}")
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(r2_scores)
plt.title('R² Score per Episode')
plt.xlabel('Episode')
plt.ylabel('R² Score')

plt.tight_layout()
plt.show()

# Optuna Optimization
def build_and_train_model(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    epochs = trial.suggest_int('epochs', 1, 20)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    model.fit(x_train, y_train, epochs=epochs, verbose=0)
    predictions = model.predict(x_val)
    r2_value = r2_score(y_val, predictions)
    return r2_value

study = optuna.create_study(direction="maximize")
study.optimize(build_and_train_model, n_trials=50)

print("Best hyperparameters:")
print("  R² Score:", study.best_trial.value)
print("  Hyperparameters:", study.best_trial.params)

try:
    import optuna.visualization as vis

    # Plot optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.show()

    # Plot parameter importance
    fig2 = vis.plot_param_importances(study)
    fig2.show()
except ImportError:
    print("Optuna visualization libraries are required for plotting. Install with `pip install optuna[visualization]`.")

# Save the model
model_save_path = 'rl_model.h5'
q_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Load and make predictions
q_model = tf.keras.models.load_model(model_save_path)
new_data = np.array([[0.03]])  # Replace with actual data
predicted_action = q_model.predict(new_data)
print("Predicted action:", predicted_action)
