"""Lab 08: MNIST DNN and tabular Q-learning maze. Run on Python with TensorFlow."""
from pathlib import Path
import json
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs' / 'lab_08'
OUT.mkdir(parents=True, exist_ok=True)

def mnist_dnn():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(-1, 784).astype('float32') / 255.0
    test_x = test_x.reshape(-1, 784).astype('float32') / 255.0
    model = tf.keras.Sequential([tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(train_x, train_y, epochs=5, batch_size=128, validation_split=.1, verbose=2)
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    model.save(OUT / 'mnist_dnn.keras')
    (OUT / 'mnist_metrics.json').write_text(json.dumps(dict(test_loss=float(loss), test_accuracy=float(accuracy)), indent=2))
    plt.figure(); plt.plot(hist.history['accuracy'], label='train'); plt.plot(hist.history['val_accuracy'], label='validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'mnist_learning_curve.png'); plt.close()
    print('MNIST test accuracy:', round(accuracy, 4))

def q_learning_maze():
    maze = np.array([[0,0,0,1,0],[1,1,0,1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0]])
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    start, goal = (0,0), (4,4)
    rng = np.random.default_rng(42)
    q = np.zeros((5,5,4)); epsilon = 1.0; rewards = []
    def adjacent(state, action):
        nxt = (state[0] + moves[action][0], state[1] + moves[action][1])
        if 0 <= nxt[0] < 5 and 0 <= nxt[1] < 5 and maze[nxt] == 0:
            return nxt
        return state
    for episode in range(5000):
        state = start; reward_sum = 0
        for _ in range(100):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q[state]))
            nxt = adjacent(state, action)
            reward = 100 if nxt == goal else (-5 if nxt == state else -1)
            done = nxt == goal
            q[state + (action,)] += .2 * ((reward if done else reward + .95 * np.max(q[nxt])) - q[state + (action,)])
            state = nxt; reward_sum += reward
            if done: break
        rewards.append(reward_sum); epsilon = max(.02, epsilon * .998)
    path = [start]; state = start
    for _ in range(25):
        options = [adjacent(state, a) for a in range(4)]
        valid = [(q[state + (a,)], cell) for a, cell in enumerate(options) if cell != state and cell not in path]
        if not valid: raise RuntimeError('Learned policy did not reach goal; adjust training seed/episodes')
        nxt = max(valid, key=lambda pair: pair[0])[1]
        path.append(nxt); state = nxt
        if state == goal: break
    if state != goal: raise RuntimeError('Goal not reached')
    queue = deque([(start,0)]); seen = {start}; shortest = None
    while queue:
        cell, steps = queue.popleft()
        if cell == goal: shortest = steps; break
        for a in range(4):
            nxt = adjacent(cell, a)
            if nxt not in seen:
                seen.add(nxt); queue.append((nxt, steps+1))
    np.save(OUT / 'maze_q_table.npy', q)
    result = dict(path=path, steps=len(path)-1, bfs_shortest_steps=shortest)
    (OUT / 'maze_metrics.json').write_text(json.dumps(result, indent=2))
    plt.figure(); plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.xlabel('Episode'); plt.ylabel('Average reward over 100 episodes'); plt.tight_layout()
    plt.savefig(OUT / 'maze_learning_curve.png'); plt.close()
    plt.figure(); plt.imshow(maze, cmap='Greys'); plt.plot([c for r,c in path],[r for r,c in path],'-o')
    plt.title(f'Maze steps: {len(path)-1}, BFS minimum: {shortest}'); plt.tight_layout()
    plt.savefig(OUT / 'maze_solution.png'); plt.close()
    print('Maze:', result)

if __name__ == '__main__':
    mnist_dnn()
    q_learning_maze()
