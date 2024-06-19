# Tic Tac Toe Neural Network
This project implements a Tic Tac Toe AI using a neural network trained with TensorFlow/Keras. The AI learns to predict optimal moves based on the current game state, providing a challenging opponent or a learning experiment in AI development.
## Table of Contents
- Requirements
- Game Environment
- Dataset Generation
- Neural Network Model
- Training
- Usage
- Testing
- Contributing
## Requirements
- Python 3.x
- TensorFlow 2.x(or higher)
- NumPy
- pytest (for testing)

Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
## Game Environment
The Tic Tac Toe game environment provides functions to manage the game state, check for wins/draws, and make moves.
### Functions
- `check_win(board, player)`: Checks if the given player has won on the current board configuration.
- `check_draw(board)`: Checks if the board is full, resulting in a draw.
- `make_move(board, position, player)`: Updates the board with the player's move at the specified position.
## Dataset Generation
To train the neural network, we generate a dataset of game states and their corresponding optimal moves.
### Function
- `generate_dataset(num_games)`: Generates num_games of Tic Tac Toe games, recording game states and optimal moves for training.
## Neural Network Model
We define a neural network model using TensorFlow/Keras to predict the optimal move given the current game state.
### Model Architecture
The neural network architecture is as follows:
```python
model = Sequential([
    Flatten(input_shape=(9,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(9, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
This model:

- Takes a flattened input of size 9 (representing the Tic Tac Toe board).
- Has two hidden layers with ReLU activation functions.
- Outputs a softmax layer of size 9, predicting the probability distribution over possible moves.
## Training
To train the neural network, follow these steps:
1. Generate Training Data: Use `generate_training_data(num_games)` to create a dataset of game states (`X_train`) and optimal moves (`y_train`).
2. Build Model: Create an instance of the neural network model using the provided architecture.
3. Train Model: Train the model using `model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1).
## Usage
After training, you can use the model to predict moves and evaluate its performance.
### Predicting Moves
Use the `predict_move(board, model)` function to predict the optimal move for a given board configuration.
#### Example
```python
board = ['X', '-', 'O', '-', 'X', '-', '-', '-', 'O']
move = predict_move(board, model)
print(f"Predicted optimal move: {move}")
```
#### Running `main.py`
To run the main script main.py, which demonstrates the usage of the trained model to predict moves, follow these steps:
1. Ensure that all dependencies are installed by running:
```bash
pip install -r requirements.txt
```
2. Run the script `main.py`

This will execute main.py and demonstrate the prediction of the optimal move for a predefined board configuration using the trained model.
### Play against the AI
To play against the AI, run the script `play_game_against_ai.py`
## Testing
To ensure the functionality from data generation to model training works correctly, run tests.

To run the tests, navigate to your project root and run:
```bash
pytest
```
## Contributing
Contributions are welcome! Feel free to fork the repository, create pull requests, or open issues for any improvements or bug fixes.
