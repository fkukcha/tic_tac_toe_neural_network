from api.predict_move import predict_move
from api.train_neural_network import model


if __name__ == '__main__':
    # Example usage:
    board = ['X', '-', 'O', '-', 'X', '-', '-', '-', 'O']
    move = predict_move(board, model)
    print(f"Predicted optimal move: {move}")
