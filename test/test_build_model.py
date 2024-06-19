from api.build_model import build_model


def test_build_model():
    """
    Test if the build_model function returns a valid Keras model.
    The model should have an input shape of (None, 9) and an output shape of (None, 9).
    The model should have a valid optimizer and loss function.
    """
    model = build_model()
    assert model is not None
    assert model.input_shape == (None, 9)
    assert model.output_shape == (None, 9)
    assert model.optimizer is not None
    assert model.loss == 'categorical_crossentropy'
