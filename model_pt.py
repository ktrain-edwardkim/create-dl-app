import torch


# Define Linear Regression model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim=9, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(train_features.shape[1],)),
#         normalizer,
#         tf.keras.layers.Dense(units=1)
#     ])
#     return model