import torch

@torch.no_grad
class Predictor:
    ''' This class implements a KF-based predictor.
        Based on https://doi.org/10.1109/TAC.2004.834121 '''

    def __init__(self, obs_dim, action_dim, args, device=None):
        self.args = args
        self.device = device

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.transition_dim = obs_dim + action_dim
        self.prediction_horizon = args.diffusion_horizon

        if self.prediction_horizon != 1:
            raise ValueError("Prediction horizon must be 1 for the Kalman Filter predictor.")

        # Initialize the Kalman Filter parameters.
        # TODO: Observation matrix H...
        # TODO: Noise matrices should be set correctly.
        dt = 1.0
        self.F = torch.tensor(
            data=[
                [1, 0, dt, 0, 0, 0, 0, 0, 0], #a0p0
                [0, 1, 0, dt, 0, 0, 0, 0, 0], #a0p1
                [0, 0, 1, 0, 0, 0, 0, 0, 0],  #a0v0
                [0, 0, 0, 1, 0, 0, 0, 0, 0],  #a0v1
                [0, 0, 0, 0, 1, 0, dt, 0, 0], #a1p0
                [0, 0, 0, 0, 0, 1, 0, dt, 0], #a1p1
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  #a1v0
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  #a1v1
                [0, 0, 0, 0, 0, 0, 0, 0, 1],  #id
            ],
            dtype=torch.float32,
            device=device
        )
        self.B = torch.tensor(
            data=[
                [0, 0], #a0p0
                [0, 0], #a0p1
                [0, 0], #a0v0
                [0, 0], #a0v1
                [0, 0], #a1p0
                [0, 0], #a1p1
                [1, 0], #a1v0
                [0, 1], #a1v1
                [0, 0], #id
            ],
            dtype=torch.float32,
            device=device
        )
        self.H = torch.eye(obs_dim, device=device)
        self.Q = torch.eye(obs_dim, device=device) * 0.0 #args.kf_process_noise
        self.R = torch.eye(obs_dim, device=device) * 0.0 #args.kf_measurement_noise

        # TODO: Should probably create a new `initialize` method to set these and create the Kalman Filter.
        # Initial state and covariance.
        self.x0 = torch.zeros((obs_dim,), device=device)
        self.P0 = torch.eye(obs_dim, device=device)

        # Initialize the Kalman Filter.
        self.kf = KalmanFilter(self.F, self.B, self.H, self.Q, self.R, self.x0, self.P0)


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None):
        ''' Get a prediction from the KF.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
                visibility_mask: A tensor of shape (T, D) indicating which parts of the observation are visible.
                prediction_prev: A tensor of shape (1, T, D) for autoregression.
            Returns:
                prediction: A tensor of shape (1, T, D) containing the predicted trajectory.
        '''

        # Apply the visibility mask to the trajectory.
        trajectory = trajectory * visibility_mask

        # Get actions.
        actions = trajectory[-1, :self.action_dim]

        # Get prediction.
        prediction = torch.zeros((1, self.prediction_horizon, self.transition_dim), device=self.device)
        prediction[0, 0, self.action_dim:] = self.kf.predict(actions)

        return prediction


class KalmanFilter:
    ''' This class implements a Kalman Filter for trajectory prediction.
        Implementation is based on: https://www.geeksforgeeks.org/python/kalman-filter-in-python/ '''
    
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0 # Initial state estimate
        self.P = P0 # Initial estimate covariance
    

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x


    def update(self, z):
        S = torch.dot(self.H, torch.dot(self.P, self.H.T)) + self.R
        K = torch.dot(torch.dot(self.P, self.H.T), torch.linalg.inv(S))
        y = z - torch.dot(self.H, self.x)
        self.x = self.x + torch.dot(K, y)
        I = torch.eye(self.P.shape[0])
        self.P = torch.dot(I - torch.dot(K, self.H), self.P)
        return self.x