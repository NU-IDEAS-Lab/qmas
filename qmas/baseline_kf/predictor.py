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
        self.prediction_horizon = args.prediction_history_window

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
        self.Q = torch.eye(obs_dim, device=device) * 0.1 #args.kf_process_noise
        self.R = torch.eye(obs_dim, device=device) * 0.0 #args.kf_measurement_noise

        self.kf = None

    def initialize(self, x0, P0):
        ''' Initialize the Kalman Filter. '''

        self.kf = KalmanFilterIntermittenObservations(self.F, self.B, self.H, self.Q, self.R, x0, P0)


    def get_prediction(self, trajectory, visibility_mask, prediction_prev=None):
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

        # Check whether the filter is initialized.
        if self.kf is None:
            self.initialize(
                x0=trajectory[-1, self.action_dim:],
                P0=torch.eye(self.obs_dim, device=self.device)
            )

        # Get actions.
        actions = trajectory[-1, :self.action_dim]

        # Get prediction.
        prediction = torch.zeros((1, self.prediction_horizon, self.transition_dim), device=self.device)
        prediction[0, 0, self.action_dim:] = self.kf.predict(actions)

        # Set up a diagonal visibility mask for the observation.
        visibility_mask_diag = torch.diag(visibility_mask[-1, self.action_dim:])

        # Update the Kalman Filter with the observation.
        self.kf.update(trajectory[-1, self.action_dim:], visibility_mask_diag)

        return prediction


class KalmanFilter:
    ''' This class implements a Kalman Filter for trajectory prediction.
        Implementation is based on: https://www.geeksforgeeks.org/python/kalman-filter-in-python/ '''
    
    @torch.no_grad
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
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = torch.eye(self.P.shape[0], device=self.P.device)
        self.P = (I - K @ self.H) @ self.P
        return self.x


class KalmanFilterIntermittenObservations(KalmanFilter):
    ''' This class implements a Kalman Filter for trajectory prediction with intermittent observations.
        It inherits from the KalmanFilter class.
        It is based on work by Sinopoli et al. (2004) - https://doi.org/10.1109/TAC.2004.834121 '''

    def predict(self, u):
        return super().predict(u)

    def update(self, z, gamma=None):
        """
        Measurement update with gamma in [0, 1].
        gamma=0 -> no update (no measurement available at t)
        gamma=1 -> standard KF update
        """

        if gamma is None:
            gamma = torch.eye(z.shape[0], dtype=z.dtype, device=z.device)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ (gamma @ y)
        self.P = self.P - K @ gamma @ self.H @ self.P
        return self.x