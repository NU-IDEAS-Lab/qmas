import torch
import numpy as np

@torch.no_grad
class Predictor:
    ''' This class implements a KF-based predictor.
        Based on https://doi.org/10.1109/TAC.2004.834121 '''

    def __init__(self, obs_dim, action_dim, args, device=None):
        # super().__init__()

        transition_dim = obs_dim + action_dim
        self.prediction_horizon = args.diffusion_horizon

        # Initialize the Kalman Filter parameters.
        # TODO: State transition matrix F and control input matrix B should only be True at the adversary's position.
        # TODO: Observation matrix H...
        # TODO: Noise matrices should be set correctly.
        self.F = torch.eye(transition_dim, device=device)
        self.B = torch.zeros((transition_dim, action_dim), device=device)
        self.H = torch.eye(transition_dim, device=device)
        self.Q = torch.eye(transition_dim, device=device) * 0.0 #args.kf_process_noise
        self.R = torch.eye(transition_dim, device=device) * 0.0 #args.kf_measurement_noise

        # TODO: Should probably create a new `initialize` method to set these and create the Kalman Filter.
        # Initial state and covariance.
        self.x0 = torch.zeros((transition_dim,), device=device)
        self.P0 = torch.eye(transition_dim, device=device)

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

        # Set up trajectory and visibility mask.
        trajectory = trajectory.unsqueeze(0)  # Add sample dimension.
        if visibility_mask == None:
            visibility_mask = torch.ones_like(trajectory)  # Default to all visible.
        else:
            visibility_mask = visibility_mask.unsqueeze(0) # Add sample dimension.

        # Apply the visibility mask to the trajectory.
        trajectory = trajectory * visibility_mask

        # Initialize the prediction tensor.
        prediction = torch.zeros((1, trajectory.shape[1], trajectory.shape[2]), device=trajectory.device)

        # If we have a previous prediction, use it for autoregression.
        if prediction_prev is not None:
            prediction[0, 0, :] = prediction_prev[0, 0, :]
        else:
            prediction[0, 0, :] = trajectory[0, 0, :]

        # Iterate over the trajectory to get predictions.
        for t in range(1, trajectory.shape[1]):
            # Predict the next state using the Kalman Filter.
            self.kf.predict(trajectory[0, t-1, -trajectory.shape[2]:])
            predicted_state = self.kf.x
            
            # Update the Kalman Filter with

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
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x


    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x