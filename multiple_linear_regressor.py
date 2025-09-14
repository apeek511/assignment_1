import numpy as np

class MultipleLinearRegressor:
    def __init__ (self) -> None:
        """
        Initializes a multiple linear regression model.

        No args because we are not setting defaults and the 
        data and ground truth are passed as arguments to the fit function

        Attributes:
            _parameters: dictionary to eventually hold the parameters of the model

        """
        self._parameters = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates regression plane, expands upon simple linear regression

        Based on the eqation for optimal parameters from the assignment

        Args:
            X (np.ndarray): data, shape (n x p)
            y (np.ndarray): ground truth output values, shape (n x 1)
        """

        #the data matrix X_tilde is n data points by n features and an added column of ones
        X_tilde = np.c_[X, np.ones(X.shape[0])]

        #optimal parameters configuration w tilde star = (X_tilde^T * X_tilde)^-1 * X_tilde^T * y
        inverse_transpose = np.linalg.inv(X_tilde.T @ X_tilde)
        self._parameters = inverse_transpose @ X_tilde.T @ y

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        """
        Predicts output values for new data points.

        Args:
            new_data (np.ndarray): New input data
        
        Returns:
            np.ndarray: Predicted output values
        """
        #add a column of ones to the new data so that it matches dimensions
        new_data_tilde = np.c_[new_data, np.ones(new_data.shape[0])]

        return new_data_tilde @ self._parameters
    