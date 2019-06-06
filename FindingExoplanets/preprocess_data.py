import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft(X, n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            df_train_x = df_train_x.apply(self.fourier_transform,axis=1)

            result_array = np.empty((5087, 3197))
            for i in range(0, 5087):
                result_array[i] = df_train_x[i]
            df_train_x = result_array

            df_dev_x = df_dev_x.apply(self.fourier_transform,axis=1)


            df_train_x = df_train_x[:,:(df_train_x.shape[1]//2)]


            res_array = np.empty((570, 3197))
            for j in range(0, 570):
                res_array[j] = df_dev_x[j]

            df_dev_x = res_array

            df_dev_x = df_dev_x[:,:(df_dev_x.shape[1]//2)]


        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = pd.DataFrame(normalize(df_train_x))
            df_dev_x = pd.DataFrame(normalize(df_dev_x))

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.filters.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.filters.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x
