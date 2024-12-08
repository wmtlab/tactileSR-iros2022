import numpy as np
import matplotlib.pyplot as plt
import cv2
np.set_printoptions(threshold=np.inf)

class RegTactileData():
    def __init__(self):
        pass

    def plotRegData2D(self, data_x, data_y, data_z):
        xy_vmin, xy_vmax = -0.5, 0.5
        z_vmin, z_vmax = 0, 2
        cmap = 'winter'

        fig = plt.figure()
        ax_1 = fig.add_subplot(131)
        ax_2 = fig.add_subplot(132)
        ax_3 = fig.add_subplot(133)
        ax_1.imshow(data_x,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
        ax_2.imshow(data_y,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
        ax_3.imshow(data_z,vmin=z_vmin, vmax=z_vmax, cmap=cmap)

        ax_1.set_xticks([])
        ax_1.set_yticks([])
        ax_2.set_xticks([])
        ax_2.set_yticks([])
        ax_3.set_xticks([])
        ax_3.set_yticks([])

        ax_1.set_title('X axis')
        ax_2.set_title('Y axis')
        ax_3.set_title('Z axis')
        plt.savefig('./out.png')
        plt.close()

    def readData(self, filepath):
        data_seq = np.load(filepath, allow_pickle=True)
        data_seq = data_seq.reshape((data_seq.shape[0], data_seq.shape[1],
                            data_seq.shape[2],16,3))
        data_seq_x = data_seq[:,:,:,:,0]
        data_seq_y = data_seq[:,:,:,:,1]
        data_seq_z = data_seq[:,:,:,:,2]
        return data_seq_x, data_seq_y, data_seq_z

    def TactileSeq2Single(self, data, debuginfo='z'):
        filter_data = np.zeros((data.shape[0],data.shape[1],data.shape[3]))
        filter_data = np.zeros((data.shape[0],data.shape[1],data.shape[3]))
        for i in range(data.shape[0]):              # pos_x
            for j in range(data.shape[1]):          # pos_y
                count = 0
                for index in range(data.shape[2]):  # seq_num
                    # filter_data[i,j] = filter_data[i,j] + data[i,j,index,:]
                    # count += 1
                    # single_data_sum = data[i,j,index,:].sum()
                    if not data[i,j,index,:].sum() == 0:
                        filter_data[i,j] = filter_data[i,j] + data[i,j,index,:]
                        count += 1
                if not count == 0:
                    filter_data[i,j] = filter_data[i,j] / count
                else:
                    print("***** [{}] | TactileSeq2Single[{}][{}] ERROR! divide num = 0 ! *****"
                        .format(debuginfo, i,j))
        return filter_data

    def thresholdFilter(self, seq_data_x, seq_data_y, seq_data, thresholdScale=0.1):
        for i in range(seq_data.shape[0]):
            for j in range(seq_data.shape[1]):
                threshold_data = 0
                for seq_index in range(seq_data.shape[2]):
                    if threshold_data < seq_data[i][j][seq_index].mean():
                        threshold_data = seq_data[i][j][seq_index].mean()
                for seq_index in range(seq_data.shape[2]):
                    if seq_data[i][j][seq_index].mean() < threshold_data*thresholdScale:
                        seq_data[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
        for i in range(seq_data.shape[0]):
            for j in range(seq_data.shape[1]):
                for seq_index in range(seq_data.shape[2]):
                    if seq_data[i][j][seq_index].sum() == 0:
                        seq_data_x[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
                        seq_data_y[i][j][seq_index] = np.zeros((seq_data.shape[-1]))
        return seq_data_x, seq_data_y, seq_data 
    
    def thresholdFilterSeqs(self, seq_data, sample_num=6, LowThresholdScale=0.3):
        index_start = 0
        index_end = 0
        for index in range(seq_data.shape[0]):
            threshold_data = 0
            if threshold_data < seq_data[index].sum():
                threshold_data = seq_data[index].sum()
        for index in range(seq_data.shape[0]):
            if LowThresholdScale*threshold_data < seq_data[index].sum():
                index_start = index
                index_end = index + sample_num
                break
        return index_start, index_end

    def maxminNormalization(self,seq_data_x, seq_data_y, seq_data_z):
        x_min, x_max = 0, np.abs(seq_data_x).max()
        y_min, y_max = 0, np.abs(seq_data_y).max()
        z_min, z_max = 0, np.abs(seq_data_z).max()
        # return seq_data_x/x_max, seq_data_y/y_max, seq_data_z/z_max
        return seq_data_x/500, seq_data_y/500, seq_data_z/1000

    def maxminNormalization_xyz(self,seq_data_x, seq_data_y, seq_data, method='mean'):
        for i in range(seq_data.shape[0]):
            for j in range(seq_data.shape[1]):
                if method == 'mean':
                    max_mean = 0
                    min_mean = 0
                    for seq_index in range(seq_data.shape[2]):
                        if seq_data[i][j][seq_index].mean() > max_mean:
                            max_mean = seq_data[i][j][seq_index].mean()
                        if seq_data[i][j][seq_index].mean() < min_mean:
                            min_mean = seq_data[i][j][seq_index].mean()
                    for seq_index in range(seq_data.shape[2]):
                        seq_data_x[i][j][seq_index] = seq_data_x[i][j][seq_index] / (max_mean - min_mean)
                        seq_data_y[i][j][seq_index] = seq_data_y[i][j][seq_index] / (max_mean - min_mean)
                        seq_data[i][j][seq_index] = (seq_data[i][j][seq_index] - min_mean) / (max_mean - min_mean)
                else:
                    max_data = seq_data[i][j].max()
                    min_data = seq_data[i][j].min()
                    for seq_index in range(seq_data.shape[2]):
                        seq_data[i][j][seq_index] = (seq_data[i][j][seq_index] - min_data) / (max_data - min_data)

        return seq_data_x, seq_data_y, seq_data

    def regData(self, data):
        """
        (pos_x, pos_y, 16) -> (pos_x * 4, pos_y * 4)
        """
        sample_num_x = data.shape[0]
        sample_num_y = data.shape[1]
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_x):
                single_data = data[j,i,:].reshape((4, 4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*m + i][sample_num_y*n + j] = single_data[m][n]
        return reg_data
    
    def regData_5(self, data):
        sample_num_x = 5
        sample_num_y = 5
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_y):
                single_data = data[j*2, i*2, :].reshape((4,4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*m + i][sample_num_y*n+j] = single_data[m][n]
        return reg_data
            
    def regData_2(self, data):
        sample_num_x = 2
        sample_num_y = 2
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_y):
                single_data = data[j*5, i*5, :].reshape((4,4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*m + i][sample_num_y*n+j] = single_data[m][n]
        return reg_data
    
    def regData_new_5(self, data):
        sample_num_x = 5
        sample_num_y = 5
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_y):
                single_data = data[j*2, i*2, :].reshape((4,4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*(m+1) - i-1][sample_num_y*(n+1) - j-1] = single_data[m][n]                        
        return reg_data
            
    def regData_new_2(self, data):
        sample_num_x = 2
        sample_num_y = 2
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_y):
                single_data = data[j*5, i*5, :].reshape((4,4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*(m+1) - i-1][sample_num_y*(n+1) - j-1] = single_data[m][n]                    
        return reg_data
    
    def regData_new(self, data):
        sample_num_x = data.shape[0]
        sample_num_y = data.shape[1]
        reg_data = np.zeros((sample_num_x * 4, sample_num_y * 4))
        for i in range(sample_num_x):
            for j in range(sample_num_x):
                single_data = data[j,i,:].reshape((4, 4))
                for m in range(4):
                    for n in range(4):
                        reg_data[sample_num_x*(m+1) - i-1][sample_num_y*(n+1) - j-1] = single_data[m][n]
        return reg_data

    def smoothPattern(self, data_x, data_y, data_z, smoth_method=1):
        if smoth_method==1:
            gaussKernel = 3
            gaussSigma = 1
            reg_data_x = cv2.GaussianBlur(data_x, (gaussKernel,gaussKernel), gaussSigma)
            reg_data_y = cv2.GaussianBlur(data_y, (gaussKernel,gaussKernel), gaussSigma)
            reg_data_z = cv2.GaussianBlur(data_z, (gaussKernel,gaussKernel), gaussSigma)
        elif smoth_method==2:
            d = 0
            sigmaColor = 5
            sigmaSpace = 2
            reg_data_x = cv2.bilateralFilter(data_x.astype(np.float32), d, sigmaColor, sigmaSpace)
            reg_data_y = cv2.bilateralFilter(data_y.astype(np.float32), d, sigmaColor, sigmaSpace)
            reg_data_z = cv2.bilateralFilter(data_z.astype(np.float32), d, sigmaColor, sigmaSpace)
        else:
            reg_data_x = data_x
            reg_data_y = data_y
            reg_data_z = data_z
        return reg_data_x, reg_data_y, reg_data_z

    def scalePattern(self, data_x, data_y, data_z, scale_num=500):
        return data_x/scale_num, data_y/scale_num, data_z/(2*scale_num)

    def plotRegData3D(self, data_x, data_y, data_z):
        fig_3d = plt.figure()
        ax_1 = fig_3d.add_subplot(131, projection='3d')
        ax_2 = fig_3d.add_subplot(132, projection='3d')
        ax_3 = fig_3d.add_subplot(133, projection='3d')
        x_taxel, y_taxel = data_x.shape[0], data_x.shape[1]

        X = np.arange(0, x_taxel, 1)
        Y = np.arange(0, y_taxel, 1)
        X, Y = np.meshgrid(X, Y)

        ax_1.plot_surface(X, Y, self.getZ(data_x, X, Y), cmap='Greys')
        ax_2.plot_surface(X, Y, self.getZ(data_y, X, Y), cmap='Greys')
        ax_3.plot_surface(X, Y, self.getZ(data_z, X, Y), cmap='Greys')
        plt.show()

    def getZ(self, data, x, y):
      return data[x, y]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x) / (np.exp(x) + np.exp(-x)))

if __name__ == '__main__':
    path = './R_10x10_3.npy'
    RegTactileData(path=path)

