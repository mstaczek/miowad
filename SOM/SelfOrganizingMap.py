import copy
import numpy as np

def NeighbourhoodGaussian(distance, epoch, epochs):
    return np.exp(-1 * epoch**2 * distance**2) 

def NeighbourhoodMexicanHat(distance, epoch, epochs):
    e_2 = epoch**2
    d_2 = distance**2
    return (2 * e_2 - 4 * e_2**2 * d_2) * np.exp(-1 * e_2 * d_2)

class SelfOrganizingMap:
    def __init__(self,width, height, features, hexagonal_map=False):
        self.width = width
        self.height = height
        self.features = features
        self.use_hexagonal_map = hexagonal_map
        self.numbered_matrix_fields = np.unravel_index(np.arange(self.width * self.height)\
                                                        .reshape(self.width, self.height),\
                                                        (self.width, self.height))
        
    def init_weights(self, data=None): # randomize if data is None else sample from data
        if data is None:
            self.weights = np.random.normal(size=(self.height,self.width,self.features))
        else:
            size = (self.height,self.width,-1)
            selected_points = [data[np.random.randint(len(data))] for i in range(size[0]*size[1])]
            self.weights = np.array(selected_points).reshape(size) 

    def winner_for_sample(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def get_distances(self, winning): # euclidean, between (i,j) and (m,n)
        if not self.use_hexagonal_map:
            return np.sqrt((self.numbered_matrix_fields[0]-winning[0])**2 +\
                            (self.numbered_matrix_fields[1]-winning[1])**2)
        else:
            raise Exception("not implemented yet")

    def train(self, data, epochs,neighbourhood_scaler, learning_rate, distance_function=NeighbourhoodGaussian):
        rng = np.random.default_rng()  
        data_copy = copy.deepcopy(data)         
        for epoch in range(epochs):
            lr_current = learning_rate * np.exp(-1 * epoch / epochs)
            rng.shuffle(data_copy, axis=0)
            for data_row in data_copy:
                winning = self.winner_for_sample(data_row)
                distances_from_winning = self.get_distances(winning) * neighbourhood_scaler
                self.weights += distance_function(distances_from_winning, epoch, epochs).\
                                reshape(self.height,self.width,-1) * lr_current * (data_row - self.weights) 
