import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


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
        self._init_map()

    def _init_map(self):
        self.numbered_matrix_fields = np.unravel_index(np.arange(self.width * self.height)\
                                                        .reshape(self.width, self.height),\
                                                        (self.width, self.height))
        self.map_coords_xy = copy.deepcopy(self.numbered_matrix_fields)
        if self.use_hexagonal_map:
            cord_y = self.map_coords_xy[0]
            cord_y = cord_y.astype(float)
            cord_y = cord_y * np.sqrt(3)
            cord_y[:,(np.arange(self.width)*2)[np.arange(self.width)*2 < self.width]] += np.sqrt(3)/2

            cord_x = self.map_coords_xy[1]
            cord_x = cord_x.astype(float)
            cord_x = cord_x * 3/2

            self.map_coords_xy = (cord_x,cord_y)
        
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
        return np.sqrt((self.map_coords_xy[0]-self.map_coords_xy[0][winning[0],winning[1]])**2 +\
                            (self.map_coords_xy[1]-self.map_coords_xy[1][winning[0],winning[1]])**2)

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

    def plot_map(self, data, classes):
        fig, ax = plt.subplots(figsize=(5,5))
        markers = ['o','s','D',"P","*","+",'v', '1', '3', 'p', 'x']
        colors = ['r','g','b','c','k','y',(0.9,0.2,0.9), (1,0.5,0), (1,1,0.3), "m", (0.4,0.6,0)]

        cords_x = self.map_coords_xy[0].astype(float)
        cords_y = self.map_coords_xy[1].astype(float)

        if self.use_hexagonal_map:
            numVertices = 6
            radius = 1
            orientation = np.radians(30)
            axis_limits = [np.min(cords_x)-1/2*np.sqrt(3),np.max(cords_x)+3/2*np.sqrt(3),np.min(cords_y)-1/2,np.max(cords_y)+5/2]
        else:
            numVertices = 4
            radius = np.sqrt(2)/2
            orientation = np.radians(45)
            axis_limits = [0,np.max(cords_x)+2,0,np.max(cords_y)+2]

        # plot empty map
        for x, y in zip(cords_x.flatten() + 1, cords_y.flatten() + 1):
            ax.add_patch(
                RegularPolygon(
                    (x, y),
                    numVertices=numVertices,
                    radius=radius,
                    orientation=orientation,
                    facecolor="white",
                    alpha=1,
                    edgecolor='k'
                )
            )

        # plot colored map with classes markers
        for cnt,xx in enumerate(data):
            w = self.winner_for_sample(xx)
            ax.plot(
                cords_x[w[0],w[1]]+1,
                cords_y[w[0],w[1]]+1,
                markers[classes[cnt][0]],
                markerfacecolor='None',
                markeredgecolor=colors[classes[cnt][0]],
                markersize=12+classes[cnt]*2,
                markeredgewidth=2
            )
            ax.add_patch(
                RegularPolygon(
                    (cords_x[w[0],w[1]] + 1, cords_y[w[0],w[1]] + 1),
                    numVertices = numVertices,
                    radius = radius,             
                    orientation = orientation,             
                    facecolor = colors[classes[cnt][0]],
                    alpha = 0.05,
                    edgecolor = 'k'
                )
            )

        ax.axis(axis_limits)
        ax.set_title("Data represented by the Kohonen Network")
        plt.show() 