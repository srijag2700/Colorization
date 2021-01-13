from PIL import Image #Python Imaging Library
import math
import random

# K-Means is randomly finding centroids, assigning points to them so that they cluster...
# and then re-calculating the clusters and centroids until stuff stops changing

# so let's start by creating my own classes for Points and Clusters i guess

class Point:
    def __init__(self, coords):
        self.coords = coords

class Cluster:
    def __init__(self, center, pts):
        self.center = center
        self.pts = pts

# convert photo pixels to my own Points
def convert_to_points(img_name):
    img = Image.open(img_name)
    img = img.convert('RGB')
    width, height = img.size

    points = []
    for i, color in img.getcolors(width*height):
        for _ in range(i):
            points.append(Point(color))
    
    return points

def euclidean_dist(a, b):
    # a and b are Points
    dim = len(a.coords)
    return math.sqrt(sum([(a.coords[i] - b.coords[i])**2 for i in range(dim)]))

#this is where the action happens
class KMeans:
    def __init__(self, num_clusters, min_diff=1):
        self.num_clusters = num_clusters
        self.min_diff = min_diff
    
    def find_center(self, points):
        # center = adding all values up & dividing by # of points
        dim = len(points[0].coords)
        v = [0.0 for i in range(dim)]
        for p in points:
            for i in range(dim):
                v[i] += p.coords[i]
        c = [(val / len(points)) for val in v]
        return Point(c)
    
    def assign_to_cluster(self, clusters, points):
        # assign the points to clusters, which are structured as a list of lists
        pl = [[] for i in range(self.num_clusters)]
        index = 0 # so that VS Code stops giving me an unbound variable error

        for p in points:
            min_dist = float('inf')

            for i in range(self.num_clusters):
                dist = euclidean_dist(p, clusters[i].center)
                if dist < min_dist:
                    min_dist = dist
                    index = i
            
            pl[index].append(p)
        
        return pl
    
    def fit(self, points):
        # recalculation process
        clusters = [Cluster(p, [p]) for p in random.sample(points, self.num_clusters)]

        while True:
            pl = self.assign_to_cluster(clusters, points)
            diff = 0
            for i in range(self.num_clusters):
                if not pl[i]:
                    continue
                old = clusters[i]
                center = self.find_center(pl[i])
                new = Cluster(center, pl[i])
                clusters[i] = new
                diff = max(diff, euclidean_dist(old.center, new.center))

            if diff < self.min_diff:
                break
        
        return clusters

def get_colors(img_name, num_colors=5):
    points = convert_to_points(img_name)
    clusters = KMeans(num_colors).fit(points)
    clusters.sort(key=lambda c: len(c.pts), reverse=True)
    rgbs = [list(map(int, c.center.coords)) for c in clusters]
    return rgbs