from preprocess import preprocess
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('error')

class LMFilters(object):
    """
    generate LM filters for textons
    """
    def __init__(self):
        pass
    
    def gaussian1d(self, sigma, mean, x, ord):
        """
        return gaussian differentiation for 1st and 2nd order deravatives
        """
        x = np.array(x)
        x_ = x - mean
        var = sigma**2
        # Gaussian Function
        g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

        if ord == 0:
            g = g1
            return g #gaussian function
        elif ord == 1:
            g = -g1*((x_)/(var))
            return g #1st order differentiation
        else:
            g = g1*(((x_*x_) - var)/(var**2))
            return g #2nd order differentiation

    def gaussian2d(self,sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        return g

    def log2d(self,sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        h = g*((x*x + y*y) - var)/(var**2)
        return h

    def makefilter(self, scale, phasex, phasey, pts, sup):

        gx = self.gaussian1d(3*scale, 0, pts[0,...], phasex)
        gy = self.gaussian1d(scale,   0, pts[1,...], phasey)

        image = gx*gy

        image = np.reshape(image,(sup,sup))
        return image

    def makeLMfilters(self):
        sup     = 49
        scalex  = np.sqrt(2) * np.array([1,2,3])
        norient = 6
        nrotinv = 12

        nbar  = len(scalex)*norient
        nedge = len(scalex)*norient
        nf    = nbar+nedge+nrotinv
        F     = np.zeros([sup,sup,nf])
        hsup  = (sup - 1)/2

        x = [np.arange(-hsup,hsup+1)]
        y = [np.arange(-hsup,hsup+1)]

        [x,y] = np.meshgrid(x,y)

        orgpts = [x.flatten(), y.flatten()]
        orgpts = np.array(orgpts)

        count = 0
        for scale in range(len(scalex)):
            for orient in range(norient):
                angle = (np.pi * orient)/norient
                c = np.cos(angle)
                s = np.sin(angle)
                rotpts = [[c+0,-s+0],[s+0,c+0]]
                rotpts = np.array(rotpts)
                rotpts = np.dot(rotpts,orgpts)
                F[:,:,count] = self.makefilter(scalex[scale], 0, 1, rotpts, sup)
                F[:,:,count+nedge] = self.makefilter(scalex[scale], 0, 2, rotpts, sup)
                count = count + 1

        count = nbar+nedge
        scales = np.sqrt(2) * np.array([1,2,3,4])

        for i in range(len(scales)):
            F[:,:,count]   = self.gaussian2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:,:,count] = self.log2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:,:,count] = self.log2d(sup, 3*scales[i])
            count = count + 1

        return F

class preprocessImageWithKernles(object):
    """
    create vector features after applying kernels
    """
    def __init__(self, image):
        self.im = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        #self.im = cv2.resize(self.im, (512, 512))
    def merge(self):
        """
        merge the resulting channels into one
        """
        self.out = cv2.GaussianBlur(self.im,(5,5),0)
        

    def apply_kernel(self, kernels):
        """
        apply LM filter kernels on the image after preprocessing
        """
        #kernel = np.load("LMkernels.npy")
        image = []
        
        for i in range(kernels.shape[2]):
            image.append(cv2.filter2D(self.out, -1, kernels[:,:,i]))

        self.image_after_filters = np.array(image)
        print("creating filtered image...")
        print(self.image_after_filters.shape)
        
    def create_vectors(self):
        """
        return vectors created for each pixel
        """
        #point_vector = []
        #for row in range(self.image_after_filters.shape[1]):
        #    for col in range(self.image_after_filters.shape[2]):
        #        point_vector.append(self.image_after_filters[:,row, col])

        #point_vector = np.array(point_vector)

        #point_vector = np.reshape(point_vector,(self.image_after_filters.shape[1]*self.image_after_filters.shape[2], self.image_after_filters[0]))
        #print(point_vector.shape)
        #np.save("KmeansVectors.npy", point_vector)
        point_vector = list()
        for row in range(self.image_after_filters.shape[1]):
            for col in range(self.image_after_filters.shape[2]):
                point_vector.append(self.image_after_filters[:, row, col])
        self.pixel_point_vector = np.array(point_vector)
        print(self.pixel_point_vector.shape)
        print("initial vector...")
        return self.pixel_point_vector

class createVector(object):

    def __init__(self, vector, image_path):
        self.v = vector
        self.im = cv2.imread(image_path)
        #self.im = cv2.resize(self.im, (512, 512))
    def generateVectors(self):
        """
        generate feature vector with row, col and cluster points
        """
        hold = []
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                hold.append([i, j, -1])
        hold = np.array(hold)
        complete_vector = np.concatenate((self.v, hold), axis= 1)
        print("complete vector ...")
        print(complete_vector.shape)
        return complete_vector

class KMeansLMfilters(object):
    """
    apply kmeans on the feature vector extracted
    """
    def __init__(self, X, no_of_clusters, no_of_iterations):
        self.X = X
        self.clusters = no_of_clusters
        self.iterations = no_of_iterations

    def initializ_centroids(self):
        """
        randomly select data points as centers
        """
        centers =  self.X[np.random.choice(self.X.shape[0], self.clusters, replace=False), :]
        return centers[:,:self.X.shape[1]-3]

    def EuclideanDistance(self, X, Y):
        """
        return euclidean distace between two vectors 
        """
        return np.sqrt(np.sum((X-Y)**2, axis=1))

    def assign_centers(self, centroids):
        """
        assignn closest center to the vecctor
        """
        for features in range(self.X.shape[0]):
            mindist = self.EuclideanDistance(centroids,self.X[features][:self.X.shape[1]-3])
            index = np.argmin(mindist)
            self.X[features][self.X.shape[1]-1] = index
        return self.X

    def update_centroids(self):
        """
        update cluster centers
        """
        class_dict = {}
        for _ in range(self.clusters):
            class_dict[_] = list(self.X[i,:self.X.shape[1]-3] for i in np.asarray((self.X[:,self.X.shape[1]-1]==_).nonzero()))
        for _ in range(self.clusters):
            temp = np.array(class_dict[_])
            class_dict[_] = list(np.mean(np.reshape(temp, (temp.shape[1],temp.shape[2])),axis=0))
        new_centroids = []
        for _ in range(self.clusters):
            new_centroids.append(class_dict[_])
        centers = np.array(new_centroids)
        return centers

    def kMeans(self):
        """
        apply kMeans
        """
        centers = self.initializ_centroids()
        try:
            for _ in range(self.iterations):
                print("[",end = "" )
                print("#"*(_+1), end="")
                print("_"*(self.iterations-_-1), end="")
                print("]", end=" ")
                print("iterations completed:" + str(_+1)+"/" + str(self.iterations))
                self.X = self.assign_centers(centers)
                centers = self.update_centroids()

        except Warning:
            print("Rerun!! Problem with centroid alloaction!")
            exit
        return self.assign_centers(centers)

class reconstructImage(object):
    def __init__(self, feature_vector, image_path, number_of_centers):

        self.v = feature_vector
        self.im = cv2.imread(image_path)
        #self.im = cv2.resize(self.im, (512, 512))
        self.c = number_of_centers

    def reconstruct(self):
        """
        assign pixel values from the feature vectors
        """
        colors = np.random.randint(256, size=(self.c, 3))
        image = np.zeros_like(self.im)
        for _ in range(self.v.shape[0]):
            x,y,k = self.v[_,48], self.v[_,49], self.v[_,50]
            image[x,y] = colors[k]

        return image


def excute():
    #variables
    import sys
    path_to_image = sys.argv[1]
    noOfClusters = int(sys.argv[2])
    iterations = int(sys.argv[3])
    #create LM filters
    F = LMFilters()
    kernel_from_filters = np.array(F.makeLMfilters())
    print(kernel_from_filters.shape)
    np.save("LMkernels.npy", kernel_from_filters, allow_pickle=True)
    del kernel_from_filters
    #create vectors for kmeans
    I = preprocessImageWithKernles(path_to_image)
    I.merge()
    #I.apply_kernel(kernel_from_filters) 
    I.apply_kernel(np.load("LMkernels.npy"))
    features_for_kmeans = I.create_vectors()
    np.save("vectors_for_kmeans.npy", features_for_kmeans)
    print(features_for_kmeans.shape)
    #V = createVector(features_for_kmeans, path_to_image)
    del features_for_kmeans
    V = createVector(np.load("vectors_for_kmeans.npy"), path_to_image)
    final_feature_vectors = V.generateVectors()
    np.save("final_feature_vectors.npy", final_feature_vectors)
    del final_feature_vectors
    #apply kMeans of features obtained
    #K = KMeansLMfilters(final_feature_vectors,noOfClusters, iterations)
    K = KMeansLMfilters(np.load("final_feature_vectors.npy"), noOfClusters, iterations)
    final = K.kMeans()
    print("done")
    np.save("final.npy", final)#reconstruct image from this vector
    print("file saved on disk")
    del final
    #reconstruct image
    #R = reconstructImage(final, path_to_image, noOfClusters)
    R = reconstructImage(np.load("final.npy"), path_to_image, noOfClusters)
    segmented_image = R.reconstruct()
    return segmented_image

if __name__ == "__main__":

    ar = excute() 
    cv2.imshow("check", ar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
