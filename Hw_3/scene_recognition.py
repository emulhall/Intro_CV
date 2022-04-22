import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
import os.path


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, stride, size):
    sift=cv2.xfeatures2d.SIFT_create()
    #Build up the list of keypoints by moving through the image by given stride
    kps = []
    for v in range(0,img.shape[0],stride):
        for u in range(0,img.shape[1],stride):
            #Use given size to build a keypoint
            kp = cv2.KeyPoint(u, v, size)
            kps.append(kp)
    #Use sift.compute with our collected keypoints to get descriptors
    kps,dense_feature=sift.compute(img, kps)
    return dense_feature


def get_tiny_image(img, output_size):
    #Guarantee float
    img = img/1.0

    feature = np.zeros(output_size)

    #Note that the output size is defined to be given as (w,h)
    v_stride = img.shape[0]//output_size[1]
    u_stride = img.shape[1]//output_size[0]

    for v in range(0,feature.shape[0]):
        for u in range(0,feature.shape[1]):
            temp = img[(v*v_stride):(v*v_stride+v_stride), (u*u_stride):(u*u_stride+u_stride)]
            #Box filter
            feature[v,u]=temp.mean()

    #Ensure zero mean
    feature=feature-feature.mean()

    #Ensure unit length
    feature = feature/np.linalg.norm(feature)

    #We need to vectorize it before we return it
    return feature.flatten()


def predict_knn(feature_train, label_train, feature_test, k):
    label_test_pred = []
    #Get nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(feature_train)
    distances, indices = nbrs.kneighbors(feature_test)
    #Go through the indices to take a majority rules vote for each test sample
    for i in range(indices.shape[0]):
        votes={}
        matches = indices[i]
        for m in matches:
            l = label_train[m]
            if(l in votes):
                votes[l]+=1
            else:
                votes[l]=1
        label_test_pred.append(max(votes, key=votes.get))


    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    confusion = np.zeros((len(label_classes), len(label_classes)))
    print("Building feature_train...")
    #Build feature_train
    feature_train = np.zeros((len(img_train_list), 256))
    for i in range(len(img_train_list)):
        im = cv2.imread(img_train_list[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        tiny_im = get_tiny_image(gray, (16,16))
        feature_train[i] = tiny_im
    print("Feature train built.")

    #Build feature_test
    print("Building feature_test...")
    feature_test = np.zeros((len(img_test_list), 256))
    for i in range(len(feature_test)):
        test_im = cv2.imread(img_test_list[i])
        gray_test = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
        test_tiny_im = get_tiny_image(gray_test, (16,16))
        feature_test[i] = test_tiny_im
    print("Feature test built.")
    #Use predict_knn to make predictions
    print("Final KNN...")
    pred = predict_knn(feature_train, label_train_list, feature_test, 5)

    #Build confusion matrix from ground truth labels and predictions
    confusion = np.zeros((len(label_classes), len(label_classes)))
    count=0
    for i in range(len(pred)):
        confusion[label_classes.index(label_test_list[i]), label_classes.index(pred[i])]+=1
        if(pred[i]==label_test_list[i]):
            count+=1

    accuracy=count/len(label_test_list)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    kmeans = KMeans(n_clusters=dic_size, n_init=5, max_iter=200).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    bow_feature=np.zeros(vocab.shape[0])
    #Get the nearest cluster using nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(vocab)
    distances, indices = nbrs.kneighbors(feature)

    #Go through the indices and build up the BOW
    for i in range(len(indices)):
        bow_feature[indices[i]]+=1

    #Normalize
    bow_feature=bow_feature/np.linalg.norm(bow_feature)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    #Build dense_feature_list from the training data
    dense_feature_list=[]
    dense_feature_train=[]
    print("Creating training dense feature list...")
    for i in range(len(img_train_list)):
        im = cv2.imread(img_train_list[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dense_feature = compute_dsift(gray, 15,20)
        dense_feature_list.extend(dense_feature)
        dense_feature_train.append(dense_feature)

    #Convert to numpy
    dense_feature_list=np.array(dense_feature_list)
    print("Dense feature list has been built.")

    #Build vocabulary
    print("Building visual dictionary...")
    vocab = build_visual_dictionary(dense_feature_list, 150)
    print("Visual dictionary has been built.")


    #Build up the BOW features for the training set
    print("Building up training BOW...")
    train_bow = []
    for i in range(len(dense_feature_train)):
        train_bow.append(compute_bow(dense_feature_train[i], vocab))
    print("Training BOW complete.")

    #Build up the dense features for the test set
    print("Building up testing dense feature list...")
    dense_feature_test = []
    for i in range(len(img_test_list)):
        im = cv2.imread(img_test_list[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dense_feature = compute_dsift(gray, 15,20)
        dense_feature_test.append(dense_feature)
    print("Dense feature list for testing set has been built.")

    #Build up testing BOW
    print("Building up testing BOW...")
    test_bow=[]
    for i in range(len(dense_feature_test)):
        test_bow.append(compute_bow(dense_feature_test[i], vocab))
    print("Testing BOW complete.")

    #Apply KNN
    print("Final KNN...")
    pred=predict_knn(train_bow, label_train_list, test_bow, 20)

    #Build confusion matrix from ground truth labels and predictions
    confusion = np.zeros((len(label_classes), len(label_classes)))
    count=0
    for i in range(len(pred)):
        confusion[label_classes.index(label_test_list[i]), label_classes.index(pred[i])]+=1
        if(pred[i]==label_test_list[i]):
            count+=1

    accuracy=count/len(label_test_list)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):
    label_train = np.array(label_train)
    classes = np.unique(label_train)
    total_pred = []
    for c in classes:
        #Convert labels to one vs. all
        labels=np.where(label_train==c, 1, 0)
        #Fit SVC to training data
        svc = LinearSVC(C=1)
        svc.fit(feature_train, labels)
        #Get confidence values for testing data
        confidence = svc.decision_function(feature_test)
        total_pred.append(confidence)

    total_pred = np.array(total_pred)

    #Get the index of the maximum confidence
    max_pred=np.argmax(total_pred, axis=0)
    #Convert from indices to labels
    label_test_pred=[]
    for i in range(len(max_pred)):
        label_test_pred.append(classes[max_pred[i]])

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    #Build dense_feature_list from the training data
    dense_feature_list=[]
    dense_feature_train=[]
    print("Creating training dense feature list...")
    for i in range(len(img_train_list)):
        im = cv2.imread(img_train_list[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dense_feature = compute_dsift(gray, 15,20)
        dense_feature_list.extend(dense_feature)
        dense_feature_train.append(dense_feature)

    #Convert to numpy
    dense_feature_list=np.array(dense_feature_list)
    print("Dense feature list has been built.")

    #Build vocabulary
    print("Building visual dictionary...")
    vocab = build_visual_dictionary(dense_feature_list, 150)
    print("Visual dictionary has been built.")


    #Build up the BOW features for the training set
    print("Building up training BOW...")
    train_bow = []
    for i in range(len(dense_feature_train)):
        train_bow.append(compute_bow(dense_feature_train[i], vocab))
    print("Training BOW complete.")

    #Build up the dense features for the test set
    print("Building up testing dense feature list...")
    dense_feature_test = []
    for i in range(len(img_test_list)):
        im = cv2.imread(img_test_list[i])
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dense_feature = compute_dsift(gray, 15,20)
        dense_feature_test.append(dense_feature)
    print("Dense feature list for testing set has been built.")

    #Build up testing BOW
    print("Building up testing BOW...")
    test_bow=[]
    for i in range(len(dense_feature_test)):
        test_bow.append(compute_bow(dense_feature_test[i], vocab))
    print("Testing BOW complete.")


    #Apply SVM
    print("Final SVM...")
    pred = predict_svm(train_bow, label_train_list, test_bow)

    #Build confusion matrix from ground truth labels and predictions
    confusion = np.zeros((len(label_classes), len(label_classes)))
    count=0
    for i in range(len(pred)):
        confusion[label_classes.index(label_test_list[i]), label_classes.index(pred[i])]+=1
        if(pred[i]==label_test_list[i]):
            count+=1

    accuracy=count/len(label_test_list)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    #classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




