import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
import itertools


def find_match(img1, img2):
    # Extract SIFT features
    sift=cv2.xfeatures2d.SIFT_create()
    kp1, des1=sift.detectAndCompute(img1,None)
    kp2, des2=sift.detectAndCompute(img2,None)

    #Convert keypoints to their locations
    loc1=[kp1[index].pt for index in range(len(kp1))]
    loc2=[kp2[index].pt for index in range(len(kp2))]

    x1=[]
    x2=[]
    image_2_to_1=[]
    #Ratio test using Nearest Neighbors training on image 2 and querying image 1
    nbrs=NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices=nbrs.kneighbors(des1)
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctive Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_2_to_1.append((i,matches[d1_index]))


    #To ensure bi-directional consistency we have to do it the other way around as well
    nbrs=NearestNeighbors(n_neighbors=2).fit(des1)
    distances, indices=nbrs.kneighbors(des2)
    image_1_to_2=[]
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctive Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_1_to_2.append((matches[d1_index],i))

    #Let's compare and only keep those that are in both to ensure bi-directional consistency
    for i in range(len(image_2_to_1)):
        if(image_2_to_1[i] in image_1_to_2):
            x1.append(loc1[image_2_to_1[i][0]])
            x2.append(loc2[image_2_to_1[i][1]])


    return np.array(x1), np.array(x2)

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A=None
    inlier=[]
    #Iterate through ransac algorithm given number of times
    for n in range(ransac_iter):
        sample_x1=[]
        sample_x2=[]
        #Sample 3 random points
        indices=random.sample(range(0, len(x1)), 3)
        for indx in indices:
            sample_x1.append(x1[indx])
            sample_x2.append(x2[indx])

        #From these 3 random points, let's build our affine matrix
        C=[]
        b=[]
        for i in range(len(sample_x1)):
            u=sample_x1[i][0]
            u_prime=sample_x2[i][0]
            v=sample_x1[i][1]
            v_prime=sample_x2[i][1]

            b.append(u_prime)
            b.append(v_prime)

            C.append([u, v, 1, 0, 0, 0])
            C.append([0, 0, 0, u, v, 1])

        C=np.array(C)
        b=np.array(b)

        #Cx=b OR x=(C^TC)^-1C^Tb
        c_t_c=np.transpose(C)@C
        #Added a small amount of noise along the diagonal to avoid singularity issues
        c_t_c=c_t_c+np.eye(C.shape[1])*1e-10
        x=np.linalg.inv(c_t_c)@np.transpose(C)@b
        x=np.reshape(x, (2,3))
        #Add the final row of our affine transformation to get our estimate
        A_est=np.vstack((x,[0,0,1]))

        #Now, let's go through the points to determine which of our keypoints are inliers of our estimated affine matrix
        inlier_est=[]
        for i in range(len(x1)):
            p1=[x1[i][0], x1[i][1], 1]
            p2=[x2[i][0], x2[i][1], 1]

            #Estimate point 2 by multiplying our estimated A by point 1
            est=A_est@(p1)

            #Calculate the error as the L2 norm
            err=np.linalg.norm(p2-est)

            #If the error is less than the threshold, append this point to our list of inliers
            if(err<ransac_thr):
                inlier_est.append(i)

        #If the length of our inliers is longer than our current best this becomes our new best        
        if len(inlier_est)>len(inlier):
            A=A_est
            inlier=inlier_est

    return A

def warp_image(img, A, output_size):
    #Get the x2 indices
    x2=np.indices(output_size)
    x2=np.reshape(x2, (2, output_size[0]*output_size[1]))
    x2_xy=np.vstack((x2[1],x2[0]))
    x2_xy=np.vstack((x2_xy, np.ones(x2_xy.shape[1])))
    #Calculate the x1 indices by multiplying by A
    x1_3D=A@x2_xy
    x1=x1_3D[:2]
    #Swap x and y for interpn purposes
    output_ind=np.vstack((x1[1], x1[0]))
    img_warped=interpolate.interpn((np.arange(img.shape[0]), np.arange(img.shape[1])), img, np.transpose(output_ind), method="linear", bounds_error=False, fill_value=1)
    img_warped=np.reshape(img_warped, output_size)
    return img_warped

def warp_image2(img, A, output_size):
    img_warped=np.zeros(output_size, dtype=np.uint8)

    for v in range(output_size[0]):
        for u in range(output_size[1]):
            #Get the point in the output image
            x2=[u,v,1]
            #A is from template to target, and our inverse warping is from template to target
            x1=A@x2

            #Check if it is a valid point (within the image height and width and positive)
            if(x1[0]<img.shape[1] and x1[0]>=0 and x1[1]<img.shape[0] and x1[1]>=0):
                #The value at x2 in the output image becomes the value at x1 in the original image
                img_warped[v][u]=img[int(x1[1]), int(x1[0])]
    return img_warped


def align_image(template, target, A):
    ###Initialize p using A
    W=A

    ###Compute the gradient of the template image
    #X and y filters as seen in Hw1
    filter_u=np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
    filter_v=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    #Filter template by filter_x and filter_y
    im_du=np.zeros(np.shape(template))
    im_dv=np.zeros(np.shape(template))

    #Pad our template image
    im_padded=np.pad(template, (1,1))

    #Iterate through to apply the filter
    for x in range(template.shape[1]):
    	for y in range(template.shape[0]):
    		#Get the slice to multiply
    		temp=im_padded[y:y+3, x:x+3]
    		#Apply the filter by multiplying the filter elementwise on our slice and taking the sum
    		im_du[y,x]=(filter_u*temp).sum()
    		im_dv[y,x]=(filter_v*temp).sum()

    jacobian=np.zeros((template.shape[0], template.shape[1], 2, 6))
    im_grad=np.zeros((template.shape[0], template.shape[1], 1, 2))
    #Iterate through each point to calculate the jacobian, steepest descent image and sum over for Hessian
    for v in range(template.shape[0]):
    	for u in range(template.shape[1]):
    		###Compute the jacobian
    		jacobian[v][u]=np.array([[u,v,1,0,0,0],[0,0,0,u,v,1]])
    		im_grad[v][u]=np.array([[im_du[v][u], im_dv[v][u]]])



    steepest_descent_images=im_grad@jacobian
    H=np.transpose(steepest_descent_images, (0, 1, 3,2))@steepest_descent_images
    H=H.sum(0)
    H=H.sum(0)

    #Initialize delta_p with some large values
    delta_p=np.array([1000,1000,1000,1000,1000,1000])
    i=0
    ###While the magnitude of p is greater than some threshold
    while(np.linalg.norm(delta_p)>0.05):
    	###Warp the target to the template domain
    	#Change from p to the 3x3 affine transformation
    	img_warped=warp_image(target, W, template.shape)

    	###Compute the error image
    	err=img_warped-template

    	###Compute F
    	F=np.transpose(steepest_descent_images,(0, 1, 3,2))@np.reshape(err,(err.shape[0], err.shape[1], 1,1))
    	F=F.sum(0)
    	F=F.sum(0)

    	###Calculate delta p
    	delta_p=np.linalg.inv(H)@F

    	###Update the warping function
    	inv_Warp=np.reshape(delta_p, (2,3))
    	inv_Warp=np.vstack((inv_Warp, [0,0,0]))
    	inv_Warp=inv_Warp+np.eye(3)
    	W=W@np.linalg.inv(inv_Warp)
    	i+=1
    	print(np.linalg.norm(delta_p))
    	print("Iteration: " + str(i)+ " Error: " + str(np.linalg.norm(err)))

    A_refined=W
    return A_refined


def track_multi_frames(template, img_list):
    ###Initialize A using feature matching
    #Find features
    x1, x2 = find_match(template, img_list[0])
    ransac_thr=5
    ransac_iter=500
    #Align using features
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    A_list=[]

    ###Iterate through the frames to align the image
    for i in range(len(img_list)):
    	A = align_image(template, img_list[i], A)
    	A_list.append(A)
    	#Update the template image 
    	template=warp_image(img_list[i], A, template.shape)

    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    #x1, x2 = find_match(template, target_list[0])
    #visualize_find_match(template, target_list[0], x1, x2)

    #ransac_thr=3
    #ransac_iter=500

    #A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    #img_warped1 = warp_image1(target_list[0], A, template.shape)
    #img_warped2 = warp_image2(target_list[0], A, template.shape)

    #plt.imshow(img_warped1, cmap='gray', vmin=0, vmax=255)
    #plt.axis('off')
    #plt.show()

    #A_refined = align_image(template, target_list[0], A)
    #visualize_align_image(template, target_list[0], A, A_refined)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


