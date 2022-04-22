import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_differential_filter():
    filter_x=np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]])
    filter_y=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    #Output 3x3 filters that differentiate along the x and y directions
    return filter_x, filter_y


def filter_image(im, filter):
    #Create an output matrix of the same shape as our input matrix
    im_filtered=np.zeros(np.shape(im))

    #Pad image with zeros to prevent losing dimensions of image
    im_padded=np.pad(im,(1,1))

    #Iterate through image applying the filter
    #Our filter first moves in the x direction, then down the y direction
    for y in range(im.shape[1]):
    	for x in range(im.shape[0]):
    		#Get the 3x3 slice of the image to multiply
    		temp=im_padded[x:x+3, y:y+3]
    		#Apply the filter by multiplying the filter elementwise on our slice and taking the sum
    		im_filtered[x,y]=(filter*temp).sum()
    return im_filtered


def get_gradient(im_dx, im_dy):
    #Set up our magnitude and orientation matrices
    #They will be mxn, like our differential images
    grad_mag=np.zeros(np.shape(im_dx))
    grad_angle=np.zeros(np.shape(im_dx))

    #Prevent division by zero
    im_dx=im_dx+1e-15

    #Iterate through the differential images and calculate the magnitude and angle
    #Magnitude is equal to the square root of (dI/dx)^2 + (dI/dy)^2 (Slide 12 of lecture 4)
    #Angle is equal to inverse tan of (dI/dy / dI/dx) (Slide 13 of lecture 4)
    for x in range(grad_mag.shape[0]):
    	for y in range(grad_mag.shape[1]):
    		grad_mag[x,y]=math.sqrt((im_dx[x,y]**2)+(im_dy[x,y]**2))
    		#We're looking for an unsigned angle between 0 and pi, so make sure we add 180 degrees
    		grad_angle[x,y]=math.atan(im_dy[x,y]/im_dx[x,y])+np.pi

    #Outputs are magnitude and orientation of the gradient images (size: mxn) The range of the angle should be [0, pi)
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    #Given the magnitude and orientation of the gradients per pixel, build the histogram of oriented gradients for each cell
    m=grad_mag.shape[0]
    n=grad_mag.shape[1]
    ori_histo=np.zeros((int(m/cell_size), int(n/cell_size), 6))

    for i in range(ori_histo.shape[0]):
        for j in range(ori_histo.shape[1]):
            sliced_grad_mag=grad_mag[cell_size*i:cell_size*i+cell_size, cell_size*j:cell_size*j+cell_size]
            sliced_grad_angle=grad_angle[cell_size*i:cell_size*i+cell_size, cell_size*j:cell_size*j+cell_size]
            for u in range(sliced_grad_angle.shape[0]):
                for v in range(sliced_grad_angle.shape[1]):
                    #bin 1 is [2.87979, pi) U [0, 0.261799)
                    if (sliced_grad_angle[u,v]>2.87979 and sliced_grad_angle[u,v]<math.pi) or (sliced_grad_angle[u,v]>=0 and sliced_grad_angle[u,v]<0.261799):
                        ori_histo[i,j,0]=ori_histo[i,j,0]+grad_mag[u, v]
                    #bin 2 is [0.261799, 0.785398)
                    elif (sliced_grad_angle[u,v]>=0.261799 and sliced_grad_angle[u,v]<0.785398):
                        ori_histo[i,j,1]=ori_histo[i,j,1]+grad_mag[u, v]
                    #bin 3 is [0.785398, 1.309)
                    elif (sliced_grad_angle[u,v]>=0.785398 and sliced_grad_angle[u,v]<1.309):
                        ori_histo[i,j,2]=ori_histo[i,j,2]+grad_mag[u, v]
                    #bin 4 is [1.309, 1.8326)
                    elif (sliced_grad_angle[u,v]>=1.309 and sliced_grad_angle[u,v]<1.8326):
                        ori_histo[i,j,3]=ori_histo[i,j,3]+grad_mag[u, v]
                    #bin 5 is [1.8326, 2.35619)
                    elif (sliced_grad_angle[u,v]>=1.8326 and sliced_grad_angle[u,v]<2.35619):
                        ori_histo[i,j,4]=ori_histo[i,j,4]+grad_mag[u, v]
                    #bin 6 is [2.35619, 2.87979)
                    elif (sliced_grad_angle[u,v]>=2.35619 and sliced_grad_angle[u,v]<2.87979):
                        ori_histo[i,j,5]=ori_histo[i,j,5]+grad_mag[u, v]



    #ori_histo is a 3D tensor with size MxNx6 where M and N are the number of cells along y and x axes
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    m=ori_histo.shape[0]
    n=ori_histo.shape[1]
    ori_histo_normalized=np.zeros((m-(block_size-1), n-(block_size-1), 6*block_size**2))
    for i in range(ori_histo_normalized.shape[0]):
        for j in range(ori_histo_normalized.shape[1]):
            #Build descriptor of the first block by concatenating the HOG within the block, using block_size=2
            block=ori_histo[i:i+block_size,j:j+block_size]
            block=block.reshape((block_size*block_size*ori_histo.shape[2]))
            sum_squared=sum(l*l for l in block)
            #Normalize the descriptor
            ori_histo_normalized[i,j]=block*(1/math.sqrt(sum_squared+(0.001)**2))
    #Move to the next block with stride 1 and iterate

    #ori_histo_normalized is the normalized histogram
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    #Get differential images using get_differential_filter and filter_image
    filter_x, filter_y=get_differential_filter()
    im_dx=filter_image(im, filter_x)
    im_dy=filter_image(im, filter_y)
    #Compute the gradients using get_gradient
    grad_mag, grad_angle=get_gradient(im_dx, im_dy)
    #Build the HoG using build_histogram
    ori_histo=build_histogram(grad_mag, grad_angle, 8)
    #Build the descriptor using get_block_descriptor
    ori_histo_normalized=get_block_descriptor(ori_histo, 2)
    #Return long vector(hog) by concatenating all block descriptors
    hog=ori_histo_normalized.reshape(-1)

    # visualize to verify
    #visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()





def face_recognition(I_target, I_template):
    #Use thresholding and non-maximum suppresion with IoU 50% to localize the faces
    temp_hog=extract_hog(I_template)
    #The below 5 lines are for verification. You can uncomment them to visualize the HOG descriptors of the images
    #I_template_im=I_template.astype('float') / 255.0
    #visualize_hog(I_template_im, temp_hog, 8, 2)
    #whole_targ_hog=extract_hog(I_target)
    #I_target_im=I_target.astype('float') / 255.0
    #visualize_hog(I_target_im, whole_targ_hog, 8,2)
    temp_mean=np.mean(temp_hog)
    temp_norm=temp_hog-temp_mean
    boxes=[]
    target_h, target_w=np.shape(I_target)
    temp_h, temp_w=np.shape(I_template)
    temp_norm_norm=np.linalg.norm(temp_norm)

    for i in range(target_h-temp_h):
        for j in range(target_w-temp_w):
            targ_hog=extract_hog(I_target[i:i+temp_h, j:j+temp_w])
            targ_mean=np.mean(targ_hog)
            targ_norm=targ_hog-targ_mean
            #cross-correlation score between the bounding box patch and the template
            score=np.dot(targ_norm, np.transpose(temp_norm))/(np.linalg.norm(targ_norm)*temp_norm_norm)
            if(score>0.71):
                #each row of the array is [x, y, s] where (x,y) is the top-left corner of the bounding box and s is the normalized 
                boxes.append([j,i,score])
    boxes=np.array(boxes)
    bounding_boxes=[]
    #Non-maximum suppression
    #Sort the boxes by their scores
    sorted_boxes=boxes[boxes[:,2].argsort()]
    while(len(sorted_boxes)>0):
        #Get the max score box
        max_score=sorted_boxes[-1]
        #Add this to our bounding_boxes array
        bounding_boxes.append(max_score)
        #Update the list of boxes
        sorted_boxes=sorted_boxes[:-1]
        to_del=[]
        for b in range(len(sorted_boxes)):

            #Get intersection of boxes
            x1=max(max_score[0], sorted_boxes[b][0])
            x2=min(max_score[0]+temp_w, sorted_boxes[b][0]+temp_w)
            y1=max(max_score[1], sorted_boxes[b][1])
            y2=min(max_score[1]+temp_h, sorted_boxes[b][1]+temp_h)

            #Area of intersection
            area_of_inter=(x2-x1)*(y2-y1)

            #Intersection over union
            iou=area_of_inter/((2*(temp_w)*(temp_h))-area_of_inter)

            if(iou>0.5):
                to_del.append(b)

        sorted_boxes=np.delete(sorted_boxes,to_del,axis=0)

    return  np.array(bounding_boxes)


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.




