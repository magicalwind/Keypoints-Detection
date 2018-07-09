
# coding: utf-8

# In[62]:


import cv2 
import torch


# In[63]:


torch.cuda.current_device()
torch.cuda.get_device_name(0)


# In[64]:


import os
os.getcwd()


# In[65]:



import matplotlib.pyplot as plt
import numpy as np

# watch for any changes in model.py, if it changes, re-load it automatically
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[132]:



import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Net

net = Net()
print(net)
net.cuda()


# In[133]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

crop = 224

data_transform = transforms.Compose([Rescale(250),
                                    RandomCrop(crop),
                                    Normalize(crop),
                                    ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'


# In[134]:


# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='training_frames_keypoints.csv',
                                             root_dir='/home/tianbai/P1_Facial_Keypoints/data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))


for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())
    print (sample['keypoints'][0:5])


# In[135]:


# load training data in batches
batch_size = 10
train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


# In[136]:



test_dataset = FacialKeypointsDataset(csv_file='test_frames_keypoints.csv',
                                      root_dir='/home/tianbai/P1_Facial_Keypoints/data/test/',
                                      transform=data_transform)


# In[137]:


print (test_dataset[0]['keypoints'].size())


# In[138]:


# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


# ## Test the model on a sample before training
# 
# 

# In[139]:


# test the model on a batch of test images
from torch.autograd import Variable
def net_sample_output():
    criterion = nn.L1Loss().cuda()
    running_loss = 0.0
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        images = images.cuda(async=True)
        
        key_pts = key_pts.cuda(async=True)
        key_pts = Variable(key_pts)
        key_pts = key_pts.type(torch.cuda.FloatTensor)
        
        
        # wrap images in a torch Variable
        # key_pts do not need to be wrapped until they are used for training
        images = Variable(images)

        # convert images to FloatTensors
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        loss = criterion(output_pts, key_pts)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        key_pts = key_pts.view(key_pts.size()[0], 68, -1)
        
        running_loss += loss.data[0]/1000
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts, running_loss
            


# In[140]:


# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts, _ = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


# In[141]:


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# # Visualize the prediction points without training

# In[142]:


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.cpu().numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu().numpy()
        #if i==0:
           # print (predicted_key_pts)
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*112+112
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*112+112
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
          
        #plt.axis('off')

    plt.show()
    
# call it
visualize_output(test_images, test_outputs, gt_pts)


# ## Training
# 
# 
# ---

# In[145]:



import torch.optim as optim

criterion = nn.L1Loss().cuda()

optimizer = optim.Adam(params=net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# In[146]:


def train_net(n_epochs):
    loss_log_tr =[]
    loss_log_te = []
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            
            # wrap them in a torch Variable
            images, key_pts = Variable(images), Variable(key_pts)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.data[0]
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                loss_log_tr.append(running_loss/1000)
                running_loss = 0.0
                _, _, _, test_loss = net_sample_output()
                loss_log_te.append(test_loss)
            #requests.request("POST", "https://nebula.udacity.com/api/v1/remote/keep-alive", headers={'Authorization': "STAR " + token})
    print('Finished Training')
    return loss_log_tr,loss_log_te


# In[147]:


# train your network
n_epochs = 200 # start small, and increase when you've decided on your model structure and hyperparams

train_loss, test_loss= train_net(n_epochs)


# In[149]:


print (len(train_loss))
print (len(test_loss))


# In[154]:


test_loss_sca = list(np.array(test_loss)*10)


# In[159]:


fig = plt.figure(figsize = (15,6))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(train_loss)
ax1.set_title('Train Loss of the NN')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2 = fig.add_subplot(1,2,2)
ax2.plot(test_loss_sca)
ax2.set_title('Test Loss of the NN')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')

plt.show()


# ## Test data
# 
# 

# In[161]:



test_images, test_outputs, gt_pts,_ = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


# In[162]:



visualize_output(test_images, test_outputs, gt_pts)


# Once you've found a good model (or two), save your model so you can load it and use it later!

# In[163]:



model_dir = 'saved_models/'
model_name = 'Tianbai_2ndtrial.pt'

# Save model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)


# In[168]:


# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer
weights1 = net.conv1.weight.data

w = weights1.cpu().numpy()
print (w.shape)
filter_index = 0

print(w[filter_index][0])
print(w[filter_index][0].shape)

# display the filter weights
plt.imshow(w[filter_index][0], cmap='gray')
plt.show()


# ## Filter an image to see the effect of a convolutional kernel
# ---

# In[167]:


import cv2
##TODO: load in and display any image from the transformed test dataset
image = test_images[4].data
image = image.cpu().numpy()
image = np.transpose(image, (1, 2, 0))
image = np.squeeze(image)
#print (image.shape)
plt.imshow(image, cmap='gray')
## TODO: Using cv's filter2D function,
## apply a specific set of filter weights (like the one displayed above) to the test image
w = np.squeeze(w)
fig = plt.figure(figsize = (30,10))
columns = 5*2
rows =2
for i in range(0,20):
    fig.add_subplot(2,10,i+1)
    if ((i%2)==0):
        plt.imshow(w[int(i/2)], cmap = 'gray')
    else:
        fil_image = cv2.filter2D(image,-1,w[int((i-1)/2)])
        plt.imshow(fil_image,cmap ='gray')

plt.show()

