#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -Uqq fastbook


# In[2]:


import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *


# In[68]:


path = untar_data(URLs.MNIST_SAMPLE)
#download data from url
#path shows us where the file is
Path.BASE_PATH = path 
#sets ls to show current working directory where data is held


# In[69]:


path.ls()
#output was downlaoded by code in 3rd cell


# In[6]:


(path/'train').ls()


# In[67]:


#let's Import our training dataset 
threes = (path/'train'/'3').ls().sorted()
print(threes)
sevens = (path/'train'/'7').ls().sorted()
print(sevens)


# In[43]:


im3_path = threes[44]
im3 = Image.open(im3_path)
im3


# In[9]:


tensor(im3)[4:10,4:10]
#im3 is an image, computers store images as numbers, using tensors from PyTorch, we can view 
#im3 is stored as numbers,
#we tell pytorch tensor to only show us the rows from index 4 (included) to 10 (not included) and the 
#rows are horizontal, columns are vertical
#columns from index 4 (included) to 10 (not included) of the full tensor that makes up im3


# In[10]:


im3_tensor = tensor(im3)
#now we pick out the part of the im3 tensor that we want and convert it to a pandas dataframe
df = pd.DataFrame(im3_tensor[4:15,4:22])

df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')


# In[11]:


seven_tensors = [tensor(Image.open(o))for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)

# using list comprehension, basically a faster way to create a list, instd of declaring list and using append 
#We could use the lambda function map too https://www.programiz.com/python-programming/list-comprehension


# In[12]:



stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

#we shall create a "layer cake" of all the 3s/7s in our data set, like 
#im3_tensor we have produced 6131, 6265 images of 3s and 7s from our dataset respectively
#we divide by 255 since images, when converted to pixels are expected to be btwn 0 and 1 , as shown above
# the gradient of colors spans from 0 to 255


# In[35]:


stacked_sevens.size() == stacked_sevens.shape
#size of each axis , 6265,28,28 respectively


# In[14]:


len(stacked_sevens.shape) == stacked_sevens.ndim
#above are both equal to dimension,3dimensions


# In[15]:




mean_3 = stacked_threes.mean(0)
mean_7 = stacked_sevens.mean(0)

#Below, we calculate the mean, by averaging out the value of each pixel in each of our 
#stacked sets of images, we have sort of superimposed and then averaged out to find out what the "ideal"
# 7/3 looks like


show_image(mean_3);
show_image(mean_7);


# In[16]:


tester_3 = stacked_threes[1]
show_image(tester_3);


# In[17]:


#Now, lets determine how much tester_3 deviates from our ideal 3, given by mean_3

#To do this, we can find the difference in value between each pixel in tester_3 and mean_3 and add them 
#up and then find the mean.This will return us some negative values but we can always apply the 
#modulus to each of the negative values.This method is called L1 Norm

#We could also find the  difference in value between each pixel in tester_3 and mean_3 , square each difference,
#add them up and then find the mean.We can then apply the square root function to this value.
#This method is called the root mean squared error (RMSE) or L2 norm.

#We will do both


# In[18]:


dist_3_L1 = (tester_3 - mean_3).abs().mean()
dist_3_L2 = ((tester_3 - mean_3)**2).mean().sqrt()
dist_3_L1,dist_3_L2


# In[19]:



#we are now doing L1 and L2 to see how far tester_3 deviates from the mean7

dist_7_L1 = (tester_3 - mean_7).abs().mean()
dist_7_L2 = ((tester_3 - mean_7)**2).mean().sqrt()
dist_7_L1,dist_7_L2


# In[20]:


#instead of doing L1/L2 manually we can call torch.nn.functional that already has the 2 as loss functions

#L1
L1 = F.l1_loss(tester_3.float(),mean_7) 
#L2
L2 = F.mse_loss(tester_3.float(),mean_7).sqrt()


# In[21]:


L1,L2


# In[22]:


#Refresher on Numby Arrays and tensors


# In[23]:


data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)


# In[24]:


#How to access data in tensor ?


# In[25]:


#first index , to the left of "," helps us pick the the horizontal Row
#second index, to the right of "," helps us pick the vertical column 
#we can use the [start:end]  format to pick both the horizntla row and vertical column
#where we include the data in the start but exclude data in the end column
tns[0:1 , 2:3]


# In[26]:


#Computing meterics using Broadcasting, as we know we need to compute Metric, a measure of how effective our model is,
#based on 


# In[27]:


valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls().sorted()])


# In[28]:


# this will fail since "valid_3_tens" is an image that has already been converted to a tensor, in contrast to
#im3_path = threes[44]
#im3 = Image.open(im3_path)
#where im3 is an image that has been just freshly taken out of the test set, Personally 
#I was stumped at first but had to slow down and recollect that im3 is still an image file while valid_3_tens below
# is a tensor

#below will fail when run
#valid_3_tens = valid_3_tens[9]
#Image.open(valid_3_tens)


# In[29]:



#we are creating the "layer cake" of tensors, tensors that were created by opening each image in our validation set
# and appending them into a tensor array then stacking each of the elements in the array

valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape


# In[30]:


#Let's write a function to to find how far our tester_3 image differs from our mean_3 image a.k.a "Ideal 3"
#We have done this earlier but lets make this a function now, We are using the L1 Norm method for now 
#mean(-1,-2) is impportant I'll explain it in abit
def mnist_distance(a,b): 
    return(a-b).abs().mean((-1,-2))
mnist_distance(tester_3, mean_3)


# In[31]:


# all we have done so far is calculate how an image differs from our ideal 3, for our validation however  we need to 
# calculate the distance between the ideal 3 for every image in the validation dataset

valid_3_dist = mnist_distance(valid_3_tens, mean_3)
valid_3_dist, valid_3_dist.shape

#HOW??? We just inserted a 3 dimensional tensor into a function that was meant to be applied to our 2 dimensional 
# tensor (to do the image difference btwn tester_3 and mean_3 ). This time howeer we input a 3dimensional  tensor ,
# 1010 stacked tensors of images from our validation set.


# In[32]:


#In the output above, we receive a 1 dimensional tensor containing the variance from 
#the ideal 3 of each of our images in our dataset, since we have 1010 images of 3s, our tensor has a size of 1010
# on its single axis


#This is called broadcasting in pytorch. When encountering tensors of different shapes/sizes, Pytorch  
#expands the tensor with a s smaller rank  to have the same shape/size as the one with the larger rank
# and then carries out the operations that has been requested

#Lastly note that we did "return(a-b).abs().mean((-1,-2))" in doing this, we tell pytorch to carry out the operations
#specified for each item in the valid_3_tens stack against the mean 3 and that this operation is to be carried out on the
#last and second last axis of each tensor , as specified by -1 and -2 respectively.


# In[59]:


#Lets write a function to tell us if the image is a 3 or a 7. We will check if the mnist_distance between our target
# and mean_3 is less than its distance to mean_7 if it is, the target is a 3, the converse is done for is_7()
def is_3(x): 
    return mnist_distance(x,mean_3) < mnist_distance(x,mean_7)
def is_7(x): 
    return mnist_distance(x,mean_3) > mnist_distance(x,mean_7)


# In[60]:


#lets test our function on the first tensor in the stack of tensors of our validation set of images of the number 3
is_3(valid_3_tens[0]), is_3(valid_3_tens[0]).float()

#we can also apply this function to the whole validation set ... once again using Broadcasting, this returns a tensor
# containing 1 and 0 , 1 being that the image has been determined to be a 3 and 0 being that it has determined to be a 7
is_3(valid_3_tens), is_3(valid_3_tens).float()


# In[66]:


#now lets find the mean ... simply put, on average what is the percentage of correctly identifying this image as a 3, 
# given that the image is actually a 3 and has been taken from the validation set of images of the number 3
#we do the same for the images in the validation set of images of the number 
accuracy_3s =      is_3(valid_3_tens).float().mean()
accuracy_7s =      is_7(valid_7_tens).float().mean()
accuracy_3s , accuracy_7s


# In[71]:


doc(untar_data)


# In[ ]:





# In[ ]:




