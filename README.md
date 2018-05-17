## Project: Search and Sample Return
### AJNR First Project. 
[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
![alt text][image1]

[image4]: ./output/1_calibData.jpg
[image5]: ./output/warped_example.jpg
[image6]: ./output/warped_threshed_fig.jpg
[image7]: ./output/warped_threshed.jpg
[image7]: ./output/rock_obs_threshed.jpg
[image8]: ./output/circular_filter.jpg
[image9]: ./output/section_filters.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

![alt text][image4]

For the perception step, some functions and filters has been added, for the purpose of improve mapping fidelity, and help with the decision making step. I followed the original steps order, that is, warp the image first, apply color filters latter. 

The first modification comes in the perspective transform step. The general idea is to have a mask that leave only the Field of View (FOV). This is useful for applying to the obstacle image. I will talk latter about this point. 

```python
def perspect_transform(img, src, dst):
    
    """
    # M is the transformation matrix, the one that is going to rotate and translate points
    according to the src and dst selection.
    # cv2.warpPerspective arguments are: an image, a transformation M (builded with
    getPerspectiveTransform), and the shape of the destination (In this case we maintain shape)
    # The mask, that is going to be used for filtering on the obstacle view, is not more than
    the same image, but in binary, and full with 1. 
    """
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))    
    return warped, mask
```


![alt text][image5]

In terrain color threshold, I tried different thresh, but in the end, the suggested one is the better after doing tests running in autonomous mode. The color_thresh() function is modified too.  


![alt text][image6]

For obstacles, the hint is followed. That is, to invert the terrain image.

```python
obstacles = np.absolute(np.float32(threshed)-1)
```

For rock detecting, the *yellow enough* strategy is used.

```python
rockThresh=(110,110,50)
def find_rocks(img, rgb_thresh=rockThresh):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

![image7]


I make some mask functions to try to improve perception step and map fidelity. For future references and help, I leave here basic procedure to make a mask.

*  Make an image full of zeros, exactly with the same size of the images we have been using since now (160x320)
*  Draw the desired figure. For that, you can use several CV2 functions available, Some examples in my mask functions
*  Apply a bit wise_and operation with the original image as src and the mask figure as mask.


```python
def ellipse_mask(img,center,axes,angle,startAngle,endAngle):
    """
    Input should be a 3 channels image 
    """
    
    #1. Make an image full of zeros, exactly with the same size of the images we have been using
    # since now (160x320)
    mask=np.zeros_like(img[:,:,0])
    #2. Draw the desired figure. Note: 1 means color=1 (white), -1 means fill ellipse
    cv2.ellipse(mask, center, axes, angle, startAngle, endAngle,1,-1);
    #3. Use a bitwise_and for selecting the region of interest (RoI)
    img2 = cv2.bitwise_and(img,img,mask=mask)
    return img2, mask
```

![image8]


```python
def maskPoly(img,pts):
    '''
    Make a poligonal mask
    '''
    mask=np.zeros_like(warped[:,:,0])
    # I used the function fillPoly to make a filled poligon, and them make a hand selection
    # for the poligon cut. 
    cv2.fillPoly(mask,[pts],1)
    img2 = cv2.bitwise_and(img,img,mask=mask)
    return img2,mask
```

![image9]



#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a world map.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

Now that we have the new defined functions, process_image() final result is like follows:

```python
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO: 
    # 1) Define source and destination points for perspective transform

    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])


    #And values for the thresh
    terrainThresh=(160, 160, 160)
    rockThresh=(110,110,50)
    
    # 2) Apply perspective transform
    warped, maskFOV = perspect_transform(img, source, destination)
    
    # Masks here.
    # I know calculate all this again is very burdening, but do not have time for refactoring
    # TODO, make function only return mask, and only use a binary image imput or size of original
    # image
    
    # Some others values needed for masks 
    center=(160,155)
    radius=110
    axes = (radius,radius);
    angle=0;
    startAngle=180;
    endAngle=360;
    
    ptsA = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]/3,0],
        [0,0],
        [0,warped.shape[0]]
    ],
               np.int32)

    ptsB = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]/3,0],
        [warped.shape[1]*2/3,0]
    ],
               np.int32)

    ptsC = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]*2/3,0],
        [warped.shape[1],0],
        [warped.shape[1],warped.shape[0]]
    ],
               np.int32)

    
    
    warpedThrash, mask_circle = ellipse_mask(warped,center,axes,angle,startAngle,endAngle)
    warpedThrash, mask_A = maskPoly(warped,ptsA)
    warpedThrash, mask_B = maskPoly(warped,ptsB)
    warpedThrash, mask_C = maskPoly(warped,ptsC)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    navigableMap  = color_thresh(warped) * mask_circle
    navigableMapA = navigableMap * mask_A
    navigableMapB = navigableMap * mask_B
    navigableMapC = navigableMap * mask_C
    
    # Using hint, just the inverse image
    obstaclesMap = np.absolute(np.float32(navigableMap)-1) * maskFOV * mask_circle
    
    rocksMap = find_rocks(warped) * mask_circle
    
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    navXpix , navYpix = rover_coords(navigableMap)
    obsXpix , obsYpix = rover_coords(obstaclesMap)
    rocXpix , rocYpix = rover_coords(rocksMap)
    
    # 5) Convert rover-centric pixel values to world coords
    #Follows section 8
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    
    rover_xpos = data.xpos[data.count]
    rover_ypos = data.ypos[data.count]
    rover_yaw  = data.yaw[data.count]
    
    # Get navigable pixel positions in world coords
    x_world_nav, y_world_nav = pix_to_world(navXpix, navYpix,
                                            rover_xpos,rover_ypos,rover_yaw,
                                            world_size,scale)
    
    # Get obstacle pixel positions in world coords
    x_world_obstacle, y_world_obstacle = pix_to_world(obsXpix, obsYpix,
                                                      rover_xpos,rover_ypos,rover_yaw,
                                                      world_size, scale)
    
    # Get rock pixel positions in world coords
    x_world_rock, y_world_rock = pix_to_world(rocXpix, rocYpix,
                                              rover_xpos, rover_ypos, rover_yaw,
                                              world_size, scale)
    
    
    
    # 6) Update worldmap (to be displayed on right side of screen)
        # Example: data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        
    data.worldmap[y_world_nav,x_world_nav, 2] = 255
    data.worldmap[y_world_obstacle, x_world_obstacle, 0] = 255
    data.worldmap[y_world_rock, x_world_rock, 1] = 255
    
    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped,mask_p = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image

``` 



### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the write up of how and why these functions were modified as they were.

#### perception_step()

The perception step is basically the same than in the notebook, with the addition of some conditionals for only mapping when the car is rolling or pitching a lot.

```python

def perception_step(Rover):

    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    #Actual rover "gps" 
    rover_xpos = Rover.pos[0]
    rover_ypos = Rover.pos[1]
    rover_yaw  = Rover.yaw
    #Threshold
    terrainThresh=(160, 160, 160)
    rockThresh=(110,110,50)

    # 2) Apply perspective transform
    warped, maskFOV = perspect_transform(img, source, destination)
   
    # Masks here.
    # I know calculate all this again is very burdening, but do not have time for refactoring
    # TODO, make function only return mask, and only use a binary image imput or size of original
    # image
    
    # Some others values needed for masks 
    center=(160,155)
    radius=110
    axes = (radius,radius);
    angle=0;
    startAngle=180;
    endAngle=360;
    
    ptsA = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]/3,0],
        [0,0],
        [0,warped.shape[0]]
    ],
               np.int32)

    ptsB = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]/3,0],
        [warped.shape[1]*2/3,0]
    ],
               np.int32)

    ptsC = np.array(
    [
        [warped.shape[1]/2,warped.shape[0]],
        [warped.shape[1]*2/3,0],
        [warped.shape[1],0],
        [warped.shape[1],warped.shape[0]]
    ],
               np.int32)

    
    
    warpedThrash, mask_circle = ellipse_mask(warped,center,axes,angle,startAngle,endAngle)
    warpedThrash, mask_A = maskPoly(warped,ptsA)
    warpedThrash, mask_B = maskPoly(warped,ptsB)
    warpedThrash, mask_C = maskPoly(warped,ptsC)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigableMap  = color_thresh(warped) * mask_circle
    navigableMapA = navigableMap * mask_A
    navigableMapB = navigableMap * mask_B
    navigableMapC = navigableMap * mask_C

    # Using hint, just the inverse image
    obstaclesMap = np.absolute(np.float32(navigableMap)-1) * maskFOV * mask_circle
    # Rocks
    rocksMap = find_rocks(warped) * mask_circle


    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:,:,0] = obstaclesMap *255
    Rover.vision_image[:,:,1] = rocksMap * 255
    Rover.vision_image[:,:,2] = navigableMap* 255
    
    # 5) Convert map image pixel values to rover-centric coords
    navXpix , navYpix = rover_coords(navigableMap)
    obsXpix , obsYpix = rover_coords(obstaclesMap)
    rocXpix , rocYpix = rover_coords(rocksMap)   
   
    navAXpix , navAYpix = rover_coords(navigableMapA)
    navBXpix , navBYpix = rover_coords(navigableMapB)
    navCXpix , navCYpix = rover_coords(navigableMapC)

    # 6) Convert rover-centric pixel values to world coordinates

    # Get navigable pixel positions in world coords
    x_world_nav, y_world_nav = pix_to_world(navXpix, navYpix,
                                            rover_xpos,rover_ypos,rover_yaw,
                                            world_size,scale)
    
    # Get obstacle pixel positions in world coords
    x_world_obstacle, y_world_obstacle = pix_to_world(obsXpix, obsYpix,
                                                      rover_xpos,rover_ypos,rover_yaw,
                                                      world_size, scale)
    
    # Get rock pixel positions in world coords
    x_world_rock, y_world_rock = pix_to_world(rocXpix, rocYpix,
                                              rover_xpos, rover_ypos, rover_yaw,
                                              world_size, scale) 

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    #Updating the map when roll and pitch are very low
    if (Rover.pitch < 1.5 or Rover.pitch > 359) and (Rover.roll < 1.5 or Rover.roll > 358.5):
        Rover.worldmap[y_world_nav,x_world_nav, 2] = 255
        Rover.worldmap[y_world_obstacle, x_world_obstacle, 0] = 255
        Rover.worldmap[y_world_rock, x_world_rock, 1] = 255
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    Rover.nav_dist , Rover.nav_angles = to_polar_coords(navXpix , navYpix)
    Rover.nav_distA , Rover.nav_anglesA = to_polar_coords(navAXpix , navAYpix)
    Rover.nav_distB , Rover.nav_anglesB = to_polar_coords(navBXpix , navBYpix)
    Rover.nav_distC , Rover.nav_anglesC = to_polar_coords(navCXpix , navCYpix)
    

    Rover.obs_dist , Rover.obs_angles = to_polar_coords(obsXpix , obsYpix)
    Rover.rock_dist , Rover.rock_angles = to_polar_coords(rocXpix , rocYpix)
```


The decision step is definitely the more interesting part or the project. I used the state machine programmed and did some little modifications to add an stuck function, an returning home function, and a future rock collecting function.


##### Forward mode
```python
def forward_mode(Rover):

    # Calm the roll and pitch 
    if (Rover.roll >= 1.5 and Rover.roll <= 358.5) or (Rover.pitch >= 1.5 and Rover.pitch <= 359):
       Rover.max_vel = 0.5
    else:
        Rover.max_vel = 5
```

This help with setting the speed when there is too much rolling and pitching

```python

    #Segment for follow only nearest left wall angles
    steer_angles = np.concatenate((Rover.nav_anglesA,Rover.nav_anglesB),axis=0)
    if len(steer_angles) <= 20:
        steer_angles = np.copy(Rover.nav_anglesC) #if you don't have enough at the left, turn right

```

The idea with this is that when the rover has navigable terrain to follow in the left side of the image, it is going to average the steer with that part of the image. If it is not navigable terrain there, then just try with the right side of the image.

```pyton      
    
    # Check the extent of navigable terrain
    if len(Rover.nav_angles) >= Rover.stop_forward:  
        # If mode is forward, navigable terrain looks good 
        # and velocity is below max, then throttle 
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
        else: # Else coast
            Rover.throttle = 0
        Rover.brake = 0
        # Set steering to average angle clipped to the range +/- 15
        #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
        Rover.steer = np.clip(np.mean(steer_angles* 180/np.pi), -15, 15) #Using this eventually crash, not founded explain
    # If there's a lack of navigable terrain pixels then go to 'stop' mode
    elif len(Rover.nav_angles) < Rover.stop_forward:
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'

     # If rover have the six samples, and is near home, just stop. (Homing by chance xD)
    if Rover.samples_collected == 6:
    #if Rover.total_time >= 30:
        dist_to_home = np.sqrt((Rover.pos[0] - Rover.start_pos[0])**2+ (Rover.pos[1] - Rover.start_pos[1])**2)
        #print("dist_home",dist_to_home,"\n\n")
        #print("Have all the balls! \n \n \n \n \n \n \n")
        #print("Home position",Rover.pos,"\n\n")
        if dist_to_home <= 30.0:
            #print("Enter home_alone")
            Rover.mode='home_alone'
 
     ## Checking if stuck
    if Rover.vel <= 0.05:
        Rover.stuck +=1
    else:
        Rover.stuck = 0
 
    if Rover.stuck >=100:
        Rover.mode = 'stuck'
        Rover.stuck = 0
``` 


The last two section are the one who is going to work for the home returning and to do something if the rover gets stuck. Stuck mode is simply a time routine where it tries to goes backward, make some turns and then try to go forward again.

##### Stuck mode 

```python
def stuck_mode(Rover):
    if Rover.stuck == 0:
        Rover.stuck_time = time.time()
        Rover.stuck+=1
    
    if time.time() <= Rover.stuck_time + 3:
        Rover.throttle = -1
    elif time.time()>Rover.stuck_time + 3 and time.time()<=Rover.stuck_time + 6:
        Rover.throttle = 0
        Rover.steer = -10
    elif time.time()>Rover.stuck_time + 6:
        Rover.stuck=0
        Rover.stuck_time=0
        Rover.mode='forward'
``` 


##### Returning home mode 

Returning home is just a routine that make the rover "dance" if is near home.
```python

def home_alone(Rover):
    #this mode does not need a pass to another state, it is the end
    # If rover have the six samples, and is near home, just stop. (Homing by chance xD)
    Rover.brake = 10
    Rover.steer = 10
    Rover.throttle = 0
``` 


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I run the simulation at 1024x768 with fastest quality, having not less than 20 FPSs. The rover map at least 40% of the map with not less than 65% of fidelity, normally in the high 70%. There is a sweet spot on the east of the map that have some obstacles, that is very tricky and the rover normally get stuck there, and unstucking is hard even in manual mode. 

A video of the result is here:

<video width="640" height="380" controls>
  <source src="Out2.mp4" type="video/mp4">
</video>

##### TODO
* Write the rock chasing functions. This can be reach asking if you have any point in the rock image, and then setting steering as the mean angle of the image. When points are near enough, stop and launch a pick.
* A PID for steering and throttle, to add stability to the driving.
* Improve the steering behavior, because actually there is a high tendency to have 15 steering because using only the left part of the image.
* Improve stuck function.  
* Max velocity dependent of the farthest pixel in the navigable area.
 


