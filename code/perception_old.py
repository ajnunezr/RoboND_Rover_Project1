import numpy as np
import cv2

# I set the threshold here for comodity.
#terrainThresh=(170, 170, 170)
terrainThresh=(160, 160, 160)  
rockThresh=(110,110,50)

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=terrainThresh):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

    ############################# ADDED #############################
    
def find_obstacle(img, rgb_thresh=terrainThresh):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # below_threshold will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] <= rgb_thresh[0]) \
                & (img[:,:,1] <= rgb_thresh[1]) \
                & (img[:,:,2] <= rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

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
    
    #####################################################################
    
    
# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)  # Note to self, This make the transformation Matrix. 
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    # This part is added to add de FOV mask
    # The mask is just a "full white" image warped.
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))    
    return warped, mask

# We have to noted that now perspect_transform return two images, the warped one, and the mask. We need to take that into
# account for the posterior uses of the function. 


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # 1) Define source and destination points for perspective transform AND PARAMETERS
	# Define calibration box in source (actual) and destination (desired) coordinates
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover
    bottom_offset = 6
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    #Actual rover "gps" 
    rover_xpos = Rover.pos[0]
    rover_ypos = Rover.pos[1]
    rover_yaw  = Rover.yaw
    #Threshold
    terrainThresh=(160, 160, 160)
    rockThresh=(110,110,50)
    
    
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]]) 
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    ## Build mask for navigable terrain
    mask2 =  np.array(np.copy(mask)) 
    pts = np.array([[0,0],[0,50],[mask2.shape[1],50],[mask2.shape[1],0]], np.int32)
    cv2.fillPoly(mask2,[pts],0)

    mask3 =  np.array(np.copy(mask)) 
    pts = np.array([[0,0],[0,20],[mask3.shape[1],20],[mask3.shape[1],0]], np.int32)
    cv2.fillPoly(mask3,[pts],0)
    
    #threshed = color_thresh(warped) * mask2
    #obs_map = find_obstacle(warped) * mask3
    #rock_map = find_rocks(warped)
    #
    threshed = color_thresh(warped) *mask
    obs_map = find_obstacle(warped) *mask
    rock_map = find_rocks(warped) *mask

    # Update only if pitch and roll are less than a number:
    if (Rover.roll <= 2 or Rover.roll >= 358) and (Rover.pitch <=2 or Rover.pitch >=358):
        # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    
        Rover.vision_image[:,:,0] = obs_map *255
        Rover.vision_image[:,:,1] = rock_map * 255
        Rover.vision_image[:,:,2] = threshed* 255
        
        # 5) Convert map image pixel values to rover-centric coords
        xpix, ypix = rover_coords(threshed)
        obsxpix,obsypix = rover_coords(obs_map)
        
        # 6) Convert rover-centric pixel values to world coordinates

        # Get navigable pixel positions in world coords
        x_world, y_world = pix_to_world(xpix, ypix,
                                        rover_xpos,rover_ypos,rover_yaw,
                                        world_size,scale)
        # Get obstacle pixel positions in world coords
        x_world_obstacle, y_world_obstacle = pix_to_world(obsxpix, obsypix,
                                                          rover_xpos,rover_ypos,rover_yaw,
                                                          world_size, scale)
        
        
        # 7) Update Rover worldmap (to be displayed on right side of screen)
            # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
            #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
            #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1


        Rover.worldmap[y_world,x_world, 2] += 10
        Rover.worldmap[y_world_obstacle, x_world_obstacle, 0] += 3
                
        
        # 8) Convert rover-centric pixel positions to polar coordinates
        # Update Rover pixel distances and angles
        #Rover.nav_dists = np.sqrt(xpix**2 + ypix**2)
        #Rover.nav_angles = np.arctan2(y_pixel,x_pixel)
        
        #using to_polar_coords duh.
        Rover.nav_dist , Rover.nav_angles = to_polar_coords(xpix, ypix)
        
        ## OTHERS
        
        # I separated the rock part to put all inside the same IF (As in the tutorial)
        # check coordinates of rocks only if there are ones's in the image for rocks
        if rock_map.any():
            rock_x, rock_y = rover_coords(rock_map)
            # Get rock pixel positions in world coords
            x_world_rock, y_world_rock = pix_to_world(rock_x, rock_y, rover_xpos, 
                                    rover_ypos, rover_yaw,world_size, scale)

            Rover.rock_dists , Rover.nav_angles = to_polar_coords(x_world_rock, y_world_rock)
            # Following tip,I did not 
 
            #Update worldmap for rocks (to be displayed on right side of screen)
            Rover.worldmap[y_world_rock, x_world_rock, 1] = 255 
 
        
    return Rover
