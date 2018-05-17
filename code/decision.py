import numpy as np
import time

#Note, all modes should have a return to forward, to prevent stucks 
def forward_mode(Rover):

    # Calm the roll and pitch 
    if (Rover.roll >= 1.5 and Rover.roll <= 358.5) or (Rover.pitch >= 1.5 and Rover.pitch <= 359):
       Rover.max_vel = 0.5
    else:
        Rover.max_vel = 5

    #Segment for follow only nearest left wall angles
    steer_angles = np.concatenate((Rover.nav_anglesA,Rover.nav_anglesB),axis=0)
    if len(steer_angles) <= 20:
        steer_angles = np.copy(Rover.nav_anglesC) #if you don't have enough at the left, turn right
    
    
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

def stop_mode(Rover):
    # If we're in stop mode but still moving keep braking
    if Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
    # If we're not moving (vel < 0.2) then do something else
    elif Rover.vel <= 0.2:
        # Now we're stopped and we have vision data to see if there's a path forward
        if len(Rover.nav_angles) < Rover.go_forward:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
            Rover.steer = -15 # Could be more clever here about which way to turn
        # If we're stopped but see sufficient navigable terrain in front then go!
        if len(Rover.nav_angles) >= Rover.go_forward:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to mean angle
            Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            Rover.mode = 'forward'
    
def stuck_mode(Rover):
   # print("HEEEEYYYY  ____------___--_--____\n \n \n \n ")
   # print(Rover.mode)
    
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

def home_alone(Rover):
    #this mode does not need a pass to another state, it is the end
    # If rover have the six samples, and is near home, just stop. (Homing by chance xD)
    Rover.brake = 10
    Rover.steer = 10
    Rover.throttle = 0

def rock_chasin(Rover):
    pass


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    #print(Rover.nav_anglesA.dtype)
    #print(Rover.nav_anglesA)
    #print(Rover.nav_anglesB)
    #print(steer_angles)
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    print('this is the decision step')
    
    # Save starting position
    if Rover.start_pos==None:
        Rover.start_pos = Rover.pos

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print('Rover.mode ==',Rover.mode,'\n') 


        # Check for Rover.mode status
        #Note, Machine states reordered for clarity and easy of modification
        # forward mode is basically the default mode. Lot of main decision are made there.
        if Rover.mode == 'forward': 
            forward_mode(Rover)
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            stop_mode(Rover)
        elif Rover.mode == 'stuck':
            stuck_mode(Rover)
        elif Rover.mode == 'home_alone':
            home_alone(Rover)
        elif Rover.mode == 'rock_chasin':
            rock_chasin(Rover)
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

