import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 1
initial_conditions = np.array(np.mat('0.01;0.01;0'))
#initial_conditions = np.array(np.mat('1 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

# Define goal points by removing orientation from poses - modified to lie outside the shown area
arrayExtra = np.array([2.5, 0, 0])
arrayExtra.shape = (3,1)
goal_points = arrayExtra
#goal_points = generate_initial_conditions(N, width=r.boundaries[2]-2*r.robot_diameter, height = r.boundaries[3]-2*r.robot_diameter, spacing=0.5)

# Create unicycle pose controller
unicycle_pose_controller = create_hybrid_unicycle_pose_controller()

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate()

# define x initially
x = r.get_poses()

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2
robot_marker_size_m = 0.15
marker_size_goal = determine_marker_size(r,goal_marker_size_m)
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
font_size = determine_font_size(r,0.1)
line_width = 5

# Create Goal Point Markers
#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Arrow for desired orientation
goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.02, length_includes_head=True, color = CM[ii,:], zorder=-2)
for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]
robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width) 
for ii in range(goal_points.shape[1])]


# ----  choose lane radius and the number of simulation steps
laneRadius      = float(0.8)
countMax        = 1500           # integer number of steps

# ----  draw the circular track 
circle = plt.Circle( (0,0),  laneRadius, color='r',fill = False, linewidth = line_width)
r.axes.add_artist(circle)

# ----  parameters of the control law
maxLinearSpeed  = float(0.2) # extracted from line 47 of robotarium_abc.py
maxAngularSpeed = float(3.9) # extracted from line 48 of robotarium_abc.py
linearSpeed     = float(0.15)
if ( linearSpeed * linearSpeed > maxLinearSpeed * maxLinearSpeed ):
    linearSpeed = maxLinearSpeed
kProportional   = float(1)
kDerivative     = float(0.3)
kCurvature      = linearSpeed / laneRadius 
smallRadius     = laneRadius / float(10)


#  initialize the simulation
r.step()

count = 0
while ( (count < countMax)):

    # Get poses of agents
    x = r.get_poses()

    # -------------- retain this from here
    #Update Plot
    # Update Robot Marker Plotted Visualization
    for i in range(x.shape[1]):
        robot_markers[i].set_offsets(x[:2,i].T)
        # This updates the marker sizes if the figure window size is changed. 
        # This should be removed when submitting to the Robotarium.
#        robot_markers[i].set_sizes([determine_marker_size(r, robot_marker_size_m)])

    #  is needed for plotting
    for j in range(goal_points.shape[1]):
        goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])

    # Create unicycle control inputs -- is needed for plotting
    dxu = unicycle_pose_controller(x, goal_points)

    # Create safe control inputs (i.e., no collisions)
    dxu = uni_barrier_cert(dxu, x)
    # -------------- up to here

    # increment simulation step count
    count = count + 1

    # calculate present location and orientation
    xRob     = float(x.item(0))
    yRob     = float(x.item(1))
    psiRob   = float(x.item(2))
    rRob     = np.sqrt( float( xRob * xRob + yRob * yRob )  )

    #    if the robot is too close to the origin
    if ( rRob <= smallRadius ):
        thetaRob = psiRob   
    #    otherwise
    else:
        thetaRob = np.arctan2( yRob , xRob )
    # and the orientation error is:
    dpsi     = psiRob - thetaRob - np.pi / float (2)
    # and the lateral error is
    dy       = laneRadius - rRob

    # calculate and set controls (set angular velocity)
    propTerm  = - kProportional * dy
    derivTerm = - kProportional * kDerivative * np.sin(dpsi)
    curvTerm  =   kCurvature
    angularSpeed = propTerm + derivTerm + curvTerm
    # now ensure that angular speed falls within limit
    if (angularSpeed > maxAngularSpeed ):
        angularSpeed = maxAngularSpeed
    elif (angularSpeed < -1 *maxAngularSpeed ):
        angularSpeed = -1 * maxAngularSpeed

    #  now set the velocity as a row vector ( row vector = 1 * 2 array)
    vel = np.array([ linearSpeed ,  angularSpeed ])
    #  now reshape as a column vector ( column vector = 2 * 1 array)
    vel.shape = (2,1)
    # Set the velocity command
    r.set_velocities(np.arange(N), vel)

    # Iterate the simulation
    r.step()

print(count)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
