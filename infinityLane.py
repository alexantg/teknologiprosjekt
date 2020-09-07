import rps.robotarium as robotarium
import math
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *


import numpy as np
import time

rb= RobotariumBuilder();

#Number for robots
N=1

#initializing the robot/object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)


#max number of itterations
itterations= 1000;

#Specifyes the center of the circle
center= [0,0];

radius= 0.5

error_margin= 0.02;

#These waypoints get spread 
numberOfWaypoints= 100;

th_vec= linespace(0,2*pi, numberOfWaypoints);
waypoints = [radius.*cos(th_vec);radius.*sin(th_vec)];

current_index= 1; 

controller= position_controller();

for i= 1 : itterations

x=r.get_states();

if norm(x(1:2 - waypoints( :,current_index))) <= error_margin 

current_index= mod(current_index, numberOfWaypoints) +1:

end

velocity = controller(x(1:2), waypoints(:,current_index));

r.set_inputs(1,velocity);

r.step();

end

r.call_at_scripts_end();








