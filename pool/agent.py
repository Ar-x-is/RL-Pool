import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy
random.seed(73)

class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.ns = utils.NextState()


    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius


    def action(self, ball_pos=None):
        time.sleep(5)
        ## Code you agent here ##
        ## You can access data from config.py for geometry of the table, configuration of the levels, etc.
        ## You are NOT allowed to change the variables of config.py (we will fetch variables from a different file during evaluation)
        ## Do not use any library other than those that are already imported.
        ## Try out different ideas and have fun!

        num_solids = len(ball_pos) - 2 # number of solid balls on table
        cue_ball = ball_pos["white"]
        solid_balls = ball_pos.copy()
        solid_balls.pop(0)
        solid_balls.pop("white")

        # real_holes = numpy.array(self.holes)
        # # print(real_holes)
        # # Mirror along x=0, y=0, x=1000, and y=500
        # mirrored_x0 = numpy.array([-real_holes[:][0], real_holes[:][1]]).T
        # # mirrored_x0 = numpy.where(mirrored_x0 == 0, real_holes, mirrored_x0)
        # mirrored_y0 = numpy.array([real_holes[:][0], -real_holes[:][1]]).T
        # # mirrored_y0 = numpy.where(mirrored_y0 == 0, real_holes, mirrored_y0)
        # mirrored_x1000 = numpy.array([2000 - real_holes[:][0], real_holes[:][1]]).T
        # # mirrored_x1000 = numpy.where(mirrored_x1000 == 0, real_holes, mirrored_x1000)
        # mirrored_y500 = numpy.array([real_holes[:][0], 1000 - real_holes[:][1]]).T
        # # mirrored_y500 = numpy.where(mirrored_y500 == 0, real_holes, mirrored_y500)
        # # Concatenate the original and mirrored arrays
        # holes = numpy.concatenate((real_holes, mirrored_x0, mirrored_y0, mirrored_x1000, mirrored_y500), axis=0)

        # find the best ball and hole to pot
        ball, hole = find_best(cue_ball, solid_balls, self.holes)

        # find the angle at which the cue ball needs to hit the min_ball to pot it
        angle = find_angle(cue_ball, ball, hole, ball_pos, self.ns, num_solids)

        # Using self.ns.next_state, find the best velocity to hit the cue ball with
        v = find_best_velocity(ball_pos, angle, self.ns, ball, hole)
        return (angle, v)
    
def find_best_velocity(ball_pos, angle, ns, ball, hole):
    return 0.67

def find_best(cue_ball, solid_balls, holes):

    # a matrix that represents for each ball, hole pair whether there is another ball in the way 
    uninterrupted = numpy.zeros((len(solid_balls), len(holes)))
    # a matrix that represents for each ball, hole pair the straightness of the path
    straightness = numpy.zeros((len(solid_balls), len(holes)))

    proximity_of_cue_to_solid = numpy.zeros(len(solid_balls))
    proximity_of_solids_to_hole = numpy.zeros((len(solid_balls), len(holes)))

    for i in range(len(solid_balls.keys())):
        proximity_of_cue_to_solid[i] = distance(cue_ball, solid_balls[list(solid_balls.keys())[i]])
        for j in range(len(holes)):
            proximity_of_solids_to_hole[i][j] = distance(solid_balls[list(solid_balls.keys())[i]], holes[j])
            key = list(solid_balls.keys())[i]
            uninterrupted[i][j] = not are_points_on_line(solid_balls[key], holes[j], solid_balls)
            straightness[i][j] = abs(calculate_straightness(cue_ball, solid_balls[key], holes[j]))

    k1 = 0
    k2 = 0
    weights = numpy.zeros((len(solid_balls), len(holes)))
    for i in range(len(solid_balls.keys())):
        for j in range(len(holes)):
            weights[i][j] = activation_potential(straightness[i][j]) + \
            (1 - math.tanh(k1*1e-5*proximity_of_cue_to_solid[i]*proximity_of_cue_to_solid[i])) + \
            (1- math.tanh(k2*1e-5*proximity_of_solids_to_hole[i][j]*proximity_of_solids_to_hole[i][j]))

    # the uninterrupted matrix serves as a mask for the weights matrix
    masked_weights = numpy.multiply(uninterrupted, weights)
    # print("straightness: ", straightness)
    # print("masked_weights: ", masked_weights)
    ball_idx, hole = numpy.unravel_index(masked_weights.argmax(), masked_weights.shape)
    ball = list(solid_balls.keys())[ball_idx]
    # print("ball:", ball)
    return solid_balls[ball], holes[hole]    

def calculate_straightness(cue_ball, solid_ball, hole):
    theta_1 = math.atan2(solid_ball[1] - cue_ball[1], solid_ball[0] - cue_ball[0])
    theta_2 = math.atan2(hole[1] - solid_ball[1], hole[0] - solid_ball[0])

    delta_y = solid_ball[1] - cue_ball[1]
    delta_x = solid_ball[0] - cue_ball[0]

    straightness = (math.pi + theta_1 - theta_2)
    if straightness > math.pi:
        straightness = 2*math.pi - straightness 
    return straightness

def activation_potential(x):
    return 1 + math.tanh(x - 2*math.pi/3)

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Calculate the distance between point1 (x1, y1) and point2 (x2, y2)
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def distance_to_line(point1, point2, point):
    x1, y1 = point1
    x2, y2 = point2
    x, y = point
    # Calculate the shortest distance between point (x, y) and the line defined by (x1, y1) and (x2, y2)
    A = y2 - y1
    B = x1 - x2
    C = (x1 * y2) - (x2 * y1)
    distance = abs((A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2))
    return distance

def are_points_on_line(point1, point2, point_list):
    # Check if any of the points in point_list are on the line defined by point1 and point2
    points = point_list.copy()
    keys = {i for i in point_list if numpy.all(point_list[i] == point1) or numpy.all(point_list[i] == point2)}
    for i in keys:
        points.pop(i)
    for point in points:
        distance = distance_to_line(point1, point2, points[point])
        if distance <= 2*config.ball_radius:
            return True
    return False

def find_angle(cue_ball, solid_ball, hole, ball_pos, ns, num_solids):
    cx, cy = cue_ball
    bx, by = solid_ball
    hx, hy = hole

    angle_solid_to_hole = math.atan2(hy - by, hx - bx)    
    angle_cue_to_solid = math.atan2((by - cy) + (by-hy)/distance(solid_ball, hole)*2*config.ball_radius, 
                                    (bx - cx) + (bx-hx)/distance(solid_ball, hole)*2*config.ball_radius)

    angle = - math.pi/2 - angle_cue_to_solid
    if (by-cy > 0 and bx-cx < 0):
        angle = angle + 2*math.pi
    return angle/math.pi
