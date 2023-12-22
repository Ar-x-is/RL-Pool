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
        self.weights = numpy.array([1,-1])


    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius


    def action(self, ball_pos=None):
        # time.sleep(5)
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

        # update the linear approximation of the value function
        if self.curr_iter > 0:
            prev_state = self.state_dict[list(self.state_dict.keys())[-1]]
            reward = get_reward(prev_state, self.prev_action, ball_pos, self.weights, solid_balls, self.holes)
            self.weights = update_weights(reward, self.prev_action, self.state_dict, self.curr_iter)

        # get the feature vector od the current state
        features = get_features(ball_pos, self.holes)

        # get the action values for each ball, hole pair
        action_values = get_action_values(features, self.weights, solid_balls, self.holes)

        # uniformly choose among the balls if none of the balls can be potted
        if numpy.all(action_values == 0):
            print("randomised choice")
            ball_key, hole_idx = numpy.random.choice(list(solid_balls.keys())), numpy.random.choice(numpy.arange(0,len(self.holes)))
        else:
            ball_key, hole_idx = list(solid_balls.keys())[int(numpy.argmax(action_values)/6)], numpy.argmax(action_values)%6

        # print(self.holes)
        # print(ball_key, hole_idx)
        # print(features)
        # print(action_values)
        # time.sleep(20)

        ball = solid_balls[ball_key]
        hole = self.holes[hole_idx]

        # find the angle at which the cue ball needs to hit the min_ball to pot it
        angle = find_angle(cue_ball, ball, hole, ball_pos, self.ns, num_solids)

        # find the best speed at which to hit the cue ball
        speed = find_best_velocity(cue_ball, angle, self.ns, ball, hole)

        return (angle, speed)
    
def get_features(state, holes):
    # make a feature vector for each ball, hole pair
    cue_ball = state["white"]
    solid_balls = state.copy()
    solid_balls.pop(0)
    solid_balls.pop("white")
    features = numpy.zeros((7, len(holes), 2))
    for i in solid_balls.keys():
        for j in range(len(holes)):
            # feature 1: straightness of the shot
            features[i-1][j][0] = calculate_straightness(cue_ball, solid_balls[i], holes[j])
            # feature 2: distance from cue ball to solid ball
            features[i-1][j][1] = distance(cue_ball, solid_balls[i])/500
            # 0 if solid to hole path is obstructed
            if are_points_on_line(solid_balls[i], holes[j], solid_balls):
                features[i-1][j] = features[i-1][j] * 0
            if abs(features[i-1][j][0]/math.pi) < 0.45:
                features[i-1][j] = features[i-1][j] * 0
    return features
    
def get_action_values(features, weights, solid_balls, holes):
    action_values = numpy.zeros((len(solid_balls), len(holes)))
    n = len(solid_balls)
    m = len(holes)
    for i in range(n):
        for j in range(m):
            action_values[i][j] = numpy.dot(features[i][j], weights)
    return action_values

def get_reward(prev_state, prev_action, curr_state, weights, solid_balls, holes):
    reward = 10 * (len(prev_state) - len(curr_state))
    reward = reward + (numpy.max(get_action_values(get_features(curr_state), weights, solid_balls, holes))
                       - numpy.max(get_action_values(get_features(prev_state), weights, solid_balls, holes)))
    return reward

def update_weights(reward, prev_action, state_dict, curr_iter):
    alpha = 1/curr_iter # learning rate
    gamma = 0.9 # discount factor (lookahead)
    old_features = get_features(state_dict[list(state_dict.keys())[-1]])
    new_features = get_features(state_dict[list(state_dict.keys())[-2]])
    weights = weights + alpha * (reward + gamma * numpy.dot(weights, new_features) - numpy.dot(weights, old_features)) * old_features
    return weights

def find_best_velocity(ball_pos, angle, ns, ball, hole):
    return 0.5

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
    cx, cy = cue_ball
    bx, by = solid_ball
    hx, hy = hole

    theta_2 = math.atan2(hy - by, hx - bx)    
    theta_1 = math.atan2((by - cy) + (by-hy)/distance(solid_ball, hole)*2*config.ball_radius, 
                        (bx - cx) + (bx-hx)/distance(solid_ball, hole)*2*config.ball_radius)

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

def distance_to_line_segment(point1, point2, point):
    x1, y1 = point1
    x2, y2 = point2
    x, y = point
    # Step 1: Calculate the slope of line segment PQ
    m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else math.inf

    # Step 2: Determine the equation of the line passing through P and Q
    def line_equation(x_val):
        return m * (x_val - x1) + y1

    # Step 3: Find the perpendicular line passing through A
    m_perpendicular = -1 / m if m != 0 else math.inf

    # Step 4: Find the intersection point I of the two lines
    x_i = (m * x1 - m_perpendicular * x + y - y1) / (m - m_perpendicular)
    y_i = line_equation(x_i)

    # Step 5: Check if I lies on the line segment PQ
    if x1 <= x_i <= x2 or x2 <= x_i <= x1:
        # Distance is the distance between A and I
        distance = math.sqrt((x - x_i)**2 + (y - y_i)**2)
    else:
        # # Distance is the minimum of the distances from A to P and A to Q
        # distance_p = math.sqrt((x - x1)**2 + (y - y1)**2)
        # distance_q = math.sqrt((x - x2)**2 + (y - y2)**2)
        # distance = min(distance_p, distance_q)

        # this case if the ball lies on the line connecting solid and hole but not between them
        distance = math.inf

    return distance

def are_points_on_line(point1, point2, point_list):
    # Check if any of the points in point_list are on the line defined by point1 and point2
    points = point_list.copy()
    keys = {i for i in point_list if numpy.all(point_list[i] == point1) or numpy.all(point_list[i] == point2)}
    for i in keys:
        points.pop(i)
    for point in points:
        distance = distance_to_line_segment(point1, point2, points[point])
        if distance <= 3*config.ball_radius:
            return True
    return False

def find_angle(cue_ball, solid_ball, hole, ball_pos, ns, num_solids):
    cx, cy = cue_ball
    bx, by = solid_ball
    hx, hy = hole

    angle_solid_to_hole = math.atan2(hy - by, hx - bx)    
    angle_cue_to_solid = math.atan2((by - cy) + (by-hy)/distance(solid_ball, hole)*2*config.ball_radius, 
                                    (bx - cx) + (bx-hx)/distance(solid_ball, hole)*2*config.ball_radius)

    angle = angle_cue_to_solid - math.pi/2
    if (angle < -math.pi):
        angle = angle + 2*math.pi
    return angle/math.pi
