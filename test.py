import turtle,random,time,csv
from tensorforce import Agent, Environment, Runner
import numpy as np
import os
import math

class Ant(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.speed("normal")
        self.color('red')
        self.penup()
        self.type = "insect"
        self.diet = "all"
        self.eat_dist = 20
        self.vision_dist = 200
        scale = 0.075
        with open('coordinator.csv', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            coord_list = []
            for row in reader:
                row = [round(float(c)*scale) for c in row]
                coord_list.append(row)
            xmov = np.average([pair[0] for pair in coord_list])
            ymov = np.average([pair[1] for pair in coord_list])
            for row in coord_list:
                row[0]-=xmov
                row[1]-=ymov
            coord_tuple = tuple([tuple(row) for row in coord_list])
        self.screen.register_shape("ant", coord_tuple)
        self.shape('ant')
        self.tiltangle(180)

class Food(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape('circle')
        self.speed(0)
        self.color('green')
        self.penup()
        self.type = "all"

class AntEnvironment(Environment):
    gridMax = 10
    movestep = 1

    def __init__(self,width=500,height=500):
        super().__init__()
        self.root = turtle.Screen()
        self.root.bgcolor('black')
        self.root.tracer(0)
        self.root.setup(width=width,height=height)
        self.width=width-20
        self.height=height-20
        self.score=0
        self.episodes=0
        self.delay = 0

        self.ants = []
        ant1 = Ant()
        self.ants.append(ant1)

        self.foods = []
        food1 = Food()
        self.foods.append(food1)

    def states(self):
        return dict(type='float', shape=(4,))

    def actions(self):
        return dict(type='bool', shape=(3,))

    def num_actors(self):
        return super().num_actors()

    def reset(self):
        min_distance_from_center = 100
        while True:
            x = random.randint(-self.width/2, self.width/2)
            y = random.randint(-self.height/2, self.height/2)
            if abs(x) >= min_distance_from_center and abs(y) >= min_distance_from_center:
                break
        self.foods[0].goto(x,y)
        self.ants[0].home()
        self.score=0
        self.episodes+=1
        return self.collectStates()

    def execute(self,actions):
        # reward = -0.01
        terminal = False
        for ant in self.ants:
            if actions[0]:
                self.forward(ant)
                self.checkMove(ant)
            if actions[1]:
                self.left(ant)
                self.checkMove(ant)
            if actions[2]:
                self.right(ant)
                self.checkMove(ant)
        states = self.collectStates()
        reward = self.reward()
        self.update(reward)
        return states, terminal, reward

    def update(self,reward):
        self.score+=reward
        time.sleep(self.delay)
        if self.delay>0 or 1:
            self.root.update()
            self.root.title(f"Episode: {self.episodes}, Score: {str(self.score)}")

    def collectStates(self):
        antx,anty = self.ants[0].position()
        anth = self.ants[0].heading()
        dist = self.ants[0].distance(self.foods[0])
        states = np.array([antx,anty,anth,dist])
        return states

    def checkFood(self,ant:Ant):
        reward = 0
        food:Food
        for food in self.foods:
            dist = ant.distance(food)
            reward += -0.001*dist
            dx = food.xcor() - ant.xcor()
            dy = food.ycor() - ant.ycor()
            if dist < ant.vision_dist:
                angle = math.degrees(math.atan2(dy, dx)) - ant.heading()
                reward += 0.1 * math.cos(math.radians(angle))
            if dist<ant.eat_dist:
                reward+=1000
                x = random.randint(-self.width/2,self.width/2)
                y = random.randint(-self.width/2,self.height/2)
                food.goto(x,y)
                break
        return reward

    def checkThreats(self,ant):
        reward = 0
        return reward

    def reward(self):
        reward = 0
        for ant in self.ants:
            reward+=self.checkFood(ant)
            reward+=self.checkThreats(ant)
        return reward

    ### Actions ###
    def forward(self,trtle: turtle.Turtle):
        trtle.forward(10)

    def left(self,trtle: turtle.Turtle):
        trtle.left(10)

    def right(self,trtle: turtle.Turtle):
        trtle.right(10)

    def checkMove(self,trtle: turtle.Turtle):
        x, y = trtle.position()
        if -self.width/2 < x < self.width/2 and -self.height/2 < y < self.height/2:
            return
        trtle.undo()

if __name__=="__main__":
    random.seed()
    environment = Environment.create(
        environment=AntEnvironment, max_episode_timesteps=1500
    )
    tfagent = Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', 
        reward_estimation=dict(
            horizon=20,
            ),
        state_preprocessing=dict(
            type='sequence',
            length = 4
            ),
    )
    tfagent = Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient',
        reward_estimation=dict(
            horizon=20
            )
    )

    environment.delay = 0

    environment.episodes=0

    runner = Runner(
        agent=tfagent,
        environment=environment,
    )

    environment.max_episode_timesteps = 10000
    runner.run(num_episodes=100)
