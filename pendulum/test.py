
import numpy as np
import time
import math
import pygame; pygame.init()


class Config:
    width, height = 800, 600
    title = "Pendulum Simulation"
    
    class Colors:
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)


class Pendulum:
    def __init__(self, 
            center=np.array([10,10]),
            radius=1,
            force=np.array([0,0]),
            point=np.array([10, 10]),
            mass=1.0,
            graivty=9.81,
            damping=0.99999999999,
        ):
        
        self.center = center
        self.radius = radius
        self.force = force
        self.point = point
        self.velocity = np.array([0, 0])
        self.mass = mass
        self.gravity = graivty
        self.damping = damping
        

    def update(self, dt,  additional_force=np.array([0.00001, 0.00001])):
        self.point, self.velocity = self.update_pendulum(
            self.point,
            self.center,
            self.radius,
            additional_force,
            self.velocity,
            self.mass,
            self.gravity,
            dt,
            self.damping,
        )

    @staticmethod
    def update_pendulum(
        point: np.ndarray,
        center: np.ndarray,
        radius: float,
        force: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        gravity: float,
        dt: float,
        damping_coefficient: float = 0.1
    ):
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        
        
        gravity_acceleration = -gravity * np.sin(angle)
        applied_acceleration = force / mass
        
        friction_force = -damping_coefficient * velocity
        friction_acceleration = friction_force / mass
        
        total_acceleration = gravity_acceleration + applied_acceleration + friction_acceleration
        
        new_velocity = velocity + total_acceleration * dt
        
        new_point = point + new_velocity * dt
        
        distance = np.linalg.norm(new_point - center)
        new_point = center + (new_point - center) * radius / distance
        
        return new_point, new_velocity


class CustomWindow:
    def __init__(self):
        self.window = pygame.display.set_mode((Config.width, Config.height))
        pygame.display.set_caption(Config.title)
        
        self.pendulum = Pendulum(
            center=(Config.width // 2, Config.height // 2),
            radius=100,
            mass=50, graivty=100
        )

    def update(self,dt):
        self.pendulum.update(dt)
        print(self.pendulum.velocity)

    def draw_pendulum(self, pendulum: Pendulum):
        pygame.draw.line(self.window, Config.Colors.BLACK, pendulum.center, pendulum.point, 2)
        pygame.draw.circle(self.window, Config.Colors.RED, pendulum.point, 1)

    def flip(self):
        self.window.fill(Config.Colors.WHITE)
        self.draw_pendulum(self.pendulum)
        pygame.display.flip()

    def mainloop(self):
        running = True
        
        target_hz = 1/120
        dt = 0
        loop_start = time.time()
        iteration = 0 

        while running:
            iter_start = time.time()
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Logic
            self.update(dt)
            self.flip()
            
            # Metadata
            iter_end = time.time()
            dt = iter_elapsed = (iter_end - iter_start)    
            time_to_sleep = target_hz - iter_elapsed
            time.sleep(time_to_sleep)
            iteration += 1

CustomWindow().mainloop()