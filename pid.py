import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None
        
    def update(self, measurement, dt=None):
        current_time = time.time()
        if self._last_time is None:
            self._last_time = current_time
            dt = 0.0
        elif dt is None:
            dt = current_time - self._last_time
            self._last_time = current_time
        
        error = self.setpoint - measurement
        p_term = self.kp * error
        self._integral += error * dt
        i_term = self.ki * self._integral
        derivative = (error - self._last_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        
        if self.output_limits:
            min_out, max_out = self.output_limits
            if output < min_out or output > max_out:
                self._integral -= error * dt
            output = max(min_out, min(max_out, output))
        
        self._last_error = error
        return output
    
    def reset(self):
        self._last_error = self._integral = 0.0
        self._last_time = None
    
    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None: self.kp = kp
        if ki is not None: self.ki = ki
        if kd is not None: self.kd = kd
    
    def set_setpoint(self, setpoint):
        self.setpoint = setpoint