import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_BRAKE = 700 # brake torque required to maintain car stopped

class Controller(object):
	def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, 
				decel_limit, accel_limit, wheel_radius, wheel_base, 
				steer_ratio, max_lat_accel, max_steer_angle):
		# TODO: Implement
		self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
		## PID parameters for throttle
		Kp = 0.1
		Ki = 0.02
		Kd = 0.0
		mn = 0.0 #min throttle
		mx = 0.23 #max throttle
		
		## PID parameters for braking
		Kp_b = 60.0
		Ki_b = 0.0
		Kd_b = 10.0
		mn_b = 0.0 #min brake
		
		# setup two PIDs
		self.throttle_controller = PID(Kp, Ki, Kd, mn, mx)
		self.brake_controller = PID(Kp_b, Ki_b, Kd_b,mn_b,MAX_BRAKE)
		
		tau = 0.5 #1/(2pi*tau) = cutoff freq
		ts = 0.02 # sampling time, this should reflect refresh rate from dbw_node
		self.vel_lpf = LowPassFilter(tau, ts)
		
		self.vehicle_mass = vehicle_mass
		self.fuel_capacity = fuel_capacity
		self.brake_deadband = brake_deadband
		self.decel_limit = decel_limit
		self.accel_limit = accel_limit
		self.wheel_radius = wheel_radius
		self.last_time = rospy.get_time()

	def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
		# TODO: Change the arg, kwarg list to suit your needs
		# Return throttle, brake, steer
		
		if not dbw_enabled:
			self.throttle_controller.reset()
			self.brake_controller.reset()
			return 0.0, 0.0, 0.0
		
		current_vel = self.vel_lpf.filt(current_vel)
		
		steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
		#rospy.loginfo('test:%s\n', linear_vel)
		vel_error = linear_vel - current_vel
		self.last_vel = current_vel
		
		current_time = rospy.get_time()
		sample_time = current_time - self.last_time
		self.last_time = current_time
		
		throttle = self.throttle_controller.step(vel_error, sample_time)
		brake = 0
		# when car is supposed to stop and current velocity is small, throw on max brake
		if linear_vel < 0.4 and current_vel < 0.4:
			throttle = 0
			brake = MAX_BRAKE
		# when car is going faster than target speed, let go off throttle and apply brake
		elif throttle < 0.1 and vel_error < 0:
			throttle = 0
			brake = self.brake_controller.step(-vel_error, sample_time)
		
		return throttle, brake, steering
