import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats as stats

class Particle:
    """Define physics of elastic collision."""
    
    def __init__(self, mass, radius, position, velocity):
        """Initialize a Particle object
        
        mass the mass of particle
        radius the radius of particle
        position the position vector of particle
        velocity the velocity vector of particle
        """
        self.mass = mass
        self.radius = radius
        
        # last position and velocity
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        
        # all position and velocities recorded during the simulation
        self.solpos = [np.copy(self.position)]
        self.solvel = [np.copy(self.velocity)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocity))]
        
    def compute_step(self, step):
        """Compute position of next step."""
        self.position += step * self.velocity
        self.solpos.append(np.copy(self.position)) 
        self.solvel.append(np.copy(self.velocity)) 
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocity))) 
    
    def handle_coll(self, particle):
        """Check if there is a collision with another particle."""
        
        r1, r2 = self.radius, particle.radius
        x1, x2 = self.position, particle.position
        di = x2-x1
        norm = np.linalg.norm(di)
        if norm-(r1+r2)*1.1 < 0:
            return True
        else:
            return False

    
    def resolve_coll(self, particle, step):
        """Compute velocity after collision with another particle."""
        m1, m2 = self.mass, particle.mass
        r1, r2 = self.radius, particle.radius
        v1, v2 = self.velocity, particle.velocity
        x1, x2 = self.position, particle.position
        di = x2-x1
        norm = np.linalg.norm(di)
        if norm-(r1+r2)*1.1 < step*abs(np.dot(v1-v2, di))/norm:
            self.velocity = v1 - 2. * m2/(m1+m2) * np.dot(v1-v2, di) / (np.linalg.norm(di)**2.) * di
            particle.velocity = v2 - 2. * m1/(m2+m1) * np.dot(v2-v1, (-di)) / (np.linalg.norm(di)**2.) * (-di)
            

    def compute_refl(self, step, size):
        """Compute velocity after hitting an edge.
        step the computation step
        size the medium size
        """
        r, v, x = self.radius, self.velocity, self.position
        projx = step*abs(np.dot(v,np.array([1.,0.])))
        projy = step*abs(np.dot(v,np.array([0.,1.])))
        if abs(x[0])-r < projx or abs(size-x[0])-r < projx:
            self.velocity[0] *= -1
        if abs(x[1])-r < projy or abs(size-x[1])-r < projy:
            self.velocity[1] *= -1.


def solve_step(particle_list, step, size):
    """Solve a step for every particle."""
    
    # Detect edge-hitting and collision of every particle
    for i in range(len(particle_list)):
        particle_list[i].compute_refl(step,size)
        for j in range(i+1,len(particle_list)):
                particle_list[i].resolve_coll(particle_list[j],step)    

                
    # Compute position of every particle  
    for particle in particle_list:
        particle.compute_step(step)




# Create the same initial setup for particles and their simulation as before
particle_number_1 = 60
particle_number_2 = 40
boxsize = 200.0
tfin = 15
stepnumber = 70
timestep = tfin / stepnumber

# Define a function to initialize a list of particles with the given parameters
def init_list_random(N, radius, mass, boxsize):
    """Generate N Particle objects in a random way in a list."""
    particle_list = []
    for i in range(N):
        v_mag = np.random.rand(1) * 6
        v_ang = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))
        
        collision = True
        while collision:
            collision = False
            pos = radius + np.random.rand(2) * (boxsize - 2 * radius)
            new_particle = Particle(mass, radius, pos, v)
            for existing_particle in particle_list:
                if new_particle.handle_coll(existing_particle):
                    collision = True
                    break

            if not collision:
                particle_list.append(new_particle)
    return particle_list

# Initialize the particle lists for two different simulations
particle_list_1 = init_list_random(particle_number_1, radius=2, mass=1, boxsize=boxsize)
particle_list_2 = init_list_random(particle_number_2, radius=3, mass=5, boxsize=boxsize)

# Compute simulations
for _ in range(stepnumber):
    solve_step(particle_list_1, timestep, boxsize)
    solve_step(particle_list_2, timestep, boxsize)

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# Set up the first box (particle simulation with radius 2, mass 1)
ax1.set_xlim([0, boxsize])
ax1.set_ylim([0, boxsize])
ax1.set_aspect("equal")
circle_1 = [plt.Circle((p.solpos[0][0], p.solpos[0][1]), p.radius, ec="black", lw=1.5) for p in particle_list_1]
for c in circle_1:
    ax1.add_patch(c)

# Set up the second box (Maxwell–Boltzmann distribution for the first simulation)
vel_mod_1 = [p.solvel_mag[0] for p in particle_list_1]
ax2.hist(vel_mod_1, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel("Speed")
ax2.set_ylabel("Frequency Density")

# Set up the third box (particle simulation with radius 3, mass 5)
ax3.set_xlim([0, boxsize])
ax3.set_ylim([0, boxsize])
ax3.set_aspect("equal")
circle_2 = [plt.Circle((p.solpos[0][0], p.solpos[0][1]), p.radius, ec="black", lw=1.5) for p in particle_list_2]
for c in circle_2:
    ax3.add_patch(c)

# Set up the fourth box (Maxwell–Boltzmann distribution for the second simulation)
vel_mod_2 = [p.solvel_mag[0] for p in particle_list_2]
ax4.hist(vel_mod_2, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
ax4.set_xlabel("Speed")
ax4.set_ylabel("Frequency Density")

# Create the Maxwell–Boltzmann distribution function
def maxwell_boltzmann(mass, temperature, v):
    k = 1.38064852e-23  # Boltzmann constant
    return mass * np.exp(-mass * v ** 2 / (2 * temperature * k)) * 2 * np.pi * v / (2 * np.pi * temperature * k)

# Compute the Maxwell–Boltzmann distribution for the second box
m1 = particle_list_1[0].mass
energy_1 = sum(p.mass / 2.0 * p.solvel_mag[0] ** 2 for p in particle_list_1)
avg_energy_1 = energy_1 / len(particle_list_1)
temperature_1 = 2 * avg_energy_1 / (2 * 1.38064852e-23)
v = np.linspace(0, 10, 120)
fv_1 = maxwell_boltzmann(m1, temperature_1, v)
ax2.plot(v, fv_1, label="Maxwell–Boltzmann distribution")
ax2.legend(loc="upper right")

# Compute the Maxwell–Boltzmann distribution for the fourth box
m2 = particle_list_2[0].mass
energy_2 = sum(p.mass / 2.0 * p.solvel_mag[0] ** 2 for p in particle_list_2)
avg_energy_2 = energy_2 / len(particle_list_2)
temperature_2 = 2 * avg_energy_2 / (2 * 1.38064852e-23)
fv_2 = maxwell_boltzmann(m2, temperature_2, v)
ax4.plot(v, fv_2, label="Maxwell–Boltzmann distribution")
ax4.legend(loc="upper right")

# Function to update the animation frames
def update(frame):
    # Update the positions of particles in the first and third boxes
    for j in range(particle_number_1):
        circle_1[j].center = particle_list_1[j].solpos[frame][0], particle_list_1[j].solpos[frame][1]

    for j in range(particle_number_2):
        circle_2[j].center = particle_list_2[j].solpos[frame][0], particle_list_2[j].solpos[frame][1]

    # Update the second box (re-draw histogram and Maxwell–Boltzmann distribution)
    ax2.clear()
    vel_mod_1 = [p.solvel_mag[frame] for p in particle_list_1]
    ax2.hist(
        vel_mod_1, bins=12, density=True, color='skyblue', edgecolor='black', alpha=0.7, label="Simulation Data"
    )
    ax2.set_xlabel("Speed")
    ax2.set_ylabel("Frequency Density")
    ax2.plot(v, fv_1, label="Maxwell–Boltzmann distribution")
    ax2.legend(loc="upper right")

    # Update the fourth box (re-draw histogram and Maxwell–Boltzmann distribution)
    ax4.clear()
    vel_mod_2 = [p.solvel_mag[frame] for p in particle_list_2]
    ax4.hist(
        vel_mod_2, bins=12, density=True, color='skyblue', edgecolor='black', alpha=0.7, label="Simulation Data"
    )
    ax4.set_xlabel("Speed")
    ax4.set_ylabel("Frequency Density")
    ax4.plot(v, fv_2, label="Maxwell–Boltzmann distribution")
    ax4.legend(loc="upper right")

# Set up animation with FuncAnimation
animation = FuncAnimation(fig, update, frames=stepnumber, interval=100)

# Display the plot with animation
plt.show()
