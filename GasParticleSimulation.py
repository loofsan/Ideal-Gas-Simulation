import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
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




################################################################################################################################

def init_list_random(N, radius, mass, boxsize):
    """Generate N Particle objects in a random way in a list."""
    particle_list = []

    for i in range(N):
        
        v_mag = np.random.rand(1)*6
        v_ang = np.random.rand(1)*2*np.pi
        v = np.append(v_mag*np.cos(v_ang), v_mag*np.sin(v_ang))
        
        collision = True
        while(collision == True):
            
            collision = False
            pos = radius + np.random.rand(2)*(boxsize-2*radius) 
            newparticle = Particle(mass, radius, pos, v)
            for j in range(len(particle_list)):

                collision = newparticle.handle_coll( particle_list[j] )

                if collision == True:
                    break

        particle_list.append(newparticle)
    return particle_list

# Create the same initial setup for particles and their simulation as before
particle_number = 60
boxsize = 200.0
tfin = 15
stepnumber = 70
timestep = tfin / stepnumber

particle_list = init_list_random(particle_number, radius=2, mass=1, boxsize=200)

# Compute simulation (for 150 steps)
for i in range(stepnumber):
    solve_step(particle_list, timestep, boxsize)

# Create the plot with circles for each particle and an empty histogram
fig, (ax, hist) = plt.subplots(1, 2, figsize=(12, 6))

ax.set_xlim([0, boxsize])
ax.set_ylim([0, boxsize])
ax.set_aspect("equal")

# Initialize particle representations
circle = [plt.Circle((p.solpos[0][0], p.solpos[0][1]), p.radius, ec="black", lw=1.5) for p in particle_list]
for c in circle:
    ax.add_patch(c)

# Create the histogram with the initial data
vel_mod = [p.solvel_mag[0] for p in particle_list]
hist.hist(vel_mod, bins=30, density=True, label="Simulation Data")
hist.set_xlabel("Speed")
hist.set_ylabel("Frequency Density")

# Initialize Maxwell–Boltzmann distribution
k = 1.38064852e-23
m = particle_list[0].mass

def total_Energy(particle_list, index):
    return sum(p.mass / 2.0 * p.solvel_mag[index] ** 2 for p in particle_list)

# Function to update the plot at each frame
def update(frame):
    # Update circle positions
    for j in range(particle_number):
        circle[j].center = particle_list[j].solpos[frame][0], particle_list[j].solpos[frame][1]

    # Clear the histogram and replot
    hist.clear()

    vel_mod = [p.solvel_mag[frame] for p in particle_list]

    # Customizing the histogram
    hist.hist(
        vel_mod,
        bins=12,  # Change number of bins
        density=True,
        color='skyblue',  # Change histogram color
        edgecolor='black',  # Change edge color
        alpha=0.7,  # Adjust transparency
        label="Simulation Data",
    )

    # Calculate mean and standard deviation
    mean_velocity = np.mean(vel_mod)
    std_velocity = np.std(vel_mod)


    # Set axis labels and title
    hist.set_xlabel("Speed")
    hist.set_ylabel("Frequency Density")
    hist.set_title("Particle Speed Distribution")

    # Add a legend
    hist.legend(loc="upper right")

    # Compute Maxwell–Boltzmann distribution
    m = particle_list[0].mass
    E = total_Energy(particle_list, frame)
    Average_E = E / len(particle_list)
    T = 2 * Average_E / (2* k)
    v = np.linspace(0, 10, 120)
    fv = m * np.exp(-m * v ** 2 / (2 * T * k)) * 2 * np.pi * v / (2 * np.pi * T * k)
    hist.plot(v, fv, label="Maxwell–Boltzmann distribution")
    hist.legend(loc="upper right")

# Set up animation with FuncAnimation
animation = FuncAnimation(fig, update, frames=stepnumber, interval=100)


# Show the plot with animation
plt.show()
