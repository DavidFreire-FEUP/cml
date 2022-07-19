use rand::prelude::*;
use rand::distributions::StandardNormal;

#[derive(Clone)]
pub struct Particle{
    pub chromossome: Vec<f32>,
    wi: f32,
    wm: f32,
    wc: f32,
    pub fitness: i32,
    fitfunc: fn(&Vec<f32>) -> i32,
    mutation_rate: f32
}

impl Particle {
    pub fn new(chromossome: Vec<f32>, mutation_rate: &f32, wi:f32, wm:f32, wc:f32, fitfunc: fn(&Vec<f32>)->i32) -> Self {
        let mut ret = Self {
            chromossome,
            wi,
            wm,
            wc,
            fitness: 0,
            fitfunc,
            mutation_rate: *mutation_rate,
        };
        ret.estimate_fitness();
        return ret;
    }

    /// Introduces a mutation to the travel weights of a particle
    pub fn mutate(&mut self) {
        self.wi = self.wi * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wm = self.wm * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
        self.wc = self.wc * (1.0 + self.mutation_rate*SmallRng::from_entropy().sample(StandardNormal) as f32);
    }

    /// Calculates how fit the particle is according to it's genotype
    pub fn estimate_fitness(&mut self) {
        self.fitness = (self.fitfunc)(&self.chromossome);
    }
}

pub struct Swarm{
    particles: Vec<Particle>,
    ancestors: Vec<Particle>,
    best_ancestors: Vec<Particle>,
    pub global_best: Particle,
    gen: i32, // generation count
}

impl Swarm {
    pub fn new(
        size: &usize,
        chromossome_size: &usize,
        mutation_rate: &f32,
        wi: &f32,
        wm: &f32,
        wc: &f32,
        fitness_func: fn(&Vec<f32>) -> i32,
        ) -> Self {
        
        let mut particles = Vec::with_capacity(*size);
        let mut ancestors = Vec::with_capacity(*size);
        println!("Generating random particles and ancestors");
        let mut rng = rand::thread_rng();
        for i in 0..*size{
            let mut random_particle_chromossome: Vec<f32> = Vec::with_capacity(*chromossome_size);
            let mut random_ancestor_chromossome: Vec<f32> = Vec::with_capacity(*chromossome_size);
            for _ in 0..*chromossome_size{
                random_particle_chromossome.push(rng.gen());
                random_ancestor_chromossome.push(rng.gen());
            }

            particles.push(
                Particle::new(random_particle_chromossome, mutation_rate, *wi, *wm,* wc, fitness_func)
            );
            ancestors.push(
                Particle::new(random_ancestor_chromossome, mutation_rate, *wi, *wm,* wc, fitness_func)
            );
            println!("Particle nº {}", i);
        }
        println!("Done");

        let best_ancestors = ancestors.clone();

        let global_best = particles.iter().max_by_key(|particle| particle.fitness).unwrap().clone();

        Self { particles, ancestors, best_ancestors, global_best, gen: 1 }
    }

    /// Moves the particles around according to inertia, ancestors and global best
    pub fn travel(&mut self) {
        self.gen += 1;
        let mut new_particles: Vec<Particle> = Vec::with_capacity(self.particles.len()); 
        for i in 0..self.particles.len() {
            let mut new_chromossome: Vec<f32> = self.particles[i].chromossome.clone();
            for j in 0..self.particles[0].chromossome.len() {
                let mut deviation: f32;

                // Follow inertia
                deviation = 1.0/(self.gen as f32) * self.particles[i].wi * (self.particles[i].chromossome[j] - self.ancestors[i].chromossome[j]);

                // Follow best ancestor
                deviation += (SmallRng::from_entropy().sample(StandardNormal) as f32) * self.particles[i].wm*(self.best_ancestors[i].chromossome[j]-self.particles[i].chromossome[j]);

                // Follow global best
                deviation += (SmallRng::from_entropy().sample(StandardNormal) as f32)*self.particles[i].wc*(self.global_best.chromossome[j]-self.particles[i].chromossome[j]);
                
                new_chromossome[j] += deviation;
            }
            new_particles.push(Particle::new(new_chromossome, &self.particles[i].mutation_rate, self.particles[i].wi.clone(), self.particles[i].wm.clone(), self.particles[i].wc.clone(), self.particles[i].fitfunc.clone()));
        }

        // Update ancestors
        self.ancestors = self.particles.clone();
        
        // Update best ancestors
        for i in 0..self.best_ancestors.len() {
            if self.ancestors[i].fitness > self.best_ancestors[i].fitness {
                self.best_ancestors[i] = self.ancestors[i].clone();
            }
        }

        // Update particles
        self.particles = new_particles;

        // Update global best
        self.global_best = self.particles.iter().max_by_key(|particle| particle.fitness).unwrap().clone();
    }
    
    /// Create a mutated child for each particle and adds it to the swarm (duplicates population)
    pub fn reproduce(&mut self) {
        // Clone current particle population i.e. each particle has a "son"
        let mut children: Vec<Particle> = self.particles.clone();
        
        // Mutate the children so it becomes different than the father
        // The child will have it's fitness evaluated only after having a chance to move
        children.iter_mut().for_each(
            |son: &mut Particle| 
                son.mutate()
            );

        // Insert children into particle swarm
        self.particles.append(&mut children);
        
        // The children will have the same ancestors and best ancestors
        let mut ancestors_copy: Vec<Particle> = self.ancestors.clone();
        let mut best_ancestors_copy: Vec<Particle> = self.best_ancestors.clone();

        self.ancestors.append(&mut ancestors_copy);
        self.best_ancestors.append(&mut best_ancestors_copy);
    }

    /// Kills the worst half of the particle population
    pub fn select(&mut self) {
        let size = self.particles.len() / 2;
        for i in 0..size {
            // Replace parent for child if it performs better
            if self.particles[i+size].fitness > self.particles[i].fitness {
                self.particles[i] = self.particles[i+size].clone();
                self.ancestors[i] = self.ancestors[i+size].clone();
                self.best_ancestors[i] = self.best_ancestors[i+size].clone();
            }
        }

        // Kill the worst half
        self.particles.truncate(size);
        self.ancestors.truncate(size);
        self.best_ancestors.truncate(size);
    }
}